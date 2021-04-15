from thetis import create_directory, print_output

import argparse
import numpy as np
import os
from time import perf_counter

from adapt_utils.swe.turbine.solver import AdaptiveTurbineProblem
from adapt_utils.unsteady.test_cases.turbine_array.options import TurbineArrayOptions


# --- Parse arguments

parser = argparse.ArgumentParser()
parser.add_argument("-num_meshes", help="Number of meshes (for debugging)")
parser.add_argument("-load_mesh", help="Load meshes from a previous run")
parser.add_argument("-plot_pvd", help="Toggle plotting to .pvd")
parser.add_argument("-debug", help="Print all debugging statements")
parser.add_argument("-debug_mode", help="Choose debugging mode from 'basic' and 'full'")
args = parser.parse_args()


# --- Set parameters

approach = 'fixed_mesh'
load_mesh = None if args.load_mesh is None else 'plex'
plot_pvd = bool(args.plot_pvd or False)
kwargs = {
    'approach': approach,
    'num_meshes': int(args.num_meshes or 1),
    'plot_pvd': plot_pvd,
    'debug': bool(args.debug or False),
    'debug_mode': args.debug_mode or 'basic',
}
op = TurbineArrayOptions(3.0, **kwargs)
mode = 'memory'  # TODO: disk

# Create directories and check if spun-up solution exists
data_dir = create_directory(os.path.join(os.path.dirname(__file__), "data"))
ramp_dir = create_directory(os.path.join(data_dir, "ramp"))
data_dir = create_directory(os.path.join(data_dir, approach))
spun = np.all([os.path.isfile(os.path.join(ramp_dir, f + ".h5")) for f in ('velocity', 'elevation')])
power_watts = [np.array([]) for i in range(15)]
if spun:
    for i, turbine in enumerate(op.farm_ids):
        fname = os.path.join(ramp_dir, "power_output_{:d}.npy".format(turbine))
        power_watts[i] = np.append(power_watts[i], np.load(fname)*op.sea_water_density)
else:
    op.end_time += op.T_ramp


# --- Forward solve

# Run forward model and save QoI timeseries
swp = AdaptiveTurbineProblem(op, callback_dir=data_dir, ramp_dir=ramp_dir, load_mesh=load_mesh, checkpointing=False)

for i in range(swp.num_meshes):

    # Set initial condition / transfer data from previous mesh
    swp.transfer_forward_solution(i)

    # Checkpoint solution at start of mesh iteration
    swp.save_to_checkpoint(swp.fwd_solutions[i], mode=mode)
    if i == swp.num_meshes-1:
        continue

    # Create forward solver
    swp.setup_solver_forward_step(i)

    # Solve forward problem
    cpu_timestamp = perf_counter()
    swp.solve_forward_step(i)
    cpu_time = perf_counter() - cpu_timestamp
    msg = "CPU time for forward solve {:d}: {:.1f} seconds / {:.1f} minutes / {:.3f} hours"
    print_output(msg.format(i, cpu_time, cpu_time/60, cpu_time/3600))
    average_power = swp.quantity_of_interest()
    print_output("Average power output of array: {:.1f}W".format(average_power))

    # Free forward solver
    swp.free_solver_forward_step(i)

if not spun:
    op.end_time -= op.T_ramp


# --- Adjoint solve

swp.checkpointing = True
for i in reversed(range(swp.num_meshes)):

    # --- First, solve forward from checkpoint

    # Load from checkpoint
    swp.fwd_solutions[i].assign(swp.collect_from_checkpoint(mode=mode))

    # Create forward solver
    swp.setup_solver_forward_step(i)

    # Solve forward problem
    cpu_timestamp = perf_counter()
    swp.solve_forward_step(i)
    cpu_time = perf_counter() - cpu_timestamp
    msg = "CPU time for forward solve {:d}: {:.1f} seconds / {:.1f} minutes / {:.3f} hours"
    print_output(msg.format(i, cpu_time, cpu_time/60, cpu_time/3600))
    average_power = swp.quantity_of_interest()
    print_output("Average power output of array: {:.1f}W".format(average_power))

    # Free forward solver
    swp.free_solver_forward_step(i)

    # --- Next, solve adjoint using checkpointed state

    # Set terminal condition / transfer data from previous mesh
    swp.transfer_adjoint_solution()

    # Create adjoint solver
    swp.setup_solver_adjoint_step(i)

    # Solve adjoint problem
    cpu_timestamp = perf_counter()
    swp.solve_adjoint_step(i)
    cpu_time = perf_counter() - cpu_timestamp
    msg = "CPU time for adjoint solve {:d}: {:.1f} seconds / {:.1f} minutes / {:.3f} hours"
    print_output(msg.format(i, cpu_time, cpu_time/60, cpu_time/3600))

    # Free adjoint solver
    swp.free_solver_adjoint_step(i)
