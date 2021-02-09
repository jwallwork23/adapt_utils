import argparse
import h5py
import os
from time import perf_counter

from adapt_utils.io import create_directory
from adapt_utils.norms import *
from adapt_utils.unsteady.solver import AdaptiveProblem
from adapt_utils.unsteady.test_cases.bubble_shear.options import BubbleOptions


# --- Parse arguments

parser = argparse.ArgumentParser()
parser.add_argument("-n", help="Resolution of initial mesh.")
parser.add_argument("-dt", help="Timestep.")
parser.add_argument("-end_time", help="Simulation end time.")

parser.add_argument("-conservative", help="Toggle conservative tracer equation")
parser.add_argument("-limiters", help="Toggle limiters for tracer equation")
parser.add_argument("-stabilisation", help="Stabilisation method")
parser.add_argument("-anisotropic_stabilisation", help="Toggle anisotropic stabilisation")
parser.add_argument("-family", help="Choose finite element from 'cg' and 'dg'")

parser.add_argument("-approach", help="Mesh adaptation approach")
parser.add_argument("-min_level", help="Lowest target level considered")
parser.add_argument("-max_level", help="Largest target level considered")
parser.add_argument("-num_meshes", help="Number of meshes in the sequence")
parser.add_argument("-max_adapt", help="Maximum number of adaptation steps")
parser.add_argument("-hessian_time_combination")

parser.add_argument("-debug", help="Toggle debugging mode.")
args = parser.parse_args()


# --- Set parameters

kwargs = {

    # Solver
    'tracer_family': args.family or 'cg',
    'stabilisation_tracer': args.stabilisation or 'supg',
    'anisotropic_stabilisation': False if args.anisotropic_stabilisation == "0" else True,
    'use_automatic_sipg_parameter': False,  # We have an inviscid problem
    'use_limiter_for_tracers': bool(args.limiters or False),
    'use_tracer_conservative_form': bool(args.conservative or False),  # FIXME?

    # Mesh adaptation
    'approach': args.approach or 'hessian',
    'num_meshes': int(args.num_meshes or 50),
    'max_adapt': int(args.max_adapt or 3),
    'hessian_time_combination': args.hessian_time_combination or 'integrate',
    'norm_order': 1,
    'normalisation': 'complexity',

    # I/O and debugging
    'plot_pvd': False,
    'debug': bool(args.debug or False),
}
op = BubbleOptions(approach='hessian', n=int(args.n or 1))
op.update(kwargs)
if args.dt is not None:
    op.dt = float(args.dt)
if args.end_time is not None:
    op.end_time = float(args.end_time)
op.di = create_directory(os.path.join(op.di, op.hessian_time_combination))

# --- Solve the tracer transport problem

assert op.approach != 'fixed_mesh'
for n in range(int(args.min_level or 0), int(args.max_level or 5)):
    op.target = 1000*2**n
    op.dt = 0.01*0.5**n
    op.dt_per_export = 2**n

    # Run simulation
    tp = AdaptiveProblem(op)
    cpu_timestamp = perf_counter()
    tp.run()
    times = [perf_counter() - cpu_timestamp]
    dofs = [Q.dof_count for Q in tp.Q]
    num_cells = [mesh.num_cells() for mesh in tp.meshes]

    # Assess error
    final_sol = tp.fwd_solutions_tracer[-1].copy(deepcopy=True)
    final_l1_norm = norm(final_sol, norm_type='L1')
    final_l2_norm = norm(final_sol, norm_type='L2')
    tp.set_initial_condition(i=-1)
    init_sol = tp.fwd_solutions_tracer[-1].copy(deepcopy=True)
    init_l1_norm = norm(init_sol, norm_type='L1')
    init_l2_norm = norm(init_sol, norm_type='L2')
    abs_l2_error = errornorm(init_sol, final_sol, norm_type='L2')
    cons_error = [100*abs(init_l1_norm-final_l1_norm)/init_l1_norm]
    l2_error = [100*abs_l2_error/init_l2_norm]

    # Save to HDF5
    with h5py.File(os.path.join(op.di, 'convergence_{:d}.h5'.format(n)), 'w') as outfile:
        outfile.create_dataset('iterations', data=[len(tp.outer_iteration)])
        outfile.create_dataset('elements', data=num_cells)
        outfile.create_dataset('dofs', data=dofs)
        outfile.create_dataset('time', data=times)
        outfile.create_dataset('l2_error', data=l2_error)
        outfile.create_dataset('cons_error', data=cons_error)
