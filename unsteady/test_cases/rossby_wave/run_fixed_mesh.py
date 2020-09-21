from thetis import *

import argparse
import os

from adapt_utils.adapt.r import MeshMover
from adapt_utils.unsteady.solver import AdaptiveProblem
from adapt_utils.unsteady.test_cases.rossby_wave.monitors import *
from adapt_utils.unsteady.test_cases.rossby_wave.options import BoydOptions


# --- Parse arguments

parser = argparse.ArgumentParser()
parser.add_argument("-n_coarse", help="Resolution of coarse mesh.")
parser.add_argument("-n_fine", help="Resolution of fine mesh.")
parser.add_argument("-end_time", help="Simulation end time.")
parser.add_argument("-refine_equator", help="""
    Apply Monge-Ampere based r-adaptation to refine equatorial region.""")
parser.add_argument("-refine_soliton", help="""
    Apply Monge-Ampere based r-adaptation to refine around initial soliton.""")
parser.add_argument("-calculate_metrics", help="Compute metrics using the fine mesh.")
parser.add_argument("-ale", help="Use ALE mesh movement to track soliton.")
parser.add_argument("-debug", help="Toggle debugging mode.")
args = parser.parse_args()


# --- Set parameters

n_coarse = int(args.n_coarse or 1)  # NOTE: [Huang et al 2008] considers n = 4, 8, 20
n_fine = int(args.n_fine or 50)
ale = bool(args.ale or False)

# Monitor function for initial mesh
refine_equator = bool(args.refine_equator or False)
refine_soliton = bool(args.refine_soliton or False)
monitor = None
monitor_type = None
if refine_equator:
    if refine_soliton:
        monitor = lambda mesh: equator_monitor(mesh) + soliton_monitor(mesh)
        monitor_type = 'equator_and_soliton'
    else:
        monitor = equator_monitor
        monitor_type = 'equator'
elif refine_soliton:
    monitor = soliton_monitor
    monitor_type = 'soliton'

# Gather parameters
kwargs = {
    'num_meshes': 1,

    # Asymptotic expansion
    # 'order': 0,
    'order': 1,

    # Timestepping
    'end_time': float(args.end_time or 120.0),
    'dt': 0.04/n_coarse,
    'dt_per_export': 50*n_coarse,

    # Outputs
    'plot_pvd': True,
    # 'plot_pvd': n_coarse < 5,

    # Mesh movement
    'r_adapt_rtol': 1.0e-3,

    # Misc
    'debug': bool(args.debug or False),
}
if os.getenv('REGRESSION_TEST') is not None:
    kwargs['end_time'] = 30.0
fpath = 'resolution_{:d}'.format(n_coarse)
if monitor is not None:
    fpath = os.path.join(fpath, monitor_type)
op = BoydOptions(approach='ale' if ale else 'fixed_mesh', fpath=fpath, n=n_coarse, order=kwargs['order'])
op.update(kwargs)


# --- Initialise mesh

swp = AdaptiveProblem(op)

# Refine around equator and/or soliton
if monitor is not None:
    mesh_mover = MeshMover(swp.meshes[0], monitor, method='monge_ampere', op=op)
    mesh_mover.adapt()
    mesh = Mesh(mesh_mover.x)
    op.__init__(mesh=mesh, **kwargs)
    swp.__init__(op, meshes=[mesh])

# Apply constant mesh velocity  # FIXME
if ale:
    swp.mesh_velocities[0] = Constant(as_vector([-op.lx/op.end_time, 0.0]))


# --- Solve forward problem and print diagnostics

swp.solve_forward()
if bool(args.calculate_metrics or False):
    print_output("\nCalculating error metrics...")
    metrics = op.get_peaks(swp.fwd_solutions[-1].split()[1], reference_mesh_resolution=n_fine)
    print_output("h+       : {:.8e}".format(metrics[0]))
    print_output("h-       : {:.8e}".format(metrics[1]))
    print_output("C+       : {:.8e}".format(metrics[2]))
    print_output("C-       : {:.8e}".format(metrics[3]))
    print_output("RMS error: {:.8e}".format(metrics[4]))
