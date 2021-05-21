from thetis import *

import argparse
import os

from adapt_utils.adapt.metric import metric_intersection
from adapt_utils.adapt.r import MeshMover
from adapt_utils.adapt.recovery import recover_hessian
from adapt_utils.norms import *
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
parser.add_argument("-monitor", help="Choose monitor from 'uv', 'elev', 'both'.")
parser.add_argument("-calculate_metrics", help="Compute metrics using the fine mesh.")
parser.add_argument("-debug", help="Toggle debugging mode.")
args = parser.parse_args()


# --- Set parameters

n_coarse = int(args.n_coarse or 1)  # NOTE: [Huang et al 2008] considers n = 4, 8, 20
n_fine = int(args.n_fine or 50)

# Monitor function for initial mesh
refine_equator = bool(args.refine_equator or False)
refine_soliton = bool(args.refine_soliton or False)
monitor_type = args.monitor or 'elev'
initial_monitor = None
initial_monitor_type = None
if refine_equator:
    if refine_soliton:
        initial_monitor = lambda mesh: equator_monitor(mesh) + soliton_monitor(mesh)
        initial_monitor_type = 'equator_and_soliton'
    else:
        initial_monitor = equator_monitor
        initial_monitor_type = 'equator'
elif refine_soliton:
    initial_monitor = soliton_monitor
    initial_monitor_type = 'soliton'

kwargs = {

    # Asymptotic expansion
    # 'order': 0,
    'order': 1,

    # Spatial discretisation
    'n': n_coarse,

    # Timesteppring
    'end_time': float(args.end_time or 120.0),
    'dt': 0.04/n_coarse,
    'dt_per_export': 50*n_coarse,

    # Outputs
    'plot_pvd': True,
    # 'plot_pvd': n_coarse < 5,

    # Mesh movement
    'r_adapt_rtol': 1.0e-3,
    'nonlinear_method': 'relaxation',
    # 'nonlinear_method': 'quasi_newton',
    'dt_per_mesh_movement': 1,  # TODO: Could probably get away with many more

    # Misc
    'debug': bool(args.debug or False),
}
if os.getenv('REGRESSION_TEST') is not None:
    kwargs['end_time'] = kwargs['dt']*kwargs['dt_per_export']
fpath = 'resolution_{:d}'.format(n_coarse)
if initial_monitor_type is not None:
    fpath = os.path.join(fpath, initial_monitor_type)
fpath = os.path.join(fpath, monitor_type)
op = BoydOptions(approach='monge_ampere', n=n_coarse, fpath=fpath, order=kwargs['order'])
op.update(kwargs)


# --- Initialise mesh

swp = AdaptiveProblem(op)

# Refine around equator and/or soliton
if initial_monitor is not None:
    mesh_mover = MeshMover(swp.meshes[0], initial_monitor, method='monge_ampere', op=op)
    mesh_mover.adapt()
    mesh = Mesh(mesh_mover.x)
    op.__init__(mesh=mesh, **kwargs)
    swp.__init__(op, meshes=[mesh])


# --- Monitor function definitions

def elevation_norm_monitor(mesh, alpha=40.0, norm_type='H1', **kwargs):
    """
    Monitor function derived from the elevation `norm_type` norm.

    :kwarg alpha: controls the amplitude of the monitor function.
    """
    P1DG = FunctionSpace(mesh, "DG", 1)
    eta = project(swp.fwd_solutions[0].split()[1], P1DG)
    if norm_type == 'hessian_frobenius':
        H = recover_hessian(eta, op=op)
        return 1.0 + alpha*local_frobenius_norm(H)
    else:
        return 1.0 + alpha*local_norm(eta, norm_type=norm_type)


def velocity_norm_monitor(mesh, alpha=40.0, norm_type='HDiv', **kwargs):
    """
    Monitor function derived from the velocity `norm_type` norm.

    :kwarg alpha: controls the amplitude of the monitor function.
    """
    P1DG_vec = VectorFunctionSpace(mesh, "DG", 1)
    u = project(swp.fwd_solutions[0].split()[0], P1DG_vec)
    if norm_type == 'hessian_frobenius':
        H1 = recover_hessian(u[0], op=op)
        H2 = recover_hessian(u[1], op=op)
        return 1.0 + alpha*local_frobenius_norm(metric_intersection(H1, H2))
    else:
        return 1.0 + alpha*local_norm(u, norm_type=norm_type)


def mixed_monitor(mesh, **kwargs):
    return 0.5*(velocity_norm_monitor(mesh, norm_type='HDiv') + elevation_norm_monitor(mesh, norm_type='H1'))


# Set monitor function to use during simulation
if monitor_type == 'elev':
    monitor = elevation_norm_monitor
elif monitor_type == 'uv':
    monitor = velocity_norm_monitor
else:
    monitor = mixed_monitor
swp.set_monitor_functions(monitor)


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
