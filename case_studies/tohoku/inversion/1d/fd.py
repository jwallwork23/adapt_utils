from thetis import *

import numpy as np

from adapt_utils.argparse import ArgumentParser
from adapt_utils.case_studies.tohoku.options.radial_options import TohokuRadialBasisOptions
from adapt_utils.unsteady.solver import AdaptiveProblem


# --- Parse arguments

parser = ArgumentParser(
    shallow_water=True,
)
parser.add_argument("-control_parameter", help="Where to evaluate gradient")
args = parser.parse_args()


# --- Setup

# Setup output directories
dirname = os.path.dirname(__file__)
di = create_directory(os.path.join(dirname, 'outputs', 'synthetic', 'compare_gradients'))

# Set parameters
level = int(args.level or 0)
nonlinear = bool(args.nonlinear or False)
family = args.family or 'dg-cg'
stabilisation = args.stabilisation or 'lax_friedrichs'
if stabilisation == 'none' or family == 'cg-cg' or not nonlinear:
    stabilisation = None
kwargs = {
    'level': level,
    'save_timeseries': True,

    # Spatial discretisation
    'family': family,
    'stabilisation': stabilisation,
    'use_automatic_sipg_parameter': False,  # the problem is inviscid

    # Adjoint
    'control_parameters': [float(args.control_parameter or 7.5)],
    'optimal_value': 5.0,
    'synthetic': True,
    'qoi_scaling': 1.0,
    'nx': 1,
    'ny': 1,

    # Misc
    'plot_pvd': False,
    'debug': bool(args.debug or False),
    'debug_mode': args.debug_mode or 'basic',
}
op = TohokuRadialBasisOptions(**kwargs)
gauges = list(op.gauges)


# --- Synthetic run

# Solve the forward problem / load data
fnames = [os.path.join(di, '{:s}_data_{:d}.npy'.format(gauge, level)) for gauge in gauges]
try:
    assert np.all([os.path.isfile(fname) for fname in fnames])
    print_output("Loading timeseries data...")
    for fname, gauge in zip(fnames, gauges):
        op.gauges[gauge]['data'] = np.load(fname)
except AssertionError:
    print_output("Run forward to get 'data'...")
    with stop_annotating():
        swp = AdaptiveProblem(op, nonlinear=nonlinear, print_progress=False)
        op.assign_control_parameters([5.0], mesh=swp.meshes[0])
        swp.solve_forward()
    for gauge, fname in zip(gauges, fnames):
        op.gauges[gauge]['data'] = op.gauges[gauge][timeseries]
        np.save(fname, op.gauges[gauge]['data'])

# Solve the forward problem with 'suboptimal' control parameter m = 10, checkpointing state
swp = AdaptiveProblem(op, nonlinear=nonlinear, print_progress=False)
op.assign_control_parameters(kwargs['control_parameters'])
swp.solve_forward()
J = swp.quantity_of_interest()
assert not np.allclose(J, 0.0)


# ---  Establish gradient using finite differences

epsilon = 1.0
converged = False
rtol = 1.0e-05
g_fd_ = None
while not converged:
    op.assign_control_parameters([kwargs['control_parameters'][0] + epsilon])
    swp.solve_forward(plot_pvd=False)
    J_step = swp.quantity_of_interest()
    g_fd = (J_step - J)/epsilon
    print_output("J(epsilon=0) = {:.8e}  J(epsilon={:.1e}) = {:.8e}".format(J, epsilon, J_step))
    if g_fd_ is not None:
        print_output("gradient = {:.8e}  difference = {:.8e}".format(g_fd, abs(g_fd - g_fd_)))
        if abs(g_fd - g_fd_) < rtol*J:
            converged = True
        elif epsilon < 1.0e-10:
            raise ConvergenceError
    epsilon *= 0.5
    g_fd_ = g_fd


# --- Logging

logstr = "elements: {:d}\n".format(swp.meshes[0].num_cells())
logstr += "finite difference gradient (rtol={:.1e}): {:.4e}\n".format(rtol, g_fd)
print_output(logstr)
