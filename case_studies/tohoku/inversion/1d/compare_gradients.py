from thetis import *
from firedrake_adjoint import *

import numpy as np
import os

from adapt_utils.argparse import ArgumentParser
from adapt_utils.case_studies.tohoku.options.radial_options import TohokuRadialBasisOptions
from adapt_utils.unsteady.solver import AdaptiveProblem
from adapt_utils.unsteady.solver_adjoint import AdaptiveDiscreteAdjointProblem


# --- Parse arguments

parser = ArgumentParser(
    prog="compare_gradients",
    description="""
        Compare gradients arising from discrete and continuous adjoint methods against those
        computed using a finite difference method.
        """,
    shallow_water=True,
)
parser.add_argument("-control_parameter", help="Where to evaluate gradient (default 7.5)")
parser.add_argument("-adjoint_free", help="Toggle adjoint-free computation")
parser.add_argument("-discrete_adjoint", help="Toggle discrete adjoint computation")
parser.add_argument("-continuous_adjoint", help="Toggle continuous adjoint computation")
parser.add_argument("-finite_differences", help="Toggle finite difference computation")
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
af = False if args.adjoint_free == "0" else True
da = False if args.discrete_adjoint == "0" else True
ca = False if args.continuous_adjoint == "0" else True
fd = False if args.finite_differences == "0" else True
op = TohokuRadialBasisOptions(**kwargs)
gauges = list(op.gauges)

# Toggle smoothed or discrete timeseries
timeseries = "timeseries"
use_smoothed_timeseries = True
if use_smoothed_timeseries:
    timeseries = "_".join([timeseries, "smooth"])


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
        swp = AdaptiveDiscreteAdjointProblem(op, nonlinear=nonlinear, print_progress=False)
        op.assign_control_parameters([5.0], mesh=swp.meshes[0])
        swp.solve_forward()
    for gauge, fname in zip(gauges, fnames):
        op.gauges[gauge]['data'] = op.gauges[gauge][timeseries]
        np.save(fname, op.gauges[gauge]['data'])


# --- Adjoint-free

if af:
    print_output("*** ADJOINT-FREE ***...")

    # Solve the forward problem with 'suboptimal' control parameter m = 7.5, no checkpointing
    op.di = create_directory(os.path.join(di, 'adjoint_free'))
    swp = AdaptiveProblem(op, nonlinear=nonlinear, checkpointing=False, print_progress=False)
    op.assign_control_parameters(kwargs['control_parameters'])
    print_output("Solve forward...")
    swp.solve_forward()
    g_adjoint_free = op.adjoint_free_gradient/kwargs['control_parameters'][0]
    print_output("Gradient computed without adjoint: {:.4e}".format(g_adjoint_free))

# --- Discrete adjoint

if da:
    print_output("*** DISCRETE ADJOINT ***...")

    # Solve the forward problem with 'suboptimal' control parameter m = 7.5
    op.di = create_directory(os.path.join(di, 'discrete'))
    swp = AdaptiveDiscreteAdjointProblem(op, nonlinear=nonlinear, checkpointing=False, print_progress=False)
    swp.clear_tape()
    print_output("Setting initial guess...")
    op.assign_control_parameters(kwargs['control_parameters'], mesh=swp.meshes[0])
    control = Control(op.control_parameter)

    # Solve the forward problem
    fname = '{:s}_{:s}_{:d}.npy'
    fnames = [os.path.join(di, fname.format(gauge, timeseries, level)) for gauge in gauges]
    print_output("Run forward to get initial timeseries...")
    swp.solve_forward()
    for gauge, fname in zip(gauges, fnames):
        np.save(fname, op.gauges[gauge][timeseries])

    # Compute gradient
    # g_discrete = swp.compute_gradient(control).dat.data[0]
    g_discrete = compute_gradient(swp.quantity_of_interest(), control).dat.data[0]
    print_output("Gradient computed by dolfin-adjoint (discrete): {:.4e}".format(g_discrete))

    # Check consistency of by-hand gradient formula
    swp.save_adjoint_trajectory()
    g_by_hand_discrete = assemble(inner(op.basis_function, swp.adj_solution)*dx)
    print_output("Gradient computed by hand (discrete): {:.4e}".format(g_by_hand_discrete))
    relative_error = abs((g_discrete - g_by_hand_discrete)/g_discrete)
    assert np.isclose(relative_error, 0.0)
    swp.clear_tape()
    stop_annotating()


# --- Continuous adjoint

if ca:
    print_output("*** CONTINUOUS ADJOINT ***...")

    # Solve the forward problem with 'suboptimal' control parameter m = 7.5, checkpointing state
    op.di = create_directory(os.path.join(di, 'continuous'))
    swp = AdaptiveProblem(op, nonlinear=nonlinear, checkpointing=True, print_progress=False)
    swp.solve_forward()

    # Solve adjoint equation in continuous form
    swp.solve_adjoint()
    g_continuous = assemble(inner(op.basis_function, swp.adj_solution)*dx)
    print_output("Gradient computed by hand (continuous): {:.4e}".format(g_continuous))


# --- Finite differences

# Establish gradient using finite differences
if fd:
    print_output("*** FINITE DIFFERENCES ***...")

    swp = AdaptiveProblem(op, nonlinear=nonlinear, checkpointing=False, print_progress=False)
    op.assign_control_parameters(kwargs['control_parameters'])
    swp.solve_forward(plot_pvd=False)
    J = swp.quantity_of_interest()
    epsilon = 1.0
    converged = False
    rtol = 1.0e-05
    g_fd_ = None
    op.save_timeseries = False
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
if af:
    logstr += "adjoint-free gradient: {:.8e}\n".format(g_adjoint_free)
if da:
    logstr += "discrete gradient: {:.8e}\n".format(g_discrete)
if ca:
    logstr += "continuous gradient: {:.8e}\n".format(g_continuous)
if fd:
    logstr += "finite difference gradient (rtol={:.1e}): {:.8e}\n".format(rtol, g_fd)
print_output(logstr)
fname = os.path.join(di, "gradient_at_{:.1f}_level{:d}.log")
with open(fname.format(int(kwargs['control_parameters'][0]), level), 'w') as logfile:
    logfile.write(logstr)
