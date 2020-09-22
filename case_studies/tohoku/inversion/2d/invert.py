from thetis import *

import numpy as np
import os
import scipy
import sys

from adapt_utils.argparse import ArgumentParser
from adapt_utils.case_studies.tohoku.options.box_options import TohokuBoxBasisOptions


# --- Parse arguments

parser = ArgumentParser(
    prog="invert",
    description="""
            Invert for an initial condition defined over a piecewise constant basis with two
            basis functions.

            A 'synthetic' tsunami is generated from an initial condition given by the 'optimal'
            scaling parameter is m = (5, 5). We apply PDE constrained optimisation with an initial
            guess m = (2.5, 7.5) and the objective of minimising the square gauge timeseries error,
            J.

            The gradient of J w.r.t. m is computed using either a discrete or continuous adjoint
            approach.
        """,
    adjoint=True,
    optimisation=True,
    shallow_water=True,
)
parser.add_argument("-recompute_parameter_space", help="Recompute parameter space")
parser.add_argument("-initial_guess", help="Initial guess for control parameter")
parser.add_argument("-optimal_control", help="Artificially choose an optimum to invert for")
parser.add_argument("-regularisation", help="Parameter for Tikhonov regularisation term")


# --- Imports relevant to adjoint mode

args = parser.args
if args.adjoint == 'continuous':
    from adapt_utils.pyadjoint_dummy import *
    from adapt_utils.unsteady.solver import AdaptiveProblem
    problem_constructor = AdaptiveProblem
elif args.adjoint == 'discrete':
    from firedrake_adjoint import *
    from adapt_utils.unsteady.solver_adjoint import AdaptiveDiscreteAdjointProblem
    problem_constructor = AdaptiveDiscreteAdjointProblem
else:
    raise ValueError("Adjoint mode '{:}' not recognised.".format(args.adjoint))


# --- Set parameters

# Parsed arguments
level = int(args.level or 0)
recompute = bool(args.recompute_parameter_space or False)
optimise = bool(args.rerun_optimisation or False)
gtol = float(args.gtol or 1.0e-04)
plot_pvd = bool(args.plot_pvd or False)
timeseries = 'timeseries'
if bool(args.continuous_timeseries or False):
    timeseries = '_'.join([timeseries, 'smooth'])

# Setup output directories
dirname = os.path.dirname(__file__)
di = create_directory(os.path.join(dirname, 'outputs', 'synthetic'))
if args.extension is not None:
    di = '_'.join([di, args.extension])

# Collect initialisation parameters
nonlinear = bool(args.nonlinear or False)
family = args.family or 'dg-cg'
stabilisation = args.stabilisation or 'lax_friedrichs'
if stabilisation == 'none' or family == 'cg-cg' or not nonlinear:
    stabilisation = None
taylor = bool(args.taylor_test or False)
chk = args.checkpointing_mode or 'disk'
init = [2.5, 7.5]
if args.initial_guess is not None:
    init = [float(ig) for ig in args.initial_guess.split(',')]
opt = [5.0, 5.0]
if args.optimal_control is not None:
    opt = [float(oc) for oc in args.optimal_control.split(',')]
kwargs = {
    'level': level,
    'save_timeseries': True,

    # Spatial discretisation
    'family': family,
    'stabilisation': stabilisation,
    'use_automatic_sipg_parameter': False,  # the problem is inviscid

    # Optimisation
    'control_parameters': init,
    'synthetic': True,
    'qoi_scaling': 1.0,
    'nx': 2,
    'ny': 1,
    'regularisation': float(args.regularisation or 0.0),

    # I/O and debugging
    'plot_pvd': False,
    'debug': bool(args.debug or False),
    'debug_mode': args.debug_mode or 'basic',
    'di': di,
}
if args.end_time is not None:
    kwargs['end_time'] = float(args.end_time)
use_regularisation = not np.isclose(kwargs['regularisation'], 0.0)
op = TohokuBoxBasisOptions(**kwargs)
op.dirty_cache = bool(args.dirty_cache or False)
gauges = list(op.gauges.keys())


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
        swp = problem_constructor(op, nonlinear=nonlinear, print_progress=False)
        op.assign_control_parameters(opt, mesh=swp.meshes[0])
        swp.solve_forward()
    for gauge, fname in zip(gauges, fnames):
        op.gauges[gauge]['data'] = op.gauges[gauge][timeseries]
        np.save(fname, op.gauges[gauge]['data'])


# --- Explore parameter spaces

n = 8
op.save_timeseries = False
control_values = np.linspace(0.5, 7.5, n)

# Unregularised parameter space
fname = os.path.join(di, 'parameter_space_{:d}.npy'.format(level))
if recompute or not os.path.isfile(fname):
    msg = "({:2d}, {:2d}): control values ({:.4e}, {:.4e})  functional value {:.4e}"
    func_values = np.zeros((n, n))
    with stop_annotating():
        swp = problem_constructor(op, nonlinear=nonlinear, print_progress=False)
        for i, m1 in enumerate(control_values):
            for j, m2 in enumerate(control_values):
                op.assign_control_parameters([m1, m2], mesh=swp.meshes[0])
                swp.solve_forward()
                func_values[i, j] = swp.quantity_of_interest()
                print_output(msg.format(i, j, m1, m2, func_values[i, j]))
    np.save(fname, func_values)

# Regularised parameter space
fname = os.path.join(di, 'parameter_space_reg_{:d}.npy'.format(level))
if use_regularisation and (recompute or os.path.isfile(fname)):
    msg = "({:2d}, {:2d}): control value ({:.4e}, {:.4e})  regularised functional value {:.4e}"
    func_values_reg = np.zeros((n, n))
    with stop_annotating():
        swp = problem_constructor(op, nonlinear=nonlinear, print_progress=False)
        for i, m1 in enumerate(control_values):
            for j, m2 in enumerate(control_values):
                op.assign_control_parameters([m1, m2], mesh=swp.meshes[0])
                swp.solve_forward()
                func_values_reg[i, j] = swp.quantity_of_interest()
                print_output(msg.format(i, j, m1, m2, func_values_reg[i, j]))
    np.save(fname, func_values_reg)


# --- Tracing

# Set initial guess
op.save_timeseries = True
swp = problem_constructor(op, nonlinear=nonlinear, print_progress=False)
swp.clear_tape()
print_output("Setting initial guess...")
op.assign_control_parameters(init, mesh=swp.meshes[0])
control = [Control(m) for m in op.control_parameters]

# Solve the forward problem / load data
fname = '{:s}_{:s}_{:d}.npy'
fnames = [os.path.join(di, fname.format(gauge, timeseries, level)) for gauge in gauges]
try:
    assert args.adjoint == 'continuous'
    assert np.all([os.path.isfile(fname) for fname in fnames])
    print_output("Loading initial timeseries...")
    for gauge, fname in zip(gauges, fnames):
        op.gauges[gauge][timeseries] = np.load(fname)
except AssertionError:
    print_output("Run forward to get initial timeseries...")
    swp.solve_forward()
    J = swp.quantity_of_interest()
    for gauge, fname in zip(gauges, fnames):
        np.save(fname, op.gauges[gauge][timeseries])
di = create_directory(os.path.join(di, args.adjoint))
op.get_basis_functions(swp.V[0])

# Define reduced functional and gradient functions
if args.adjoint == 'discrete':
    Jhat = ReducedFunctional(J, control)
    gradient = None
else:

    def Jhat(m):
        """
        Reduced functional for continuous adjoint inversion.
        """
        op.assign_control_parameters(m)
        swp.solve_forward(checkpointing_mode=chk)
        return swp.quantity_of_interest()

    def gradient(m):
        """
        Gradient of reduced functional for continuous adjoint inversion.
        """
        J = Jhat(m) if len(swp.checkpoint) == 0 else swp.quantity_of_interest()
        swp.solve_adjoint(checkpointing_mode=chk)
        g = np.array([
            assemble(inner(op.basis_functions[0], swp.adj_solution)*dx),
            assemble(inner(op.basis_functions[1], swp.adj_solution)*dx),
        ])  # TODO: No minus sign?
        if use_regularisation:
            g += op.regularisation_term_gradients[0]
        msg = "control = ({:15.8e}, {:15.8e})  functional = {:15.8e}  gradient = ({:15.8e}, {:15.8e})"
        try:
            print_output(msg.format(*m, J, *g))
        except Exception:
            print_output(msg.format(*m.dat.data, J, *g))
        return g

# --- Taylor test

if taylor:

    def get_control(mode):
        """
        Get control parameter values assuming one of three modes:

          'init' - control parameter is set to the initial guess;
          'random' - control parameter is set randomly with a Normal distribution;
          'optimised' - optimal control parameter from a previous run is used.
        """
        c = np.array([Function(m) for m in op.control_parameter])
        if mode == 'init':
            pass
        elif mode == 'random':
            np.random.seed(0)                      # Ensure consistency of tests
            c.dat.data[0] = 10*np.random.random()  # Random number between 0 and 10
        elif mode == 'optimised':
            fname = os.path.join(di, 'optimisation_progress_ctrl_{:d}.npy'.format(level))
            try:
                c_dat = np.load(fname)[-1]
            except FileNotFoundError:
                msg = "Skipping Taylor test at optimised control because no data found."
                print_output(msg)
                sys.exit(0)
            c.dat.data[0] = c_dat[0]
        else:
            raise ValueError("Taylor test mode '{:s}' not recognised.".format(mode))
        return c

    # Increasing search direction of length 0.1
    dc = np.array([Function(m) for m in op.control_parameter])
    dc.assign(0.1)

    # Run tests
    for mode in ("init", "random", "optimised"):
        print_output("Taylor test '{:s}' begin...".format(mode))
        c = get_control(mode)
        swp.checkpointing = True
        dJdm = None if args.adjoint == 'discrete' else gradient(c)
        swp.checkpointing = False
        minconv = taylor_test(Jhat, c, dc, dJdm=dJdm)
        if minconv > 1.90:
            print_output("Taylor test '{:s}' passed!".format(mode))
        else:
            msg = "Taylor test '{:s}' failed! Convergence ratio {:.2f} < 2."
            raise ConvergenceError(msg.format(mode, minconv))
    sys.exit(0)


# --- Optimisation

fname = os.path.join(di, 'optimisation_progress_{:s}')
if use_regularisation:
    fname = '_'.join([fname, 'reg'])
fname += '_{:d}.npy'.format(level)
if optimise:
    control_values_opt = []
    func_values_opt = []
    gradient_values_opt = []

    opt_kwargs = {'maxiter': 1000, 'gtol': gtol}
    print_output("Optimisation begin...")
    if args.adjoint == 'discrete':

        def derivative_cb_post(j, dj, m):
            """
            Callback for saving progress data to file during discrete adjoint inversion.
            """
            control = [mi.dat.data[0] for mi in m]
            djdm = [dji.dat.data[0] for dji in dj]
            msg = "control ({:15.8e}, {:15.8e})  functional {:15.8e}  gradient ({:15.8e}, {:15.8e})"
            print_output(msg.format(*control, j, *djdm))

            # Save progress to NumPy arrays on-the-fly
            control_values_opt.append(control)
            func_values_opt.append(j)
            gradient_values_opt.append(djdm)
            np.save(fname.format('ctrl'), np.array(control_values_opt))
            np.save(fname.format('func'), np.array(func_values_opt))
            np.save(fname.format('grad'), np.array(gradient_values_opt))

        # Run BFGS optimisation
        Jhat_save_data = ReducedFunctional(J, control, derivative_cb_post=derivative_cb_post)
        optimised_value = minimize(Jhat_save_data, method='BFGS', options=opt_kwargs).dat.data
    else:
        swp.checkpointing = True

        def Jhat_save_data(m):
            """
            Reduced functional for the continuous adjoint approach which saves progress data to
            file during the inversion.
            """
            J = Jhat(m)
            control_values_opt.append(m)
            np.save(fname.format('ctrl'), np.array(control_values_opt))
            func_values_opt.append(J)
            np.save(fname.format('func'), np.array(func_values_opt))
            return J

        def gradient_save_data(m):
            """
            Gradient of the reduced functional for the continuous adjoint approach which saves
            progress data to file during the inversion.
            """
            g = gradient(m)
            gradient_values_opt.append(g)
            np.save(fname.format('grad'), np.array(gradient_values_opt))
            return g

        opt_kwargs['fprime'] = gradient_save_data
        opt_kwargs['callback'] = lambda m: print_output("LINE SEARCH COMPLETE")
        m_init = [m.dat.data[0] for m in op.control_parameters]
        optimised_value = scipy.optimize.fmin_bfgs(Jhat_save_data, m_init, **opt_kwargs)
else:
    optimised_value = np.array([np.load(fname.format('ctrl'))[-1]])


# --- Run with optimised controls

# Run forward again so that we can compare timeseries
op.plot_pvd = plot_pvd
op.di = di
swp = problem_constructor(op, nonlinear=nonlinear, print_progress=False)
swp.clear_tape()
print_output("Assigning optimised control parameters...")
op.assign_control_parameters(optimised_value)
print_output("Run to plot optimised timeseries...")
swp.solve_forward()

# Save timeseries to file
for gauge in gauges:
    fname = os.path.join(di, '{:s}_{:s}_{:d}'.format(gauge, timeseries, level))
    np.save(fname, op.gauges[gauge][timeseries])

# Solve adjoint problem and plot solution fields
if plot_pvd:
    swp.solve_adjoint()
    swp.save_adjoint_trajectory()
