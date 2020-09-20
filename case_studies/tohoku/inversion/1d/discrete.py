"""
Invert for an initial condition defined over a (Gaussian) radial basis with a single basis function.
If we have N_g gauges and N_T timesteps then we have N_g*N_T data points we would like to fit using
a least squares fit. If N_g = 15 and N_T = 288 (as below) then we have 4320 data points.
Compared with the single control parameter, this implies a massively overconstrained problem!

[In practice the number of data points is smaller because we do not try to fit the gauge data in
the period before the tsunami wave arrives.]

A 'synthetic' tsunami is generated from an initial condition given by the 'optimal' scaling
parameter is m = 5. We apply PDE constrained optimisation with an initial guess m = 10.

In this script, we use the discrete adjoint approach to approximate the gradient of J w.r.t. m.
"""
from thetis import *
from firedrake_adjoint import *

import matplotlib.pyplot as plt
import numpy as np
import os
import scipy
import sys

from adapt_utils.argparse import ArgumentParser
from adapt_utils.case_studies.tohoku.options.radial_options import TohokuRadialBasisOptions
from adapt_utils.optimisation import StagnationError
from adapt_utils.plotting import *
from adapt_utils.unsteady.solver_adjoint import AdaptiveDiscreteAdjointProblem


class DiscreteAdjointTsunamiProblem(AdaptiveDiscreteAdjointProblem):
    """
    The subclass exists to pass the QoI as required.
    """
    def quantity_of_interest(self):
        return self.op.J


# --- Parse arguments

parser = ArgumentParser(plotting=True, shallow_water=True)

# Resolution
parser.add_argument("-level", help="Mesh resolution level")

# Inversion
parser.add_argument("-rerun_optimisation", help="Rerun optimisation routine")
parser.add_argument("-recompute_parameter_space", help="Recompute parameter space")
parser.add_argument("-initial_guess", help="Initial guess for control parameter")
parser.add_argument("-optimal_control", help="Artificially choose an optimum to invert for")
parser.add_argument("-continuous_timeseries", help="Toggle discrete or continuous timeseries data")
parser.add_argument("-gtol", help="Gradient tolerance (default 1.0e-08)")

# Testing
parser.add_argument("-end_time", help="End time of simulation (to shorten Taylor test)")
parser.add_argument("-taylor_test", help="Toggle Taylor testing")


# --- Set parameters

# Parsed arguments
args = parser.parse_args()
level = int(args.level or 0)
recompute = bool(args.recompute_parameter_space or False)
optimise = bool(args.rerun_optimisation or False)
gtol = float(args.gtol or 1.0e-08)
plot_pvd = bool(args.plot_pvd or False)
plot_pdf = bool(args.plot_pdf or False)
plot_png = bool(args.plot_png or False)
plot_all = bool(args.plot_all or False)
plot_only = bool(args.plot_only or False)
if plot_only:
    plot_all = True
if plot_all:
    plot_pvd = plot_pdf = plot_png = True
extensions = []
if plot_pdf:
    extensions.append('pdf')
if plot_png:
    extensions.append('png')
plot_any = len(extensions) > 0
if optimise or recompute:
    assert not plot_only
timeseries_type = "timeseries"
if bool(args.continuous_timeseries or False):
    timeseries_type = "_".join([timeseries_type, "smooth"])

# Do not attempt to plot in parallel
if COMM_WORLD.size > 1 and plot_any:
    print_output(120*'*' + "\nWARNING: Plotting turned off when running in parallel.\n" + 120*'*')
    plot_pdf = plot_png = False

# Setup output directories
dirname = os.path.dirname(__file__)
di = create_directory(os.path.join(dirname, 'outputs', 'synthetic'))
if args.extension is not None:
    di = '_'.join([di, args.extension])
di = create_directory(os.path.join(di, 'discrete'))
plot_dir = create_directory(os.path.join(di, 'plots'))
create_directory(os.path.join(plot_dir, 'discrete'))

# Collect initialisation parameters
nonlinear = bool(args.nonlinear or False)
family = args.family or 'dg-cg'
stabilisation = args.stabilisation or 'lax_friedrichs'
if stabilisation == 'none' or family == 'cg-cg' or not nonlinear:
    stabilisation = None
taylor = bool(args.taylor_test or False)
kwargs = {
    'level': level,
    'save_timeseries': True,

    # Spatial discretisation
    'family': family,
    'stabilisation': stabilisation,
    'use_automatic_sipg_parameter': False,  # the problem is inviscid

    # Optimisation
    'control_parameters': [float(args.initial_guess or 7.5), ],
    'synthetic': True,
    'qoi_scaling': 1.0,
    'nx': 1,
    'ny': 1,

    # Misc
    'plot_pvd': False,
    'debug': bool(args.debug or False),
    'debug_mode': args.debug_mode or 'basic',
    'di': di,
}
op = TohokuRadialBasisOptions(**kwargs)
gauges = list(op.gauges.keys())

# Plotting parameters  # TODO: Move elsewhere
fontsize = 22
fontsize_tick = 18
plotting_kwargs = {'markevery': 5}

# Synthetic run
try:
    fnames = [os.path.join(di, '{:s}_data_{:d}.npy'.format(gauge, level)) for gauge in gauges]
    assert np.all([os.path.isfile(fname) for fname in fnames])
    for fname, gauge in zip(fnames, gauges):
        op.gauges[gauge]['data'] = np.load(fname)
except AssertionError:
    print_output("Run forward to get 'data'...")
    with stop_annotating():
        swp = DiscreteAdjointTsunamiProblem(op, nonlinear=nonlinear, print_progress=False)
        control_value = [float(args.optimal_control or 5.0), ]
        op.assign_control_parameters(control_value, mesh=swp.meshes[0])
        swp.solve_forward()
    for gauge in gauges:
        op.gauges[gauge]['data'] = op.gauges[gauge][timeseries_type]
        fname = os.path.join(di, '_'.join([gauge, 'data', str(level)]))
        np.save(fname, op.gauges[gauge]['data'])

# Explore parameter space
n = 8
op.save_timeseries = False
control_values = [[m, ] for m in np.linspace(0.5, 7.5, n)]
fname = os.path.join(di, 'parameter_space_{:d}.npy'.format(level))
msg = "{:2d}: control value {:.4e}  functional value {:.4e}"
if recompute:
    with stop_annotating():
        func_values = np.zeros(n)
        swp = DiscreteAdjointTsunamiProblem(op, nonlinear=nonlinear, print_progress=False)
        for i, m in enumerate(control_values):
            op.assign_control_parameters(m, mesh=swp.meshes[0])
            swp.solve_forward()
            func_values[i] = swp.quantity_of_interest()
            print_output(msg.format(i, m[0], func_values[i]))
        np.save(fname, func_values)
    if plot_any:  # TODO: Move elsewhere
        print_output("Explore parameter space...")
        fig, axes = plt.subplots(figsize=(8, 8))
        axes.plot(control_values, func_values, '--x', linewidth=2, markersize=8, markevery=10)
        axes.set_xlabel("Basis function coefficient", fontsize=fontsize)
        axes.set_ylabel("Mean square error quantity of interest", fontsize=fontsize)
        plt.xticks(fontsize=fontsize_tick)
        plt.yticks(fontsize=fontsize_tick)
        axes.grid()
        savefig('parameter_space_{:d}'.format(level), plot_dir, extensions=extensions)


# --- Tracing

if plot_only:

    # Load timeseries  # TODO: Move elsewhere
    for gauge in gauges:
        fname = os.path.join(di, '_'.join([gauge, timeseries_type, str(level) + '.npy']))
        op.gauges[gauge][timeseries_type] = np.load(fname)

else:
    op.save_timeseries = True
    swp = DiscreteAdjointTsunamiProblem(op, nonlinear=nonlinear, print_progress=False)

    # Set initial guess
    print_output("Clearing tape...")
    get_working_tape().clear_tape()
    print_output("Setting initial guess...")
    control_value = [float(args.initial_guess or 7.5), ]
    op.assign_control_parameters(control_value, mesh=swp.meshes[0])
    control = Control(op.control_parameters[0])

    # Solve the forward problem
    print_output("Run forward to get timeseries...")
    swp.solve_forward()
    J = swp.quantity_of_interest()

    # Save timeseries
    for gauge in gauges:
        fname = os.path.join(di, '_'.join([gauge, timeseries_type, str(level)]))
        np.save(fname, op.gauges[gauge][timeseries_type])

# Plot timeseries  # TODO: Move elsewhere
if plot_any:
    N = int(np.ceil(np.sqrt(len(gauges))))
    fig, axes = plt.subplots(nrows=N, ncols=N, figsize=(14, 12))
    for i, gauge in enumerate(gauges):
        T = np.array(op.gauges[gauge]['times'])/60
        ax = axes[i//N, i % N]
        ax.plot(T, op.gauges[gauge]['data'], '--x', label=gauge + ' data', **plotting_kwargs)
        ax.plot(T, op.gauges[gauge][timeseries_type], '--x', label=gauge + ' simulated', **plotting_kwargs)
        ax.legend(loc='upper left')
        ax.set_xlabel('Time (min)', fontsize=fontsize)
        ax.set_ylabel('Elevation (m)', fontsize=fontsize)
        plt.xticks(fontsize=fontsize_tick)
        plt.yticks(fontsize=fontsize_tick)
        ax.grid()
    for i in range(len(gauges), N*N):
        axes[i//N, i % N].axis(False)
    savefig('timeseries_{:d}'.format(level), plot_dir, extensions=extensions)


# --- Taylor test

if taylor:
    Jhat = ReducedFunctional(J, control)

    def get_control(mode):
        """
        Get control parameter values assuming one of three modes:

          'init' - control parameter is set to the initial guess;
          'random' - control parameter is set randomly with a Normal distribution;
          'optimised' - optimal control parameter from a previous run is used.
        """
        c = Function(op.control_parameters[0])
        if mode == 'init':
            pass
        elif mode == 'random':
            c.dat.data[0] += np.random.random() - 0.5
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

    # Ensure consistency of tests
    np.random.seed(0)

    # Random search direction
    dc = Function(op.control_parameters[0])
    dc.dat.data[0] = np.random.random() - 0.5

    # Run tests
    for mode in ("init", "random", "optimised"):
        print_output("Taylor test '{:s}' begin...".format(mode))
        minconv = taylor_test(Jhat, get_control(mode), dc)
        if minconv > 1.90:
            print_output("Taylor test '{:s}' passed!".format(mode))
        else:
            msg = "Taylor test '{:s}' failed! Convergence ratio {:.2f} < 2."
            raise ConvergenceError(msg.format(mode, minconv))
    sys.exit(0)


# --- Optimisation

fname = os.path.join(di, 'discrete', 'optimisation_progress_{:s}' + '_{:d}.npy'.format(level))
if np.all([os.path.exists(fname.format(ext)) for ext in ('ctrl', 'func', 'grad')]) and not optimise:

    # Load trajectory
    control_values_opt = np.load(fname.format('ctrl', level))
    func_values_opt = np.load(fname.format('func', level))
    gradient_values_opt = np.load(fname.format('grad', level))
    optimised_value = control_values_opt[-1]

else:

    # Arrays to log progress
    control_values_opt = []
    func_values_opt = []
    gradient_values_opt = []

    def derivative_cb_post(j, dj, m):
        control = m.dat.data[0]
        djdm = dj.dat.data[0]
        print_output("control {:.8e}  functional {:.8e}  gradient {:.8e}".format(control, j, djdm))

        # Save progress to NumPy arrays on-the-fly
        control_values_opt.append(control)
        func_values_opt.append(j)
        gradient_values_opt.append(djdm)
        np.save(fname.format('ctrl'), np.array(control_values_opt))
        np.save(fname.format('func'), np.array(func_values_opt))
        np.save(fname.format('grad'), np.array(gradient_values_opt))

        # Stagnation termination condition
        if len(func_values_opt) > 1:
            if abs(func_values_opt[-1] - func_values_opt[-2]) < 1.0e-06*abs(func_values_opt[-2]):
                raise StagnationError

    # Run BFGS optimisation
    opt_kwargs = {
        'maxiter': 100,
        'gtol': gtol,
    }
    print_output("Optimisation begin...")
    Jhat = ReducedFunctional(J, Control(op.control_parameters[0]), derivative_cb_post=derivative_cb_post)
    try:
        optimised_value = minimize(Jhat, method='BFGS', options=opt_kwargs).dat.data[0]
    except StagnationError:
        optimised_value = control_values_opt[-1]
        print_output("StagnationError: Stagnation of objective functional")

if plot_any:  # TODO: Move elsewhere
    func_values = np.load(fname)
    for i, m in enumerate(control_values):
        print_output(msg.format(i, m[0], func_values[i]))

    # Fit a quadratic to the first three points and find its root
    assert len(control_values[::3]) == 3
    q = scipy.interpolate.lagrange(control_values[::3], func_values[::3])
    dq = q.deriv()
    q_min = -dq.coefficients[1]/dq.coefficients[0]
    assert dq.deriv().coefficients[0] > 0
    print_output("Minimiser of quadratic: {:.4f}".format(q_min))
    assert np.isclose(dq(q_min), 0.0)

    # Plot progress of optimisation routine
    fig, axes = plt.subplots(figsize=(8, 8))
    params = {'linewidth': 1, 'markersize': 8, 'color': 'C0', 'markevery': 10, 'label': 'Parameter space', }
    x = np.linspace(control_values[0], control_values[-1], 10*len(control_values))
    axes.plot(x, q(x), '--x', **params)
    params = {'markersize': 14, 'color': 'C0', 'label': r'$m^\star = {:.4f}$'.format(q_min), }
    axes.plot(q_min, q(q_min), '*', **params)
    params = {'markersize': 8, 'color': 'C1', 'label': 'Optimisation progress', }
    axes.plot(control_values_opt, func_values_opt, 'o', **params)
    delta_m = 0.25
    params = {'linewidth': 3, 'markersize': 8, 'color': 'C2', }
    for m, f, g in zip(control_values_opt, func_values_opt, gradient_values_opt):
        x = np.array([m - delta_m, m + delta_m])
        axes.plot(x, g*(x-m) + f, '-', **params)
    params['label'] = 'Computed gradient'
    axes.plot(x, g*(x-m) + f, '-', **params)
    axes.set_xlabel(r"Basis function coefficient, $m$", fontsize=fontsize)
    axes.set_ylabel(r"Mean square error", fontsize=fontsize)
    plt.xticks(fontsize=fontsize_tick)
    plt.yticks(fontsize=fontsize_tick)
    plt.xlim([0, 8])
    axes.grid()
    plt.legend(fontsize=fontsize)
    axes.annotate(
        r'$m = {:.4f}$'.format(control_values_opt[-1]),
        xy=(q_min+2, func_values_opt[-1]), color='C1', fontsize=fontsize
    )
    fname = 'optimisation_progress_{:d}'.format(level)
    savefig(os.path.join(plot_dir, 'discrete'), extensions=extensions)

# Create a new parameter class
kwargs['control_parameters'] = [optimised_value, ]
kwargs['plot_pvd'] = plot_pvd
op_opt = TohokuRadialBasisOptions(**kwargs)

if plot_only:

    # Load timeseries
    for gauge in gauges:
        fname = os.path.join(di, '_'.join([gauge, timeseries_type, str(level) + '.npy']))
        op_opt.gauges[gauge][timeseries_type] = np.load(fname)

else:
    get_working_tape().clear_tape()

    # Run forward again so that we can compare timeseries
    op_opt.gauges[gauge]["data"] = op.gauges[gauge]["data"]
    print_output("Run to plot optimised timeseries...")
    swp = DiscreteAdjointTsunamiProblem(op_opt, nonlinear=nonlinear, print_progress=False)
    swp.solve_forward()
    J = swp.quantity_of_interest()

    # Save timeseries
    for gauge in gauges:
        fname = os.path.join(di, '_'.join([gauge, timeseries_type, str(level)]))
        np.save(fname, op_opt.gauges[gauge][timeseries_type])

    # Solve adjoint problem and plot solution fields
    if plot_pvd:
        swp.solve_adjoint()
        swp.get_solve_blocks()
        swp.save_adjoint_trajectory()

# Plot timeseries for both initial guess and optimised control
if plot_any:  # TODO: Move elsewhere
    fig, axes = plt.subplots(nrows=N, ncols=N, figsize=(14, 12))
    for i, gauge in enumerate(gauges):
        T = np.array(op.gauges[gauge]['times'])/60
        ax = axes[i//N, i % N]
        ax.plot(T, op.gauges[gauge]['data'], '--x', label=gauge + ' data', **plotting_kwargs)
        ax.plot(T, op.gauges[gauge][timeseries_type], '--x', label=gauge + ' initial guess', **plotting_kwargs)
        ax.plot(T, op_opt.gauges[gauge][timeseries_type], '--x', label=gauge + ' optimised', **plotting_kwargs)
        ax.legend(loc='upper left')
        ax.set_xlabel('Time (min)', fontsize=fontsize)
        ax.set_ylabel('Elevation (m)', fontsize=fontsize)
        plt.xticks(fontsize=fontsize_tick)
        plt.yticks(fontsize=fontsize_tick)
        ax.grid()
    for i in range(len(gauges), N*N):
        axes[i//N, i % N].axis(False)
    fname = 'timeseries_optimised_{:d}'.format(level)
    savefig(fname, os.path.join(plot_dir, 'discrete'), extensions=extensions)
