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

import numpy as np
import os
import sys

from adapt_utils.argparse import ArgumentParser
from adapt_utils.case_studies.tohoku.options.radial_options import TohokuRadialBasisOptions
from adapt_utils.unsteady.solver_adjoint import AdaptiveDiscreteAdjointProblem


class DiscreteAdjointTsunamiProblem(AdaptiveDiscreteAdjointProblem):
    """
    The subclass exists to pass the QoI as required.
    """
    def quantity_of_interest(self):
        return self.op.J


# --- Parse arguments

parser = ArgumentParser(shallow_water=True)

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
timeseries_type = 'timeseries'
if bool(args.continuous_timeseries or False):
    timeseries_type = '_'.join([timeseries_type, 'smooth'])

# Setup output directories
dirname = os.path.dirname(__file__)
di = create_directory(os.path.join(dirname, 'outputs', 'synthetic'))
if args.extension is not None:
    di = '_'.join([di, args.extension])
di = create_directory(os.path.join(di, 'discrete'))

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
        fname = os.path.join(di, '{:s}_data_{:d}.npy'.format(gauge, level))
        np.save(fname, op.gauges[gauge]['data'])

# Explore parameter space
n = 8
op.save_timeseries = False
control_values = [[m, ] for m in np.linspace(0.5, 7.5, n)]
if recompute:
    fname = os.path.join(di, 'parameter_space_{:d}.npy'.format(level))
    msg = "{:2d}: control value {:.4e}  functional value {:.4e}"
    func_values = np.zeros(n)
    with stop_annotating():
        swp = DiscreteAdjointTsunamiProblem(op, nonlinear=nonlinear, print_progress=False)
        for i, m in enumerate(control_values):
            op.assign_control_parameters(m, mesh=swp.meshes[0])
            swp.solve_forward()
            func_values[i] = swp.quantity_of_interest()
            print_output(msg.format(i, m[0], func_values[i]))
    np.save(fname, func_values)


# --- Tracing

# Set initial guess
op.save_timeseries = True
swp = DiscreteAdjointTsunamiProblem(op, nonlinear=nonlinear, print_progress=False)
print_output("Clearing tape...")
swp.clear_tape()
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
    fname = os.path.join(di, '{:s}_{:s}_{:d}'.format(gauge, timeseries_type, level))
    np.save(fname, op.gauges[gauge][timeseries_type])


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
    control_values_opt = [op.control_parameters[0], ]
    func_values_opt = [J, ]
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

    # Run BFGS optimisation
    opt_kwargs = {'maxiter': 1000, 'gtol': gtol}
    print_output("Optimisation begin...")
    Jhat = ReducedFunctional(J, control, derivative_cb_post=derivative_cb_post)
    optimised_value = minimize(Jhat, method='BFGS', options=opt_kwargs).dat.data

# Run forward again so that we can compare timeseries
op.plot_pvd = plot_pvd
swp = DiscreteAdjointTsunamiProblem(op_opt, nonlinear=nonlinear, print_progress=False)
print_output("Clearing tape...")
swp.clear_tape()
print_output("Assigning optimised control parameters...")
op.assign_control_parameters(optimised_value)
print_output("Run to plot optimised timeseries...")
swp.solve_forward()
J = swp.quantity_of_interest()

# Save timeseries to file
for gauge in gauges:
    fname = os.path.join(di, '{:s}_{:s}_{:d}'.format(gauge, timeseries_type, level))
    np.save(fname, op_opt.gauges[gauge][timeseries_type])

# Solve adjoint problem and plot solution fields
if plot_pvd:
    swp.solve_adjoint()
    swp.save_adjoint_trajectory()
