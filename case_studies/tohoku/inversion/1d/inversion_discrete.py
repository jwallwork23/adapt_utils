"""
Invert for an initial condition defined over a Gaussian radial basis with a single basis function.
If we have N_g gauges and N_T timesteps then we have N_g*N_T data points we would like to fit using
a least squares fit. If N_g = 15 and N_T = 288 (as below) then we have 4320 data points.
Compared with the single control parameter, this implies a massively overconstrained problem!

[In practice the number of data points is smaller because we do not try to fit the gauge data in
the period before the tsunami wave arrives.]

This script allows for two modes, determined by the `real_data` argument.

* `real_data == True`: gauge data are used.
* `real_data == False`: a 'synthetic' tsunami is generated from an initial condition given by the
  'optimal' scaling parameter is m = 5.

In each case, we apply PDE constrained optimisation with an initial guess m = 10.

In this script, we use the discrete adjoint approach to approximate the gradient of J w.r.t. m.
"""
from thetis import *
from firedrake_adjoint import *

import argparse
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy
import os

from adapt_utils.unsteady.solver import AdaptiveProblem
from adapt_utils.unsteady.solver_adjoint import AdaptiveDiscreteAdjointProblem
from adapt_utils.case_studies.tohoku.options.gaussian_options import TohokuGaussianBasisOptions
from adapt_utils.misc import StagnationError
from adapt_utils.norms import total_variation


# --- Parse arguments

parser = argparse.ArgumentParser()

# Model
parser.add_argument("-level", help="Mesh resolution level")
parser.add_argument("-family", help="Finite element pair")
parser.add_argument("-stabilisation", help="Stabilisation approach")
parser.add_argument("-nonlinear", help="Toggle nonlinear model")

# Inversion
parser.add_argument("-initial_guess", help="Initial guess for control parameter")
parser.add_argument("-optimal_control", help="Artificially choose an optimum to invert for")
parser.add_argument("-recompute_parameter_space", help="Recompute parameter space")
parser.add_argument("-rerun_optimisation", help="Rerun optimisation routine")
parser.add_argument("-real_data", help="Toggle whether to use real data")
parser.add_argument("-smooth_timeseries", help="Toggle discrete or smoothed timeseries data")

# I/O and debugging
parser.add_argument("-plot_only", help="Just plot parameter space and optimisation progress")
parser.add_argument("-plot_pvd", help="Toggle plotting to .pvd")
parser.add_argument("-debug", help="Toggle debugging")

# --- Set parameters

# Fonts
matplotlib.rc('text', usetex=True)
matplotlib.rcParams['mathtext.fontset'] = 'custom'
matplotlib.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
matplotlib.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
matplotlib.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'

# Plotting
fontsize = 22
fontsize_tick = 18
plotting_kwargs = {'markevery': 5}

# Parsed arguments
args = parser.parse_args()
level = int(args.level or 0)
recompute = bool(args.recompute_parameter_space or False)
optimise = bool(args.rerun_optimisation or False)
plot_only = bool(args.plot_only or False)
if optimise or recompute:
    assert not plot_only
plot_pvd = bool(args.plot_pvd or False)
real_data = bool(args.real_data or False)
if args.optimal_control is not None:
    assert not real_data
use_smoothed_timeseries = bool(args.smooth_timeseries or False)
timeseries_type = "timeseries"
if use_smoothed_timeseries:
    timeseries_type = "_".join([timeseries_type, "smooth"])
kwargs = {
    'level': level,
    'save_timeseries': True,

    # Spatial discretisation
    'family': args.family or 'dg-cg',
    'stabilisation': args.stabilisation,
    'use_automatic_sipg_parameter': False,  # the problem is inviscid

    # Optimisation
    'control_parameters': [float(args.initial_guess or 10.0), ],
    'synthetic': not real_data,
    'qoi_scaling': 1.0,

    # Misc
    'plot_pvd': False,
    'debug': bool(args.debug or False),
}
nonlinear = bool(args.nonlinear or False)
op = TohokuGaussianBasisOptions(**kwargs)

# Setup output directories
dirname = os.path.dirname(__file__)
di = create_directory(os.path.join(dirname, 'outputs', 'realistic' if real_data else 'synthetic'))
op.di = create_directory(os.path.join(di, 'discrete'))
plot_dir = create_directory(os.path.join(di, 'plots'))
create_directory(os.path.join(plot_dir, 'discrete'))

# Synthetic run
if not real_data:
    if not plot_only:
        with stop_annotating():
            op.control_parameters[0].assign(float(args.optimal_control or 5.0))
            swp = AdaptiveProblem(op, nonlinear=nonlinear)
            swp.solve_forward()
            for gauge in op.gauges:
                op.gauges[gauge]["data"] = op.gauges[gauge][timeseries_type]

# Explore parameter space
n = 9
op.save_timeseries = False
control_values = np.linspace(2.0, 10.0, n)
fname = os.path.join(di, 'parameter_space_{:d}.npy'.format(level))
recompute |= not os.path.exists(fname)
with stop_annotating():
    if recompute:
        func_values = np.zeros(n)
        swp = AdaptiveProblem(op, nonlinear=nonlinear)
        for i, m in enumerate(control_values):
            op.control_parameters[0].assign(m)
            swp.set_initial_condition()
            swp.solve_forward()
            func_values[i] = op.J
    else:
        func_values = np.load(fname)
    np.save(fname, func_values)
    op.control_parameters[0].assign(float(args.initial_guess or 10.0))
for i, m in enumerate(control_values):
    print_output("{:2d}: control value {:.4e}  functional value {:.4e}".format(i, m, func_values[i]))

# Plot parameter space
if recompute:
    fig, axes = plt.subplots(figsize=(8, 8))
    axes.plot(control_values, func_values, '--x', linewidth=2, markersize=8, markevery=10)
    axes.set_xlabel("Basis function coefficient", fontsize=fontsize)
    axes.set_ylabel("Mean square error quantity of interest", fontsize=fontsize)
    plt.xticks(fontsize=fontsize_tick)
    plt.yticks(fontsize=fontsize_tick)
    plt.tight_layout()
    axes.grid()
    plt.savefig(os.path.join(plot_dir, 'parameter_space_{:d}.pdf'.format(level)))

# --- Optimisation

if not plot_only:

    # Solve the forward problem with some initial guess
    op.save_timeseries = True
    swp = AdaptiveProblem(op, nonlinear=nonlinear)
    swp.solve_forward()
    J = op.J
    print_output("Mean square error QoI = {:.4e}".format(J))

    # Plot timeseries
    gauges = list(op.gauges.keys())
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
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'timeseries_{:d}.pdf'.format(level)))

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
        gradient = dj.dat.data[0]
        print_output("control {:.8e}  J {:.8e}  gradient {:.8e}".format(control, j, gradient))

        # Save progress to NumPy arrays on-the-fly
        control_values_opt.append(control)
        func_values_opt.append(j)
        gradient_values_opt.append(gradient)
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
        'gtol': 1.0e-08,
    }
    Jhat = ReducedFunctional(J, Control(op.control_parameters[0]), derivative_cb_post=derivative_cb_post)
    try:
        optimised_value = minimize(Jhat, method='BFGS', options=opt_kwargs).dat.data[0]
    except StagnationError:
        optimised_value = control_values_opt[-1]
        print_output("StagnationError: Stagnation of objective functional")

# Fit a quadratic to the first three points and find its root
assert len(control_values[::4]) == 3
q = scipy.interpolate.lagrange(control_values[::4], func_values[::4])
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
axes.set_ylabel(r"Square error", fontsize=fontsize)
plt.xticks(fontsize=fontsize_tick)
plt.yticks(fontsize=fontsize_tick)
plt.xlim([1.5, 10.5])
plt.tight_layout()
axes.grid()
plt.legend(fontsize=fontsize)
axes.annotate(
    r'$m = {:.4f}$'.format(control_values_opt[-1]),
    xy=(q_min+2, func_values_opt[-1]), color='C1', fontsize=fontsize
)
plt.savefig(os.path.join(plot_dir, 'discrete', 'optimisation_progress_{:d}.pdf'.format(level)))

if not plot_only:
    tape = get_working_tape()
    tape.clear_tape()

    class DiscreteAdjointTsunamiProblem(AdaptiveDiscreteAdjointProblem):
        """The subclass exists to pass the QoI as required."""
        def quantity_of_interest(self):
            return self.op.J

    # Run forward again so that we can compare timeseries
    kwargs['control_parameters'] = [optimised_value, ]
    kwargs['plot_pvd'] = plot_pvd
    op_opt = TohokuGaussianBasisOptions(**kwargs)
    gauges = list(op_opt.gauges.keys())
    for gauge in gauges:
        op_opt.gauges[gauge]["data"] = op.gauges[gauge]["data"]
    swp = DiscreteAdjointTsunamiProblem(op_opt, nonlinear=nonlinear)
    swp.solve_forward()
    J = swp.quantity_of_interest()
    print_output("Mean square error QoI after optimisation = {:.4e}".format(J))

    # Plot timeseries for both initial guess and optimised control
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
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'discrete', 'timeseries_optimised_{:d}.pdf'.format(level)))

    # Compare total variation
    msg = "total variation for gauge {:s}: before {:.4e}  after {:.4e} reduction  {:.1f}%"
    print_output("\nContinuous form QoI:")
    for gauge in op.gauges:
        tv = total_variation(op.gauges[gauge]['diff_smooth'])
        tv_opt = total_variation(op_opt.gauges[gauge]['diff_smooth'])
        print_output(msg.format(gauge, tv, tv_opt, 100*(1-tv_opt/tv)))
    print_output("\nDiscrete form QoI:")
    for gauge in op.gauges:
        tv = total_variation(op.gauges[gauge]['diff'])
        tv_opt = total_variation(op_opt.gauges[gauge]['diff'])
        print_output(msg.format(gauge, tv, tv_opt, 100*(1-tv_opt/tv)))

    # Solve adjoint problem and plot solution fields
    if plot_pvd:
        swp.compute_gradient(Control(op_opt.control_parameters[0]))
        swp.get_solve_blocks()
        swp.save_adjoint_trajectory()
