"""
Source inversion for a 'synthetic' tsunami generated from an initial condition given by scaling a
single Gaussian basis function. Given that the 'optimal' scaling parameter is m = 5, we seek to
find this value through PDE constrained optimisation with an initial guess m = 10. What is 'optimal'
is determined via an objective functional J which quantifies the misfit at timeseries with those data
recorded under the m = 5 case.

In this script, we use the continuous adjoint approach to approximate the gradient of J w.r.t. m.
We observe the phenomenon of inconsistent gradients for the continuous adjoint approach.
"""
from thetis import *

import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import scipy

from adapt_utils.unsteady.solver import AdaptiveProblem
from adapt_utils.case_studies.tohoku.options import *
from adapt_utils.misc import StagnationError
from adapt_utils.norms import total_variation


parser = argparse.ArgumentParser()
parser.add_argument("-level", help="Mesh resolution level")
parser.add_argument("-family", help="Finite element pair")
parser.add_argument("-initial_guess", help="Initial guess for control parameter")
parser.add_argument("-optimal_control", help="Artificially choose an optimum to invert for")
parser.add_argument("-regularisation", help="Parameter for Tikhonov regularisation term")
parser.add_argument("-recompute_parameter_space", help="Recompute parameter space")
parser.add_argument("-rerun_optimisation", help="Rerun optimisation routine")
parser.add_argument("-plot_only", help="Just plot parameter space and optimisation progress")
parser.add_argument("-plot_pvd", help="Toggle plotting to .pvd")
parser.add_argument("-debug", help="Toggle debugging")
args = parser.parse_args()

# Set parameters
level = int(args.level or 0)
recompute = bool(args.recompute_parameter_space or False)
optimise = bool(args.rerun_optimisation or False)
plot_only = bool(args.plot_only or False)
if optimise or recompute:
    assert not plot_only
plot_pvd = bool(args.plot_pvd or False)
kwargs = {
    'level': level,
    'save_timeseries': True,

    # Spatial discretisation
    'family': args.family or 'cg-cg',
    # 'stabilisation': 'lax_friedrichs',
    'stabilisation': None,
    'use_automatic_sipg_parameter': False,  # the problem is inviscid

    # Optimisation
    'control_parameter': float(args.initial_guess or 10.0),
    'artificial': True,
    'qoi_scaling': 1.0e-12,
    'regularisation': float(args.regularisation or 0.0),

    # Misc
    'plot_pvd': False,
    'debug': bool(args.debug or False),
}
use_regularisation = not np.isclose(kwargs['regularisation'], 0.0)
nonlinear = False  # TODO
fontsize = 22
fontsize_tick = 18
plotting_kwargs = {
    'markevery': 5,
}
op = TohokuGaussianBasisOptions(**kwargs)
di = create_directory(os.path.join(op.di, 'plots'))

# Artifical run
if not plot_only:
    op.control_parameter.assign(float(args.optimal_control or 5.0))
    swp = AdaptiveProblem(op, nonlinear=nonlinear, checkpointing=False)
    swp.solve_forward()
    for gauge in op.gauges:
        op.gauges[gauge]["data"] = op.gauges[gauge]["timeseries"]

# Explore parameter space
n = 9
op.save_timeseries = False
control_values = np.linspace(2.0, 10.0, n)
fname = os.path.join(op.di, 'parameter_space_artificial_{:d}.npy'.format(level))
recompute |= not os.path.exists(fname)
if recompute:
    func_values = np.zeros(n)
    swp = AdaptiveProblem(op, nonlinear=nonlinear, checkpointing=False)
    for i, m in enumerate(control_values):
        op.control_parameter.assign(m)
        swp.set_initial_condition()
        swp.solve_forward()
        func_values[i] = op.J
else:
    func_values = np.load(fname)
np.save(fname, func_values)
msg = "{:2d}: control value {:.4e}  functional value {:.4e}"
for i, m in enumerate(control_values):
    print_output(msg.format(i, m, func_values[i]))

# Explore regularised parameter space
if use_regularisation:
    fname = os.path.join(op.di, 'parameter_space_artificial_reg_{:d}.npy'.format(level))
    recompute &= not os.path.exists(fname)
    if recompute:
        func_values_reg = np.zeros(n)
        swp = AdaptiveProblem(op, nonlinear=nonlinear, checkpointing=False)
        for i, m in enumerate(control_values):
            op.control_parameter.assign(m)
            swp.set_initial_condition()
            swp.solve_forward()
            func_values_reg[i] = op.J
    else:
        func_values_reg = np.load(fname)
    np.save(fname, func_values_reg)
    for i, m in enumerate(control_values):
        print_output(msg.format(i, m, func_values_reg[i]))

# Plot parameter space
if recompute:
    fig, axes = plt.subplots(figsize=(8, 8))
    axes.plot(control_values, func_values, '--x', linewidth=2, markersize=8)
    if use_regularisation:
        axes.plot(control_values, func_values_reg, '--x', linewidth=2, markersize=8)
    axes.set_xlabel("Basis function coefficient", fontsize=fontsize)
    axes.set_ylabel("Mean square error quantity of interest", fontsize=fontsize)
    plt.xticks(fontsize=fontsize_tick)
    plt.yticks(fontsize=fontsize_tick)
    plt.tight_layout()
    plt.grid()
    plt.savefig(os.path.join(op.di, 'plots', 'single_bf_parameter_space_artificial_{:d}.pdf'.format(level)))

# --- Optimisation

if not plot_only:
    swp = AdaptiveProblem(op, nonlinear=nonlinear, checkpointing=True)

    def reduced_functional(m):
        """
        The QoI, reinterpreted as a function of the control parameter `m` only. We apply PDE
        constrained optimisation in order to minimise this functional.

        Note that this involves checkpointing state.
        """
        op.control_parameter.assign(m[0])
        swp.solve_forward()
        J = op.J
        print_output("control = {:.8e}  functional = {:.8e}".format(m[0], J))
        return J


    def gradient(m):
        """
        Compute the gradient of the reduced functional with respect to the control parameter using
        data stored to the checkpoints.
        """
        if len(swp.checkpoint) == 0:
            reduced_functional(m)
        swp.solve_adjoint()
        g = assemble(inner(op.basis_function, swp.adj_solutions[0])*dx)
        if use_regularisation:
            g += op.regularisation_term_gradient
        print_output("control = {:.8e}  gradient = {:.8e}".format(m[0], g))
        return np.array([g, ])

    # Solve the forward problem with some initial guess
    swp.checkpointing = False
    op.save_timeseries = True
    m_init = np.array([float(args.initial_guess or 10.0), ])
    J = reduced_functional(m_init)
    print_output("Mean square error QoI = {:.4e}".format(J))
    op.save_timeseries = False
    swp.checkpointing = True

    # Plot timeseries
    gauges = list(op.gauges.keys())
    N = int(np.ceil(np.sqrt(len(gauges))))
    fig, axes = plt.subplots(nrows=N, ncols=N, figsize=(14, 12))
    T = np.array(op.times)/60
    for i, gauge in enumerate(gauges):
        ax = axes[i//N, i % N]
        ax.plot(T, op.gauges[gauge]['data'], '--x', label=gauge + ' data', **plotting_kwargs)
        ax.plot(T, op.gauges[gauge]['timeseries'], '--x', label=gauge + ' simulated', **plotting_kwargs)
        ax.legend(loc='upper left')
        ax.set_xlabel('Time (min)', fontsize=fontsize)
        ax.set_ylabel('Elevation (m)', fontsize=fontsize)
        plt.xticks(fontsize=fontsize_tick)
        plt.yticks(fontsize=fontsize_tick)
        plt.grid()
    for i in range(len(gauges) % N):
        axes[N-1, N-i-1].axes('off')
    plt.tight_layout()
    plt.savefig(os.path.join(di, 'single_bf_timeseries_artificial_{:d}.pdf'.format(level)))

# Get filename for tracking progress
fname = os.path.join(op.di, 'opt_progress_cts_{:s}_artificial')
if use_regularisation:
    fname = '_'.join([fname, 'reg'])
fname = '_'.join([fname, '_{:d}.npy'.format(level)])

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

    def reduced_functional_hat(m):
        """Modified reduced functional which stores progress and checks for stagnation."""
        control_values_opt.append(m[0])
        np.save(fname.format('ctrl'), np.array(control_values_opt))
        J = reduced_functional(m)
        func_values_opt.append(J)
        np.save(fname.format('func'), np.array(func_values_opt))

        # Stagnation termination condition
        if len(func_values_opt) > 1:
            if abs(func_values_opt[-1] - func_values_opt[-2]) < 1.0e-06*abs(func_values_opt[-2]):
                raise StagnationError
        return J

    def gradient_hat(m):
        """Modified gradient functional which stores progress"""
        if len(swp.checkpoint) == 0:
            scaled_reduced_functional_hat(m)
        g = gradient(m)
        gradient_values_opt.append(g[0])
        np.save(fname.format('grad'), np.array(gradient_values_opt))
        return g

    # Run BFGS optimisation
    opt_kwargs = {
        'maxiter': 100,
        'gtol': 1.0e-08,
        'callback': lambda m: print_output("LINE SEARCH COMPLETE"),
        'fprime': gradient_hat,
    }
    m_init = op.control_parameter.dat.data
    try:
        m_opt = scipy.optimize.fmin_bfgs(reduced_functional_hat, m_init, **opt_kwargs)
        optimised_value = m_opt.dat.data[0]
    except StagnationError:
        optimised_value = control_values_opt[-1]
        print_output("StagnationError: Stagnation of objective functional")

# try:
#     assert len(control_values_opt) == len(gradient_values_opt) + 1
#     assert len(func_values_opt) == len(gradient_values_opt) + 1
# except AssertionError:
#     lengths = (len(func_values_opt), len(gradient_values_opt), len(control_values_opt))
#     raise ValueError("Inconsistent stored data: {:d} vs {:d} vs {:d}".format(*lengths))

# Fit a quadratic to the first three points and find its root
assert len(control_values[::4]) == 3
q = scipy.interpolate.lagrange(control_values[::4], func_values[::4])
dq = q.deriv()
q_min = -dq.coefficients[1]/dq.coefficients[0]
assert dq.deriv().coefficients[0] > 0
print_output("Minimiser of quadratic: {:.4f}".format(q_min))
assert np.isclose(dq(q_min), 0.0)

# Fit quadratic to regularised functional values
if use_regularisation:
    q_reg = scipy.interpolate.lagrange(control_values[::4], func_values_reg[::4])
    dq_reg = q_reg.deriv()
    q_reg_min = -dq_reg.coefficients[1]/dq_reg.coefficients[0]
    assert dq_reg.deriv().coefficients[0] > 0
    print_output("Minimiser of quadratic (regularised): {:.4f}".format(q_reg_min))
    assert np.isclose(dq_reg(q_reg_min), 0.0)
    params = {'linewidth': 1, 'markersize': 8, 'color': 'C6', 'label': 'Fitted quadratic (reg)', }
    x = np.linspace(control_values_opt[0], control_values_opt[-1], 10*len(control_values))
    axes.plot(x, q_reg(x), '--', **params)
    params = {'markersize': 14, 'color': 'C6', 'label': 'Minimum of quadratic (reg)', }
    axes.plot(q_reg_min, q_reg(q_reg_min), '*', **params)

# Plot progress of optimisation routine
fig, axes = plt.subplots(figsize=(8, 8))
params = {'linewidth': 3, 'markersize': 8, 'color': 'C0', 'label': 'Parameter space', }
axes.plot(control_values, func_values, 'x', **params)
x = np.linspace(control_values[0], control_values[-1], 10*len(control_values))
params = {'linewidth': 1, 'markersize': 8, 'color': 'C0', 'label': 'Fitted quadratic', }
axes.plot(x, q(x), '--', **params)
params = {'markersize': 14, 'color': 'C0', 'label': 'Minimum of quadratic', }
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
axes.set_xlabel("Basis function coefficient", fontsize=fontsize)
axes.set_ylabel("Scaled mean square error", fontsize=fontsize)
plt.xticks(fontsize=fontsize_tick)
plt.yticks(fontsize=fontsize_tick)
plt.xlim([1.5, 10.5])
# plt.ylim([0.0, 1.1*func_values[-1]])
plt.tight_layout()
plt.grid()
plt.legend(fontsize=fontsize)
opt = control_values_opt[-1]
axes.annotate('m = {:.2f}'.format(opt),
    xy=(opt-0.5, func_values_opt[-1]+0.1**level), color='C1', fontsize=fontsize)
plt.savefig(os.path.join(di, 'single_bf_optimisation_continuous_artificial_{:d}.pdf'.format(level)))

if not plot_only:

    # Run forward again so that we can compare timeseries
    kwargs['control_parameter'] = optimised_value
    kwargs['plot_pvd'] = plot_pvd
    op_opt = TohokuGaussianBasisOptions(**kwargs)
    gauges = list(op_opt.gauges.keys())
    for gauge in gauges:
        op_opt.gauges[gauge]["data"] = op.gauges[gauge]["data"]
    swp = AdaptiveProblem(op_opt, nonlinear=nonlinear, checkpointing=plot_pvd)
    swp.solve_forward()
    J = op.J
    print_output("Mean square error QoI after optimisation = {:.4e}".format(J))

    # Plot timeseries for both initial guess and optimised control
    fig, axes = plt.subplots(nrows=N, ncols=N, figsize=(14, 12))
    T = np.array(op.times)/60
    for i, gauge in enumerate(gauges):
        ax = axes[i//N, i % N]
        ax.plot(T, op.gauges[gauge]['data'], '--x', label=gauge + ' data', **plotting_kwargs)
        ax.plot(T, op.gauges[gauge]['timeseries'], '--x', label=gauge + ' initial_guess', **plotting_kwargs)
        ax.plot(T, op_opt.gauges[gauge]['timeseries'], '--x', label=gauge + ' optimised', **plotting_kwargs)
        ax.legend(loc='upper left')
        ax.set_xlabel('Time (min)', fontsize=fontsize)
        ax.set_ylabel('Elevation (m)', fontsize=fontsize)
        plt.xticks(fontsize=fontsize_tick)
        plt.yticks(fontsize=fontsize_tick)
        plt.grid()
    for i in range(len(gauges) % N):
        axes[N-1, N-i-1].axes('off')
    plt.tight_layout()
    plt.savefig(os.path.join(di, 'single_bf_timeseries_optimised_continuous_artificial_{:d}.pdf'.format(level)))

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

    if plot_pvd:
        swp.solve_adjoint()
