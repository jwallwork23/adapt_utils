from thetis import COMM_WORLD, create_directory, print_output

import matplotlib.pyplot as plt
import numpy as np
import os
import scipy
import sys

from adapt_utils.argparse import ArgumentParser
from adapt_utils.case_studies.tohoku.options.radial_options import TohokuRadialBasisOptions
from adapt_utils.plotting import *


# --- Parse arguments

parser = ArgumentParser(
    prog="plot_convergence",
    description="""
        Given tsunami source inversion run output, generate a variety of plots:
          (a) timeseries due to initial guess vs. synthetic data;
          (b) progress of the QoI during the optimisation, along with the approximate gradients;
          (c) timeseries due to converged control parameters vs. synthetic data.
        """,
    adjoint=True,
    plotting=True,
)
parser.add_argument("-level", help="Mesh resolution level")
parser.add_argument("-continuous_timeseries", help="Toggle discrete or continuous timeseries data")
parser.add_argument("-regularisation", help="Parameter for Tikhonov regularisation term")


# --- Set parameters

# Parsed arguments
args = parser.parse_args()
level = int(args.level or 0)
plot = parser.plotting_args()
timeseries_type = 'timeseries'
if bool(args.continuous_timeseries or False):
    timeseries_type = '_'.join([timeseries_type, 'smooth'])

# Do not attempt to plot in parallel
if COMM_WORLD.size > 1 and plot.any:
    print_output(120*'*' + "\nWARNING: Plotting turned off when running in parallel.\n" + 120*'*')
    plot.pdf = plot.png = False

# Setup output directories
dirname = os.path.dirname(__file__)
if args.adjoint is None or args.adjoint not in ('discrete', 'continuous'):
    raise ValueError
di = create_directory(os.path.join(dirname, 'outputs', 'synthetic'))
plot_dir = create_directory(os.path.join(di, 'plots'))

# Collect initialisation parameters
kwargs = {
    # 'level': level,  # Better not to instatiate large objects when plotting
    'save_timeseries': True,

    # Optimisation
    'synthetic': True,
    'qoi_scaling': 1.0,
    'nx': 1,
    'ny': 1,
    'regularisation': float(args.regularisation or 0.0),

    # Misc
    'debug': bool(args.debug or False),
    'debug_mode': args.debug_mode or 'basic',
    'di': di,
}
use_regularisation = not np.isclose(kwargs['regularisation'], 0.0)
op = TohokuRadialBasisOptions(**kwargs)
gauges = list(op.gauges.keys())

# Plotting parameters
fontsize = 22
fontsize_tick = 18
fontsize_legend = 18
kwargs = {'markevery': 5}


# --- Plot optimisation progress

# Load trajectory
msg = "Cannot plot optimisation progress on mesh {:d} because {:s} data don't exist."
fname = os.path.join(di, args.adjoint)
if args.extension is not None:
    fname = '_'.join([fname, args.extension])
fname = os.path.join(fname, 'optimisation_progress_{:s}' + '_{:d}.npy'.format(level))
if not os.path.isfile(fname.format('ctrl')):
    print_output(msg.format(level, 'control'))
elif not os.path.isfile(fname.format('func')):
    print_output(msg.format(level, 'functional'))
elif not os.path.isfile(fname.format('grad')):
    print_output(msg.format(level, 'gradient'))
else:
    control_values_opt = np.load(fname.format('ctrl', level))
    func_values_opt = np.load(fname.format('func', level))
    gradient_values_opt = np.load(fname.format('grad', level))
    optimised_value = control_values_opt[-1]

    # Fit a quadratic to the first three points and find its root
    q = scipy.interpolate.lagrange(control_values_opt[:3], func_values_opt[:3])
    dq = q.deriv()
    print_output("Exact gradient at 5.0:  {:.4f}".format(dq(5.0)))
    print_output("Exact gradient at 7.5:  {:.4f}".format(dq(7.5)))
    q_min = -dq.coefficients[1]/dq.coefficients[0]
    assert len(q.coefficients) == 3
    assert dq.deriv().coefficients[0] > 0, "Minimum does not exist!"
    print_output("Minimiser of quadratic: {:.4f}".format(q_min))
    assert np.isclose(dq(q_min), 0.0), "Gradient at minimum = {:.1f}"
    if not plot.any:
        print_output("Nothing to plot.")
        sys.exit(0)

    # # Fit quadratic to regularised functional values  # TODO
    # if use_regularisation:
    #     q_reg = scipy.interpolate.lagrange(control_values_opy[::3], func_values_reg_opt[::3])
    #     dq_reg = q_reg.deriv()
    #     q_reg_min = -dq_reg.coefficients[1]/dq_reg.coefficients[0]
    #     assert dq_reg.deriv().coefficients[0] > 0
    #     print_output("Minimiser of quadratic (regularised): {:.4f}".format(q_reg_min))
    #     assert np.isclose(dq_reg(q_reg_min), 0.0)

    # Plot parameter space
    fig, axes = plt.subplots(figsize=(8, 8))
    params = {'linewidth': 1, 'markersize': 8, 'color': 'C0', 'markevery': 10}
    params['label'] = r'$\alpha=0.00$' if use_regularisation else r'Parameter space'
    x = np.linspace(0.5, 7.5, 60)
    axes.plot(x, q(x), '--x', **params)
    axes.set_xlabel(r"Basis function coefficient, $m$", fontsize=fontsize)
    axes.set_ylabel(r"Quantity of interest", fontsize=fontsize)
    plt.xticks(fontsize=fontsize_tick)
    plt.yticks(fontsize=fontsize_tick)
    axes.set_xlim([0, 8])
    # axes.set_ylim([0, 8000])
    axes.grid(True)
    fname = 'parameter_space'
    if use_regularisation:
        fname += '_reg'
    savefig('{:s}_{:d}'.format(fname, level), plot_dir, extensions=plot.extensions)

    # Plot progress of optimisation routine
    params = {'markersize': 14, 'color': 'C0', 'label': r'$m^\star = {:.4f}$'.format(q_min)}
    if use_regularisation:
        params['label'] = r'$m^\star|_{{\alpha=0.00}} = {:.4f}$'.format(q_min)
    axes.plot(q_min, q(q_min), '*', **params)
    if use_regularisation:
        params = {
            'linewidth': 1, 'markersize': 8, 'color': 'C6', 'markevery': 10,
            'label': r'$\alpha = {:.2f}$'.format(op.regularisation),
        }
        axes.plot(x, q_reg(x), '--x', **params)
        params = {
            'markersize': 14, 'color': 'C6',
            'label': r'$m^\star|_{{\alpha={:.2f}}} = {:.2f}$'.format(op.regularisation, q_reg_min),
        }
        axes.plot(q_reg_min, q_reg(q_reg_min), '*', **params)
    params = {'markersize': 8, 'color': 'C1', 'label': 'Optimisation progress'}
    axes.plot(control_values_opt, func_values_opt, 'o', **params)
    delta_m = 0.25
    params = {'linewidth': 3, 'markersize': 8, 'color': 'C2', }
    for m, f, g in zip(control_values_opt, func_values_opt, gradient_values_opt):
        x = np.array([m - delta_m, m + delta_m])
        axes.plot(x, g*(x-m) + f, '-', **params)
    params['label'] = 'Computed gradient'
    axes.plot(x, g*(x-m) + f, '-', **params)
    plt.legend(fontsize=fontsize)
    offset = 2 if level == 0 else -3
    axes.annotate(
        r'$m = {:.4f}$'.format(control_values_opt[-1]),
        xy=(q_min + offset, func_values_opt[-1]), color='C1', fontsize=fontsize
    )
    fname = 'optimisation_progress'
    if use_regularisation:
        fname += '_reg'
    savefig(fname + '_{:d}'.format(level), plot_dir, extensions=plot.extensions)


# --- Plot timeseries

# Before optimisation
msg = "Cannot plot timeseries for initial guess controls on mesh {:d} because the data don't exist."
N = int(np.ceil(np.sqrt(len(gauges))))
fig, axes = plt.subplots(nrows=N, ncols=N, figsize=(14, 12))
for i, gauge in enumerate(gauges):

    # Load data
    fname = os.path.join(di, '{:s}_data_{:d}.npy'.format(gauge, level))
    if not os.path.isfile(fname):
        print_output(msg.format(level))
        break
    op.gauges[gauge]['data'] = np.load(fname)
    fname = os.path.join(di, '{:s}_{:s}_{:d}.npy'.format(gauge, timeseries_type, level))
    if not os.path.isfile(fname):
        print_output(msg.format(level))
        break
    op.gauges[gauge]['init'] = np.load(fname)
    data = np.array(op.gauges[gauge]['data'])
    init = np.array(op.gauges[gauge]['init'])
    n = len(data)

    T = np.array(op.gauges[gauge]['times'])/60
    T = np.linspace(T[0], T[-1], n)
    ax = axes[i//N, i % N]
    ax.plot(T, data, '-', **kwargs)
    ax.plot(T, init, '-', label=gauge, **kwargs)
    ax.legend(handlelength=0, handletextpad=0, fontsize=fontsize_legend)
    if i//N == 3:
        ax.set_xlabel('Time (min)', fontsize=fontsize)
    if i % N == 0:
        ax.set_ylabel('Elevation (m)', fontsize=fontsize)
    plt.xticks(fontsize=fontsize_tick)
    plt.yticks(fontsize=fontsize_tick)
    ax.set_yticks(ax.get_yticks().tolist())  # Avoid matplotlib error
    ax.set_yticklabels(["{:.1f}".format(tick) for tick in ax.get_yticks()])
    ax.grid()
for i in range(len(gauges), N*N):
    axes[i//N, i % N].axis(False)
savefig('timeseries_{:d}'.format(level), plot_dir, extensions=plot.extensions)

# After optimisation
msg = "Cannot plot timeseries for optimised controls on mesh {:d} because the data don't exist."
fig, axes = plt.subplots(nrows=N, ncols=N, figsize=(14, 12))
plotted = False
for i, gauge in enumerate(gauges):

    # Load data
    fname = os.path.join(op.di, args.adjoint, '{:s}_{:s}_{:d}.npy'.format(gauge, timeseries_type, level))
    if not os.path.isfile(fname):
        print_output(msg.format(level))
        break
    op.gauges[gauge]['opt'] = np.load(fname)
    data = np.array(op.gauges[gauge]['data'])
    opt = np.array(op.gauges[gauge]['opt'])
    n = len(opt)
    assert len(data) == n

    T = np.array(op.gauges[gauge]['times'])/60
    T = np.linspace(T[0], T[-1], n)
    ax = axes[i//N, i % N]
    ax.plot(T, data, '-', **kwargs)
    ax.plot(T, opt, '-', label=gauge, **kwargs)
    ax.legend(handlelength=0, handletextpad=0, fontsize=fontsize_legend)
    if i//N == 3:
        ax.set_xlabel('Time (min)', fontsize=fontsize)
    if i % N == 0:
        ax.set_ylabel('Elevation (m)', fontsize=fontsize)
    plt.xticks(fontsize=fontsize_tick)
    plt.yticks(fontsize=fontsize_tick)
    ax.set_yticks(ax.get_yticks().tolist())  # Avoid matplotlib error
    ax.set_yticklabels(["{:.1f}".format(tick) for tick in ax.get_yticks()])
    t0 = op.gauges[gauge]["arrival_time"]/60
    tf = op.gauges[gauge]["departure_time"]/60
    ax.set_xlim([t0, tf])
    ax.grid()
    plotted = True
if plotted:
    for i in range(len(gauges), N*N):
        axes[i//N, i % N].axis(False)
    fname = 'timeseries_optimised_{:d}'.format(level)
    savefig(fname, plot_dir, extensions=plot.extensions)