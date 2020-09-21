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
    prog="run_convergence",
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
if not plot_any:
    print_output("Nothing to plot.")
    sys.exit(0)
timeseries_type = 'timeseries'
if bool(args.continuous_timeseries or False):
    timeseries_type = '_'.join([timeseries_type, 'smooth'])

# Do not attempt to plot in parallel
if COMM_WORLD.size > 1 and plot_any:
    print_output(120*'*' + "\nWARNING: Plotting turned off when running in parallel.\n" + 120*'*')
    plot_pdf = plot_png = False

# Setup output directories
dirname = os.path.dirname(__file__)
if args.adjoint is None or args.adjoint not in ('discrete', 'continuous'):
    raise ValueError
di = create_directory(os.path.join(dirname, 'outputs', 'synthetic'))
if args.extension is not None:
    di = '_'.join([di, args.extension])
plot_dir = create_directory(os.path.join(di, 'plots'))

# Collect initialisation parameters
kwargs = {
    'level': level,
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
plotting_kwargs = {'markevery': 5}

# --- Plot parameter spaces

n = 8
control_values = np.linspace(0.5, 7.5, n)

# Unregularised parameter space
func_values = np.load(os.path.join(di, 'parameter_space_{:d}.npy'.format(level)))
fig, axes = plt.subplots(figsize=(8, 8))
axes.plot(control_values, func_values, '--x', linewidth=2, markersize=8, markevery=10)
axes.set_xlabel("Basis function coefficient", fontsize=fontsize)
axes.set_ylabel("Quantity of Interest", fontsize=fontsize)
plt.xticks(fontsize=fontsize_tick)
plt.yticks(fontsize=fontsize_tick)
axes.grid()
savefig('parameter_space_{:d}'.format(level), plot_dir, extensions=extensions)

# Regularised parameter space
if use_regularisation:
    func_values_reg = np.load(os.path.join(di, 'parameter_space_reg_{:d}.npy'.format(level)))
    fig, axes = plt.subplots(figsize=(8, 8))
    axes.plot(control_values, func_values_reg, '--x', linewidth=2, markersize=8, markevery=10)
    axes.set_xlabel("Basis function coefficient", fontsize=fontsize)
    axes.set_ylabel("Regularised Quantity of Interest", fontsize=fontsize)
    plt.xticks(fontsize=fontsize_tick)
    plt.yticks(fontsize=fontsize_tick)
    axes.grid()
    savefig('parameter_space_reg_{:d}'.format(level), plot_dir, extensions=extensions)

# Plot timeseries
for gauge in gauges:
    fname = os.path.join(di, '{:s}_{:s}_{:d}.npy'.format(gauge, timeseries_type, level))
    op.gauges[gauge]['init'] = np.load(fname)
    fname = os.path.join(di, '{:s}_data_{:d}.npy'.format(gauge, level))
    op.gauges[gauge]['data'] = np.load(fname)
N = int(np.ceil(np.sqrt(len(gauges))))
fig, axes = plt.subplots(nrows=N, ncols=N, figsize=(14, 12))
for i, gauge in enumerate(gauges):
    T = np.array(op.gauges[gauge]['times'])/60
    ax = axes[i//N, i % N]
    ax.plot(T, op.gauges[gauge]['data'], '--x', label=gauge + ' data', **plotting_kwargs)
    ax.plot(T, op.gauges[gauge]['init'], '--x', label=gauge + ' simulated', **plotting_kwargs)
    ax.legend(loc='upper left')
    ax.set_xlabel('Time (min)', fontsize=fontsize)
    ax.set_ylabel('Elevation (m)', fontsize=fontsize)
    plt.xticks(fontsize=fontsize_tick)
    plt.yticks(fontsize=fontsize_tick)
    ax.grid()
for i in range(len(gauges), N*N):
    axes[i//N, i % N].axis(False)
savefig('timeseries_{:d}'.format(level), plot_dir, extensions=extensions)
di = os.path.join(di, args.adjoint)
plot_dir = create_directory(os.path.join(plot_dir, args.adjoint))

# --- Plot optimisation progress

# Load trajectory
fname = os.path.join(di, 'optimisation_progress_{:s}' + '_{:d}.npy'.format(level))
control_values_opt = np.load(fname.format('ctrl', level))
func_values_opt = np.load(fname.format('func', level))
gradient_values_opt = np.load(fname.format('grad', level))
optimised_value = control_values_opt[-1]

# Fit a quadratic to the first three points and find its root
assert len(control_values[::3]) == 3
q = scipy.interpolate.lagrange(control_values[::3], func_values[::3])
dq = q.deriv()
q_min = -dq.coefficients[1]/dq.coefficients[0]
assert dq.deriv().coefficients[0] > 0
print_output("Minimiser of quadratic: {:.4f}".format(q_min))
assert np.isclose(dq(q_min), 0.0)

# Fit quadratic to regularised functional values
if use_regularisation:
    q_reg = scipy.interpolate.lagrange(control_values[::3], func_values_reg[::3])
    dq_reg = q_reg.deriv()
    q_reg_min = -dq_reg.coefficients[1]/dq_reg.coefficients[0]
    assert dq_reg.deriv().coefficients[0] > 0
    print_output("Minimiser of quadratic (regularised): {:.4f}".format(q_reg_min))
    assert np.isclose(dq_reg(q_reg_min), 0.0)

# Plot progress of optimisation routine
fig, axes = plt.subplots(figsize=(8, 8))
params = {'linewidth': 1, 'markersize': 8, 'color': 'C0', 'markevery': 10}
params['label'] = r'$\alpha=0.00$' if use_regularisation else r'Parameter space'
x = np.linspace(control_values[0], control_values[-1], 10*len(control_values))
axes.plot(x, q(x), '--x', **params)
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
for m, f, g in zip(control_values_opt[1:], func_values_opt[1:], gradient_values_opt):
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
fname = 'optimisation_progress'
if use_regularisation:
    fname += '_reg'
savefig(fname + '_{:d}'.format(level), plot_dir, extensions=extensions)

# Plot timeseries for both initial guess and optimised control
for gauge in gauges:
    fname = os.path.join(di, '{:s}_{:s}_{:d}.npy'.format(gauge, timeseries_type, level))
    op.gauges[gauge]['opt'] = np.load(fname)
fig, axes = plt.subplots(nrows=N, ncols=N, figsize=(14, 12))
for i, gauge in enumerate(gauges):
    T = np.array(op.gauges[gauge]['times'])/60
    ax = axes[i//N, i % N]
    ax.plot(T, op.gauges[gauge]['data'], '--x', label=gauge + ' data', **plotting_kwargs)
    ax.plot(T, op.gauges[gauge]['opt'], '--x', label=gauge + ' optimised', **plotting_kwargs)
    ax.legend(loc='upper left')
    ax.set_xlabel('Time (min)', fontsize=fontsize)
    ax.set_ylabel('Elevation (m)', fontsize=fontsize)
    plt.xticks(fontsize=fontsize_tick)
    plt.yticks(fontsize=fontsize_tick)
    ax.grid()
for i in range(len(gauges), N*N):
    axes[i//N, i % N].axis(False)
fname = 'timeseries_optimised_{:d}'.format(level)
savefig(fname, plot_dir, extensions=extensions)
