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

parser = ArgumentParser(plotting=True)
parser.add_argument("-level", help="Mesh resolution level")
parser.add_argument("-continuous_timeseries", help="Toggle discrete or continuous timeseries data")


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
di = create_directory(os.path.join(dirname, 'outputs', 'synthetic'))
if args.extension is not None:
    di = '_'.join([di, args.extension])
plot_dir = create_directory(os.path.join(di, 'plots'))
create_directory(os.path.join(plot_dir, 'discrete'))

# Collect initialisation parameters
kwargs = {
    'level': level,
    'save_timeseries': True,

    # Optimisation
    'synthetic': True,
    'qoi_scaling': 1.0,
    'nx': 1,
    'ny': 1,

    # Misc
    'debug': bool(args.debug or False),
    'debug_mode': args.debug_mode or 'basic',
    'di': di,
}
op = TohokuRadialBasisOptions(**kwargs)
gauges = list(op.gauges.keys())

# Plotting parameters
fontsize = 22
fontsize_tick = 18
plotting_kwargs = {'markevery': 5}

# Plot parameter space
n = 8
control_values = np.linspace(0.5, 7.5, n)
fname = os.path.join(di, 'parameter_space_{:d}.npy'.format(level))
func_values = np.load(fname)
print_output("Explore parameter space...")
fig, axes = plt.subplots(figsize=(8, 8))
axes.plot(control_values, func_values, '--x', linewidth=2, markersize=8, markevery=10)
axes.set_xlabel("Basis function coefficient", fontsize=fontsize)
axes.set_ylabel("Mean square error quantity of interest", fontsize=fontsize)
plt.xticks(fontsize=fontsize_tick)
plt.yticks(fontsize=fontsize_tick)
axes.grid()
savefig('parameter_space_{:d}'.format(level), plot_dir, extensions=extensions)

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
savefig('discrete', plot_dir, extensions=extensions)

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
savefig(fname, os.path.join(plot_dir, 'discrete'), extensions=extensions)
