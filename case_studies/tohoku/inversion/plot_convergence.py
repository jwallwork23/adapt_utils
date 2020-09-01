from thetis import COMM_WORLD, create_directory, print_output

import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

from adapt_utils.plotting import *
from adapt_utils.norms import vecnorm


# --- Parse arguments

parser = argparse.ArgumentParser()

# Inversion
parser.add_argument("basis", help="Basis type for inversion, from {'box', 'radial', 'okada'}.")
parser.add_argument("-levels", help="Number of mesh resolution levels considered (default 3)")
parser.add_argument("-real_data", help="Toggle whether to use real data (default False)")
parser.add_argument("-noisy_data", help="Toggle whether to sample noisy data (default False)")
parser.add_argument("-continuous_timeseries", help="Toggle discrete or continuous timeseries")

# I/O
parser.add_argument("-plot_pdf", help="Toggle plotting to .pdf")
parser.add_argument("-plot_png", help="Toggle plotting to .png")
parser.add_argument("-plot_pvd", help="Toggle plotting to .pvd")
parser.add_argument("-plot_all", help="Toggle plotting to .pdf, .png and .pvd")
parser.add_argument("-plot_only", help="Just plot using saved data")


# --- Set parameters

# Parsed arguments
args = parser.parse_args()
basis = args.basis
levels = range(int(args.levels or 3))
plot_pdf = bool(args.plot_pdf or False)
plot_png = bool(args.plot_png or False)
plot_all = bool(args.plot_all or False)
if plot_all:
    plot_pvd = plot_pdf = plot_png = True
extensions = []
if plot_pdf:
    extensions.append('pdf')
if plot_png:
    extensions.append('png')
if len(extensions) == 0:
    print_output("Nothing to plot. Please specify -plot_pdf or -plot_png.")
    sys.exit(0)
real_data = bool(args.real_data or False)
timeseries_type = "timeseries"
if bool(args.continuous_timeseries or False):
    timeseries_type = "_".join([timeseries_type, "smooth"])

# Do not attempt to plot in parallel
if COMM_WORLD.size > 1:
    print_output('Will not attempt to plot in parallel.')
    sys.exit(0)

# Collect initialisation parameters
kwargs = {'level': 0, 'synthetic': not real_data, 'noisy_data': bool(args.noisy_data or False)}
if basis == 'box':
    from adapt_utils.case_studies.tohoku.options.box_options import TohokuBoxBasisOptions
    op = TohokuBoxBasisOptions(**kwargs)
elif basis == 'radial':
    from adapt_utils.case_studies.tohoku.options.radial_options import TohokuRadialBasisOptions
    op = TohokuRadialBasisOptions(**kwargs)
elif basis == 'okada':
    from adapt_utils.case_studies.tohoku.options.okada_options import TohokuOkadaBasisOptions
    op = TohokuOkadaBasisOptions(**kwargs)
else:
    raise ValueError("Basis type '{:s}' not recognised.".format(basis))
gauges = list(op.gauges.keys())

# Plotting parameters
fontsize = 22
fontsize_tick = 18
fontsize_legend = 18
kwargs = {'markevery': 5}

# Setup output directories
dirname = os.path.join(os.path.dirname(__file__), basis)
di = create_directory(os.path.join(dirname, 'outputs', 'realistic' if real_data else 'synthetic'))
op.di = create_directory(os.path.join(di, 'discrete'))
plot_dir = create_directory(os.path.join(di, 'plots'))
create_directory(os.path.join(plot_dir, 'discrete'))


# --- Plot timeseries under initial guess

N = int(np.ceil(np.sqrt(len(gauges))))
for level in levels:
    fig, axes = plt.subplots(nrows=N, ncols=N, figsize=(14, 12))
    for i, gauge in enumerate(gauges):
        fname = os.path.join(di, '_'.join([gauge, 'data', str(level) + '.npy']))
        op.gauges[gauge]['data'] = np.load(fname)
        fname = os.path.join(di, '_'.join([gauge, timeseries_type, str(level) + '.npy']))
        op.gauges[gauge]['init'] = np.load(fname)

        T = np.array(op.gauges[gauge]['times'])/60
        T = np.linspace(T[0], T[-1], len(op.gauges[gauge]['data']))
        ax = axes[i//N, i % N]
        ax.plot(T, op.gauges[gauge]['data'], '--x', label=gauge + ' data', **kwargs)
        ax.plot(T, op.gauges[gauge]['init'], '--x', label=gauge + ' initial guess', **kwargs)
        ax.legend(loc='best')
        ax.set_xlabel('Time (min)', fontsize=fontsize)
        ax.set_ylabel('Elevation (m)', fontsize=fontsize)
        plt.xticks(fontsize=fontsize_tick)
        plt.yticks(fontsize=fontsize_tick)
        ax.grid()
    for i in range(len(gauges), N*N):
        axes[i//N, i % N].axis(False)
    plt.tight_layout()
    savefig('timeseries_{:d}'.format(level), fpath=plot_dir, extensions=extensions)


# --- Optimisation progress

# control_values_opt = np.load(fname.format('ctrl', level))
# optimised_value = control_values_opt[-1]

# Plot progress of QoI
fig, axes = plt.subplots(figsize=(6, 4))
for level in levels:
    fname = os.path.join(di, 'discrete', 'optimisation_progress_{:s}' + '_{:d}.npy'.format(level))
    func_values_opt = np.load(fname.format('func', level))
    iterations = range(1, len(func_values_opt)+1)
    label = '{:d} elements'.format(op.num_cells[level])
    axes.semilogx(iterations, func_values_opt, label=label)
axes.set_xlabel("Iteration")
axes.set_ylabel("Square error")
plot_dir = os.path.join(plot_dir, 'discrete')
axes.legend(loc='best', fontsize=fontsize_legend)
savefig('optimisation_progress_J', fpath=plot_dir, extensions=extensions)

# Plot progress of gradient
fig, axes = plt.subplots(figsize=(6, 4))
for level in levels:
    fname = os.path.join(di, 'discrete', 'optimisation_progress_{:s}' + '_{:d}.npy'.format(level))
    gradient_values_opt = np.load(fname.format('grad', level))
    label = '{:d} elements'.format(op.num_cells[level])
    axes.semilogy([vecnorm(djdm, order=np.Inf) for djdm in gradient_values_opt], label=label)
axes.set_xlabel("Iteration")
axes.set_ylabel(r"$\ell_\infty$-norm of gradient")
axes.legend(loc='best', fontsize=fontsize_legend)
savefig('optimisation_progress_dJdm', fpath=plot_dir, extensions=extensions)


# --- Timeseries for optimised run

msg = "Cannot plot timeseries for optimised controls on mesh {:d} because the data don't exist."
for level in levels:
    fig, axes = plt.subplots(nrows=N, ncols=N, figsize=(14, 12))
    for i, gauge in enumerate(gauges):
        fname = os.path.join(op.di, '_'.join([gauge, timeseries_type, str(level) + '.npy']))
        if not os.path.isfile(fname):
            print_output(msg.format(level))
            sys.exit(0)
        op.gauges[gauge]['opt'] = np.load(fname)

        T = np.array(op.gauges[gauge]['times'])/60
        TT = np.linspace(T[0], T[-1], len(op.gauges[gauge]['data']))
        ax = axes[i//N, i % N]
        ax.plot(TT, op.gauges[gauge]['data'], '--x', label=gauge + ' data', **kwargs)
        ax.plot(TT, op.gauges[gauge]['init'], '--x', label=gauge + ' initial guess', **kwargs)
        TT = np.linspace(T[0], T[-1], len(op.gauges[gauge]['opt']))
        ax.plot(TT, op.gauges[gauge]['opt'], '--x', label=gauge + ' optimised', **kwargs)
        ax.legend(loc='best')
        ax.set_xlabel('Time (min)', fontsize=fontsize)
        ax.set_ylabel('Elevation (m)', fontsize=fontsize)
        plt.xticks(fontsize=fontsize_tick)
        plt.yticks(fontsize=fontsize_tick)
        ax.grid()
    for i in range(len(gauges), N*N):
        axes[i//N, i % N].axis(False)
    plt.tight_layout()
    savefig('timeseries_optimised_{:d}'.format(level), fpath=plot_dir, extensions=extensions)
