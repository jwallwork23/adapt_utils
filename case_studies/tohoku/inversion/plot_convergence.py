from thetis import COMM_WORLD, create_directory, print_output

import matplotlib.pyplot as plt
import numpy as np
import os
import sys

from adapt_utils.argparse import ArgumentParser
from adapt_utils.plotting import *
from adapt_utils.norms import lp_norm, vecnorm


# --- Parse arguments

parser = ArgumentParser(
    prog="run_convergence",
    description="""
        Given tsunami source inversion run output, generate a variety of plots:
          (a) timeseries due to initial guess vs. gauge data;
          (b) progress of the QoI during the optimisation, as a function of iteration count;
          (c) convergence curve of final 'optimised' QoI values, as a function of mesh element count;
          (d) progress of the QoI gradient during the optimisation, as a function of iteration count;
          (e) timeseries due to converged control parameters vs. gauge data.
        In addition, the script computes mean square errors from the stored timeseries data.
        """,
    basis=True,
    plotting=True)
parser.add_argument("-levels", help="Number of mesh resolution levels considered (default 3)")
parser.add_argument("-noisy_data", help="""
    Toggle whether to consider timeseries data which has *not* been sampled (default False).
    """)
parser.add_argument("-continuous_timeseries", help="Toggle discrete or continuous timeseries")
parser.add_argument("-plot_initial_guess", help="Plot initial guess timeseries")


# --- Set parameters

# Parsed arguments
args = parser.parse_args()
basis = args.basis
levels = int(args.levels or 3)
plot_pdf = bool(args.plot_pdf or False)
plot_png = bool(args.plot_png or False)
plot_all = bool(args.plot_all or False)
if plot_all:
    plot_pdf = plot_png = True
extensions = []
if plot_pdf:
    extensions.append('pdf')
if plot_png:
    extensions.append('png')
if len(extensions) == 0:
    print_output("Nothing to plot.")
    sys.exit(0)
plot_init = bool(args.plot_initial_guess or False)
timeseries_type = "timeseries"
if bool(args.continuous_timeseries or False):
    timeseries_type = "_".join([timeseries_type, "smooth"])

# Do not attempt to plot in parallel
if COMM_WORLD.size > 1:
    print_output("Will not attempt to plot in parallel.")
    sys.exit(0)

# Collect initialisation parameters
if basis == 'box':
    from adapt_utils.case_studies.tohoku.options.box_options import TohokuBoxBasisOptions
    constructor = TohokuBoxBasisOptions
elif basis == 'radial':
    from adapt_utils.case_studies.tohoku.options.radial_options import TohokuRadialBasisOptions
    constructor = TohokuRadialBasisOptions
elif basis == 'okada':
    from adapt_utils.case_studies.tohoku.options.okada_options import TohokuOkadaBasisOptions
    constructor = TohokuOkadaBasisOptions
else:
    raise ValueError("Basis type '{:s}' not recognised.".format(basis))
op = constructor(level=0, synthetic=False, noisy_data=bool(args.noisy_data or False))
gauges = list(op.gauges.keys())

# Plotting parameters
fontsize = 22
fontsize_tick = 18
fontsize_legend = 18
kwargs = {'markevery': 5}

# Setup output directories
dirname = os.path.join(os.path.dirname(__file__), basis)
di = 'realistic'
if args.extension is not None:
    di = '_'.join([di, args.extension])
plot_dir = create_directory(os.path.join(os.path.dirname(__file__), 'plots', di, basis))
di = os.path.join(dirname, 'outputs', di)
op.di = os.path.join(di, 'discrete')
for fpath in (di, op.di):
    if not os.path.exists(fpath):
        raise IOError("Filepath {:s} does not exist.".format(fpath))

# Plot timeseries under initial guess
if plot_init:
    print_output("Plotting initial timeseries against gauge data...")
    N = int(np.ceil(np.sqrt(len(gauges))))
    for level in range(levels):
        fig, axes = plt.subplots(nrows=N, ncols=N, figsize=(17, 13))
        for i, gauge in enumerate(gauges):

            # Load data
            fname = os.path.join(di, '_'.join([gauge, 'data', str(level) + '.npy']))
            op.gauges[gauge]['data'] = np.load(fname)
            fname = os.path.join(di, '_'.join([gauge, timeseries_type, str(level) + '.npy']))
            op.gauges[gauge]['init'] = np.load(fname)
            data = np.array(op.gauges[gauge]['data'])
            init = np.array(op.gauges[gauge]['init'])
            n = len(data)

            # Plot timeseries
            T = np.array(op.gauges[gauge]['times'])/60
            T = np.linspace(T[0], T[-1], n)
            ax = axes[i//N, i % N]
            ax.plot(T, data, '-', **kwargs)
            ax.plot(T, init, '-', label=gauge, **kwargs)
            ax.legend(loc='best', fontsize=fontsize_legend)
            ax.legend(handlelength=0, handletextpad=0, fontsize=fontsize_legend)
            if i//N == 3:
                ax.set_xlabel('Time (min)', fontsize=fontsize)
            if i % N == 0:
                ax.set_ylabel('Elevation (m)', fontsize=fontsize)
            ax.xaxis.set_tick_params(labelsize=fontsize_tick)
            ax.yaxis.set_tick_params(labelsize=fontsize_tick)
            ax.set_yticks(ax.get_yticks().tolist())  # Avoid matplotlib error
            ax.set_yticklabels(["{:.1f}".format(tick) for tick in ax.get_yticks()])
            ax.grid()
        for i in range(len(gauges), N*N):
            axes[i//N, i % N].axis(False)
        plt.tight_layout()
        savefig('timeseries_{:d}'.format(level), fpath=plot_dir, extensions=extensions)

# Plot progress of QoI
print_output("Plotting progress of QoI...")
fig, axes = plt.subplots(figsize=(8, 6))
qois = []
for level in range(levels):
    fname = os.path.join(op.di, 'optimisation_progress_{:s}' + '_{:d}.npy'.format(level))
    func_values_opt = np.load(fname.format('func', level))
    qois.append(func_values_opt[-1])
    its = range(1, len(func_values_opt)+1)
    label = '{:d} elements'.format(op.num_cells[level])
    axes.loglog(its, func_values_opt, label=label)
axes.set_xticks([1, 10, 100])
axes.set_yticks([1e4, 2e4, 1e5])
for axis in (axes.xaxis, axes.yaxis):
    axis.grid(True, which='minor', color='lightgrey')
    axis.grid(True, which='major', color='lightgrey')
axes.set_xlabel("Iteration")
axes.set_ylabel("Square timeseries error QoI")
axes.legend(loc='best', fontsize=fontsize_legend)
savefig('optimisation_progress_J', fpath=plot_dir, extensions=extensions)

# Plot final QoI values
print_output("Plotting final QoI values...")
fig, axes = plt.subplots(figsize=(8, 6))
axes.semilogx(op.num_cells[:len(qois)], qois, '-x')
axes.set_xticks([1e4, 1e5])
for axis in (axes.xaxis, axes.yaxis):
    axis.grid(True, which='minor', color='lightgrey')
    axis.grid(True, which='major', color='lightgrey')
axes.set_xlabel("Mesh element count")
axes.set_ylabel("Square timeseries error QoI")
savefig('converged_J', fpath=plot_dir, extensions=extensions)

# Plot progress of gradient
print_output("Plotting progress of gradient norm...")
fig, axes = plt.subplots(figsize=(8, 6))
for level in range(levels):
    fname = os.path.join(op.di, 'optimisation_progress_{:s}' + '_{:d}.npy'.format(level))
    gradient_values_opt = np.load(fname.format('grad', level))
    label = '{:d} elements'.format(op.num_cells[level])
    its = range(1, len(gradient_values_opt)+1)
    axes.loglog(its, [vecnorm(djdm, order=np.Inf) for djdm in gradient_values_opt], label=label)
axes.set_xticks([1, 10, 100])
# axes.set_yticks([2e3, 7e3])
for axis in (axes.xaxis, axes.yaxis):
    axis.grid(True, which='minor', color='lightgrey')
    axis.grid(True, which='major', color='lightgrey')
axes.set_xlabel("Iteration")
axes.set_ylabel(r"$\ell_\infty$-norm of gradient")
axes.legend(loc='best', fontsize=fontsize_legend)
savefig('optimisation_progress_dJdm_linf', fpath=plot_dir, extensions=extensions)
fig, axes = plt.subplots(figsize=(8, 6))
for level in range(levels):
    fname = os.path.join(op.di, 'optimisation_progress_{:s}' + '_{:d}.npy'.format(level))
    gradient_values_opt = np.load(fname.format('grad', level))
    label = '{:d} elements'.format(op.num_cells[level])
    its = range(1, len(gradient_values_opt)+1)
    axes.loglog(its, [lp_norm(djdm, p='l2') for djdm in gradient_values_opt], label=label)
axes.set_xticks([1, 10, 100])
# axes.set_yticks([2e3, 7e3])
for axis in (axes.xaxis, axes.yaxis):
    axis.grid(True, which='minor', color='lightgrey')
    axis.grid(True, which='major', color='lightgrey')
axes.set_xlabel("Iteration")
axes.set_ylabel(r"$\ell_2$-norm of gradient")
axes.legend(loc='best', fontsize=fontsize_legend)
savefig('optimisation_progress_dJdm_l2', fpath=plot_dir, extensions=extensions)

# Plot timeseries for optimised run
print_output("Plotting timeseries for optimised run...")
msg = "Cannot plot timeseries for optimised controls on mesh {:d} because the data don't exist."
mean_square_errors = np.zeros(levels)
discrete_qois = np.zeros(levels)
for level in range(levels):
    N = int(np.ceil(np.sqrt(len(gauges))))
    fig, axes = plt.subplots(nrows=N, ncols=N, figsize=(17, 13))
    plotted = False
    for i, gauge in enumerate(gauges):

        # Load data
        fname = os.path.join(di, '_'.join([gauge, 'data', str(level) + '.npy']))
        op.gauges[gauge]['data'] = np.load(fname)
        fname = os.path.join(op.di, '_'.join([gauge, timeseries_type, str(level) + '.npy']))
        if not os.path.isfile(fname):
            print_output(msg.format(level))
            break
        op.gauges[gauge]['opt'] = np.load(fname)
        data = np.array(op.gauges[gauge]['data'])
        opt = np.array(op.gauges[gauge]['opt'])
        n = len(opt)
        assert len(data) == n
        T = op.gauges[gauge]['times']
        T = np.linspace(T[0], T[-1], n)

        # Compute mean square errors
        square_error = (opt - data)**2
        mse = square_error.sum()/n
        msg = "{:5s} level {:d} optimised mean square error: {:.4e}"
        print_output(msg.format(gauge, level, mse))
        mean_square_errors[level] += mse

        # Compute discrete QoI
        square_error *= 0.5
        dt = T[1] - T[0]
        for q, sq_err in enumerate(square_error):
            wq = 0.5 if q in (0, n-1) else 1.0  # TODO: Other integrators than trapezium
            discrete_qois[level] += wq*dt*sq_err

        # Compute total variation
        # TODO

        # Plot timeseries
        ax = axes[i//N, i % N]
        T /= 60
        ax.plot(T, data, '-', **kwargs)
        ax.plot(T, opt, '-', label=gauge, **kwargs)
        ax.legend(handlelength=0, handletextpad=0, fontsize=fontsize_legend)
        if i//N == 3:
            ax.set_xlabel('Time (min)', fontsize=fontsize)
        if i % N == 0:
            ax.set_ylabel('Elevation (m)', fontsize=fontsize)
        ax.xaxis.set_tick_params(labelsize=fontsize_tick)
        ax.yaxis.set_tick_params(labelsize=fontsize_tick)
        ax.set_yticks(ax.get_yticks().tolist())  # Avoid matplotlib error
        ax.set_yticklabels(["{:.1f}".format(tick) for tick in ax.get_yticks()])
        t0 = op.gauges[gauge]["arrival_time"]/60
        tf = op.gauges[gauge]["departure_time"]/60
        ax.set_xlim([t0, tf])
        ax.grid()
        plotted = True
    if not plotted:
        continue
    msg = "Level {:d} overall optimised mean square error: {:.4e}"
    print_output(msg.format(level, mean_square_errors[level]))
    msg = "Level {:d} discrete QoI: {:.4e}"
    print_output(msg.format(level, discrete_qois[level]))
    for i in range(len(gauges), N*N):
        axes[i//N, i % N].axis(False)
    plt.tight_layout()
    savefig('timeseries_optimised_{:d}'.format(level), fpath=plot_dir, extensions=extensions)

# Store errors
fname = os.path.join(di, 'mean_square_errors_{:s}_{:s}.npy')
np.save(fname.format(basis, ''.join([str(level) for level in range(levels)])), mean_square_errors)
fname = os.path.join(di, 'discrete_qois_{:s}_{:s}.npy')
np.save(fname.format(basis, ''.join([str(level) for level in range(levels)])), discrete_qois)
