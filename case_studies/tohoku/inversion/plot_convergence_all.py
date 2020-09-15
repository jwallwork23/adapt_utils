from thetis import COMM_WORLD, create_directory, print_output

import matplotlib.pyplot as plt
import numpy as np
import os
import sys

from adapt_utils.argparse import ArgumentParser
from adapt_utils.plotting import *


# --- Parse arguments

parser = ArgumentParser(
    prog="plot_convergence_all",
    description="""
        Given tsunami source inversion run output using a variety of bases for the source, plot the
        final 'optimised' QoI values on the same axis.
        """,
    bases=True,
    plotting=True,
)
parser.add_argument("-levels", help="Number of mesh resolution levels considered (default 3)")
parser.add_argument("-noisy_data", help="""
    Toggle whether to consider timeseries data which has *not* been sampled (default False).
    """)  # TODO


# --- Set parameters

# Parsed arguments
args = parser.parse_args()
bases = args.bases.split(',')
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
if len(extensions) == 0 or len(bases) == 0:
    print_output("Nothing to plot.")
    sys.exit(0)

# Do not attempt to plot in parallel
if COMM_WORLD.size > 1:
    print_output("Will not attempt to plot in parallel.")
    sys.exit(0)

# Plotting parameters
fontsize = 22
fontsize_tick = 18
fontsize_legend = 18
kwargs = {'markevery': 5}

# Paths
dirname = os.path.dirname(__file__)
extension = lambda fpath: fpath if args.extension is None else '_'.join([fpath, args.extension])
plot_dir = create_directory(os.path.join(dirname, 'plots', extension('realistic')))
data_dir = os.path.join(dirname, '{:s}', 'outputs', extension('realistic'), 'discrete')


# --- Create Options parameter objects

options = {basis: {} for basis in bases}
kwargs = dict(level=0, noisy_data=bool(args.noisy_data or False))
for basis in bases:
    if basis == 'box':
        from adapt_utils.case_studies.tohoku.options.box_options import TohokuBoxBasisOptions
        options[basis]['op'] = TohokuBoxBasisOptions(**kwargs)
        options[basis]['label'] = 'Piecewise constant'
    elif basis == 'radial':
        from adapt_utils.case_studies.tohoku.options.radial_options import TohokuRadialBasisOptions
        options[basis]['op'] = TohokuRadialBasisOptions(**kwargs)
        options[basis]['label'] = 'Radial'
    elif basis == 'okada':
        from adapt_utils.case_studies.tohoku.options.okada_options import TohokuOkadaBasisOptions
        options[basis]['op'] = TohokuOkadaBasisOptions(**kwargs)
        options[basis]['label'] = 'Okada'
    else:
        raise ValueError("Basis type '{:s}' not recognised.".format(basis))


# --- Plot all QoI convergence curves on the same axis

qois = {basis: [] for basis in bases}
fig, axes = plt.subplots(figsize=(8, 6))
for basis in bases:
    op = options[basis]['op']
    label = options[basis]['label']
    gauges = list(op.gauges.keys())

    # Get data directory
    op.di = data_dir.format(basis)
    if not os.path.exists(op.di):
        raise IOError("Filepath {:s} does not exist.".format(fpath))

    # Load data for current basis and plot
    for level in range(levels):
        fname = os.path.join(op.di, 'optimisation_progress_{:s}' + '_{:d}.npy'.format(level))
        func_values_opt = np.load(fname.format('func', level))
        qois[basis].append(func_values_opt[-1])
    axes.semilogx(op.num_cells[:len(qois[basis])], qois[basis], '-x', label=label)
axes.set_xticks([1e4, 1e5])
for axis in (axes.xaxis, axes.yaxis):
    axis.grid(True, which='minor', color='lightgrey')
    axis.grid(True, which='major', color='lightgrey')
axes.set_xlabel("Mesh element count")
axes.set_ylabel("Continuous QoI")
axes.legend()
savefig('converged_J', fpath=plot_dir, extensions=extensions)


# --- Plot all mean square error convergence curves on the same axis

fig, axes = plt.subplots(figsize=(8, 6))
for basis in bases:
    op = options[basis]['op']
    label = options[basis]['label']
    fname = 'mean_square_errors_{:s}_{:s}.npy'
    fname = os.path.join(dirname, basis, 'outputs', extension('realistic'), fname)
    try:
        mses = np.load(fname.format(basis, ''.join([str(level) for level in range(levels)])))
    except IOError:
        msg = "Cannot plot mean square error data for '{:s}' basis".format(basis)
        if args.extension is not None:
            msg += " with extension '{:s}'".format(args.extension)
        msg += ". Run the `plot_convergence.py` script first."
        print_output(msg.format(basis, args.extension))
        continue
    axes.semilogx(op.num_cells[:len(mses)], mses, '-x', label=label)
axes.set_xticks([1e4, 1e5])
for axis in (axes.xaxis, axes.yaxis):
    axis.grid(True, which='minor', color='lightgrey')
    axis.grid(True, which='major', color='lightgrey')
axes.set_xlabel("Mesh element count")
axes.set_ylabel("Mean square error")
axes.legend()
savefig('converged_mse', fpath=plot_dir, extensions=extensions)


# --- Plot all discrete QoI convergence curves on the same axis

fig, axes = plt.subplots(figsize=(8, 6))
for basis in bases:
    op = options[basis]['op']
    label = options[basis]['label']
    fname = 'discrete_qois_{:s}_{:s}.npy'
    fname = os.path.join(dirname, basis, 'outputs', extension('realistic'), fname)
    try:
        mses = np.load(fname.format(basis, ''.join([str(level) for level in range(levels)])))
    except IOError:
        msg = "Cannot plot discrete QoI data for '{:s}' basis".format(basis)
        if args.extension is not None:
            msg += " with extension '{:s}'".format(args.extension)
        msg += ". Run the `plot_convergence.py` script first."
        print_output(msg.format(basis, args.extension))
        continue
    axes.semilogx(op.num_cells[:len(mses)], mses, '-x', label=label)
axes.set_xticks([1e4, 1e5])
for axis in (axes.xaxis, axes.yaxis):
    axis.grid(True, which='minor', color='lightgrey')
    axis.grid(True, which='major', color='lightgrey')
axes.set_xlabel("Mesh element count")
axes.set_ylabel("Discrete QoI")
axes.legend()
savefig('converged_discrete_qoi', fpath=plot_dir, extensions=extensions)
