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
    """)


# --- Set parameters

# Parsed arguments
args = parser.parse_args()
bases = args.bases.split(',')
levels = range(int(args.levels or 3))
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


# --- Plot all QoI convergence curves on the same axis

qois = {basis: [] for basis in bases}
fig, axes = plt.subplots(figsize=(8, 6))
for basis in bases:

    # Collect initialisation parameters
    if basis == 'box':
        from adapt_utils.case_studies.tohoku.options.box_options import TohokuBoxBasisOptions
        constructor = TohokuBoxBasisOptions
        label = 'Piecewise constant'
    elif basis == 'radial':
        from adapt_utils.case_studies.tohoku.options.radial_options import TohokuRadialBasisOptions
        constructor = TohokuRadialBasisOptions
        label = 'Radial'
    elif basis == 'okada':
        from adapt_utils.case_studies.tohoku.options.okada_options import TohokuOkadaBasisOptions
        constructor = TohokuOkadaBasisOptions
        label = 'Okada'
    else:
        raise ValueError("Basis type '{:s}' not recognised.".format(basis))
    op = constructor(level=0, synthetic=False, noisy_data=bool(args.noisy_data or False))
    gauges = list(op.gauges.keys())

    # Get data directory
    dirname = os.path.join(os.path.dirname(__file__), basis)
    di = 'realistic'
    if args.extension is not None:
        di = '_'.join([di, args.extension])
    op.di = os.path.join(dirname, 'outputs', di, 'discrete')
    if not os.path.exists(op.di):
        raise IOError("Filepath {:s} does not exist.".format(fpath))

    # Load data for current basis and plot
    for level in levels:
        fname = os.path.join(op.di, 'optimisation_progress_{:s}' + '_{:d}.npy'.format(level))
        func_values_opt = np.load(fname.format('func', level))
        qois[basis].append(func_values_opt[-1])
    axes.semilogx(op.num_cells[:len(qois[basis])], qois[basis], '-x', label=label)
axes.set_xticks([1e4, 1e5])
for axis in (axes.xaxis, axes.yaxis):
    axis.grid(True, which='minor', color='lightgrey')
    axis.grid(True, which='major', color='lightgrey')
axes.set_xlabel("Mesh element count")
axes.set_ylabel("Square timeseries error QoI")
axes.legend()
di = 'realistic'
if args.extension is not None:
    di = '_'.join([di, args.extension])
plot_dir = create_directory(os.path.join(os.path.dirname(__file__), 'plots', di))
savefig('converged_J', fpath=plot_dir, extensions=extensions)
