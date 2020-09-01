from thetis import COMM_WORLD, create_directory, print_output, tricontourf

import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

from adapt_utils.plotting import *
from adapt_utils.norms import vecnorm
from adapt_utils.unsteady.solver import AdaptiveProblem
from adapt_utils.unsteady.swe.tsunami.conversion import lonlat_to_utm, utm_to_lonlat


# --- Parse arguments

parser = argparse.ArgumentParser()

# Inversion
parser.add_argument("basis", help="Basis type for inversion, from {'box', 'radial', 'okada'}.")
parser.add_argument("-level", help="Mesh resolution level (default 0)")
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
level = int(args.level or 0)
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
plot_any = len(extensions) > 0
real_data = bool(args.real_data or False)
timeseries_type = "timeseries"
if bool(args.continuous_timeseries or False):
    timeseries_type = "_".join([timeseries_type, "smooth"])

# Do not attempt to plot in parallel
if COMM_WORLD.size > 1:
    print_output('Will not attempt to plot in parallel.')
    sys.exit(0)

# Setup output directories
dirname = os.path.join(os.path.dirname(__file__), basis)
di = create_directory(os.path.join(dirname, 'outputs', 'realistic' if real_data else 'synthetic'))
plot_dir = create_directory(os.path.join(di, 'plots', 'discrete'))

# Collect initialisation parameters
kwargs = {
    'level': level,
    'synthetic': not real_data,
    'noisy_data':bool(args.noisy_data or False),
}
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
fontsize = 22
fontsize_tick = 20

# Load control parameters
fname = os.path.join(di, 'discrete', 'optimisation_progress_{:s}' + '_{:d}.npy'.format(level))
kwargs['control_parameters'] = np.load(fname.format('ctrl', level))[-1]
op = constructor(**kwargs)

# Plot source over whole domain
swp = AdaptiveProblem(op)
swp.set_initial_condition()
fig, axes = plt.subplots(figsize=(8, 7))
cbar = fig.colorbar(
    tricontourf(swp.fwd_solutions[0].split()[1], levels=50, cmap='coolwarm', axes=axes),
    ax=axes)
cbar.set_label(r'Elevation [$\mathrm m$]', size=fontsize)
axes.axis(False)
cbar.ax.tick_params(labelsize=fontsize_tick)
plt.tight_layout()
savefig('optimised_source_{:d}'.format(level), fpath=plot_dir, extensions=extensions)

# Zoom
lonlat_corners = [(138, 32), (148, 42), (138, 42)]
utm_corners = [lonlat_to_utm(*corner, 54) for corner in lonlat_corners]
xlim = [utm_corners[0][0], utm_corners[1][0]]
ylim = [utm_corners[0][1], utm_corners[2][1]]
x = np.linspace(*xlim, 5)
y = np.linspace(*ylim, 5)
axes.set_xticks(x)
axes.set_yticks(y)
kwargs = dict(northern=True, force_longitude=True)
xticks = [np.round(xi, 1) for xi in np.linspace(138, 148, 5)]
yticks = [np.round(yi, 1) for yi in np.linspace(32, 42, 5)]
axes.set_xticklabels(xticks, fontsize=fontsize_tick)
axes.set_yticklabels(yticks, fontsize=fontsize_tick)
axes.set_xlim(xlim)
axes.set_ylim(ylim)
axes.set_xlabel("Degrees longitude", fontsize=fontsize)
axes.set_ylabel("Degrees latitude", fontsize=fontsize)
axes.axis(True)
savefig('optimised_source_{:d}_zoom'.format(level), fpath=plot_dir, extensions=extensions)
