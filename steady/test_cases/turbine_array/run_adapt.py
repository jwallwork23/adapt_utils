from thetis import *

import argparse
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as ptch
import os
import sys

from adapt_utils.steady.test_cases.turbine_array.options import TurbineArrayOptions
from adapt_utils.swe.turbine.solver import AdaptiveSteadyTurbineProblem
from adapt_utils.plotting import *


# --- Parse arguments

parser = argparse.ArgumentParser()
parser.add_argument('approach', help="Mesh adaptation approach")

# Problem setup
parser.add_argument('-level', help="""
    Number of uniform refinements to apply to the initial mesh (default 0)""")
parser.add_argument('-offset', help="""
    Number of turbine diameters by which to offset turbines in y-direction.
    'Aligned' configuration given by offset=0, 'Offset' configuration given by offset=1.
    (Default 0)""")

# Mesh adaptation
parser.add_argument('-target', help="Target complexity for adaptive approaches (default 3200)")
parser.add_argument('-adapt_field', help="Field(s) for adaptation (default all_int)")

# I/O and debugging
parser.add_argument('-plot_pdf', help="Save plots to .pdf (default False).")
parser.add_argument('-plot_png', help="Save plots to .png (default False).")
parser.add_argument('-plot_jpg', help="Save plots to .jpg (default False).")
parser.add_argument('-plot_pvd', help="Save plots to .pvd (default False).")
parser.add_argument('-plot_all', help="Plot to .pdf, .png and .pvd (default False).")
parser.add_argument('-save_plex', help="Save DMPlex to HDF5 (default False)")
parser.add_argument('-debug', help="Toggle debugging mode (default False).")
parser.add_argument('-debug_mode', help="""
    Choose debugging mode from 'basic' and 'full' (default 'basic').""")

args = parser.parse_args()


# --- Set parameters

plot_pdf = bool(args.plot_pdf or False)
plot_png = bool(args.plot_png or False)
plot_jpg = bool(args.plot_jpg or False)
if bool(args.plot_all or False):
    plot_pdf = plot_png = plot_jpg = True
extensions = []
if plot_pdf:
    extensions.append('pdf')
if plot_png:
    extensions.append('png')
if plot_jpg:
    extensions.append('jpg')
plot_any = len(extensions) > 0
save_plex = bool(args.save_plex or False)
kwargs = {
    'approach': args.approach,

    # Problem setup
    'level': int(args.level or 0),
    'offset': int(args.offset or 0),

    # Adaptation parameters
    'target': float(args.target or 3200.0),
    'adapt_field': args.adapt_field or 'all_int',
    'normalisation': 'complexity',
    'convergence_rate': 1,
    'norm_order': None,  # i.e. infinity norm
    'h_max': 500.0,

    # Optimisation parameters
    'element_rtol': 0.001,
    'num_adapt': 35,

    # I/O and debugging
    'plot_pvd': True,
    'debug': bool(args.debug or 0),
    'debug_mode': args.debug_mode or 'basic',
}
discrete_turbines = True
# discrete_turbines = False
op = TurbineArrayOptions(**kwargs)
op.set_all_rtols(op.element_rtol)
if op.approach == 'fixed_mesh':
    raise ValueError("This script is for mesh adaptive methods.")

# Plotting
patch_kwargs = {
    'facecolor': 'none',
    'edgecolor': 'b',
    'linewidth': 2,
}
triplot_kwargs = {
    "interior_kw": {
        "linewidth": 0.1,
    },
    "boundary_kw": {
        "linewidth": 3.0,
        "colors": ["k", "k", "k"],
    },
}
tricontourf_kwargs = {
    'cmap': 'coolwarm',
    'locator': matplotlib.ticker.LogLocator(base=10),
    'levels': 50,
}
colourbar_kwargs = {
    'orientation': 'horizontal',
}
fontsizes = {
    'legend': 20,
    'tick': 24,
    'cbar': 20,
}
plot_dir = create_directory(os.path.join(os.path.dirname(__file__), 'plots'))


# --- Solve forward problem within a mesh adaptation loop

tp = AdaptiveSteadyTurbineProblem(op, discrete_adjoint=True, discrete_turbines=discrete_turbines)
tp.run()
if save_plex:
    tp.store_plexes('{:s}_{:d}.h5'.format(op.approach, op.offset))
if not plot_any:
    sys.exit(0)


# --- Plot mesh, with a zoom of the turbine array region

# Get turbine array
loc = op.region_of_interest
D = op.turbine_diameter
turbine1 = ptch.Rectangle((loc[0][0]-D/2, loc[0][1]-D/2), D, D, **patch_kwargs)
turbine2 = ptch.Rectangle((loc[1][0]-D/2, loc[1][1]-D/2), D, D, **patch_kwargs)

# Plot mesh and annotate with turbine footprint
fig, axes = plt.subplots(figsize=(12, 5))
triplot(tp.mesh, axes=axes, **triplot_kwargs)
eps = 1
axes.set_xlim([-eps, op.domain_length+eps])
axes.set_ylim([-eps, op.domain_width+eps])
axes.add_patch(turbine1)
axes.add_patch(turbine2)
fname = '{:s}__offset{:d}__target{:d}__elem{:d}'
savefig(fname.format(op.approach, op.offset, int(op.target), tp.num_cells[-1][0]), plot_dir, extensions=extensions)

# Magnify turbine region
axes.set_xlim(loc[0][0] - 2*D, loc[1][0] + 2*D)
axes.set_ylim(op.domain_width/2 - 3.5*D, op.domain_width/2 + 3.5*D)
fname = '{:s}__offset{:d}__target{:d}__elem{:d}__zoom'
savefig(fname.format(op.approach, op.offset, int(op.target), tp.num_cells[-1][0]), plot_dir, extensions=extensions)


# --- Plot goal-oriented error indicators

if op.approach not in ('dwr', 'isotropic_dwr', 'anisotropic_dwr'):
    sys.exit(0)

# Plot dwr cell residual
residual = interpolate(abs(tp.indicator['dwr_cell']), tp.indicator['dwr_cell'].function_space())
fig, axes = plt.subplots(figsize=(12, 5))
fig.colorbar(tricontourf(residual, axes=axes, **tricontourf_kwargs), ax=axes, **colourbar_kwargs)
axes.set_yticks([0, 100, 200, 300, 400, 500])
axes.set_xlim([-eps, op.domain_length+eps])
axes.set_ylim([-eps, op.domain_width+eps])
fname = 'cell_residual__offset{:d}__elem{:d}'
savefig(fname.format(op.offset, tp.num_cells[-1][0]), plot_dir, extensions=extensions)

# Plot dwr flux
flux = interpolate(abs(tp.indicator['dwr_flux']), tp.indicator['dwr_flux'].function_space())
fig, axes = plt.subplots(figsize=(12, 5))
fig.colorbar(tricontourf(flux, axes=axes, **tricontourf_kwargs), ax=axes, **colourbar_kwargs)
axes.set_yticks([0, 100, 200, 300, 400, 500])
axes.set_xlim([-eps, op.domain_length+eps])
axes.set_ylim([-eps, op.domain_width+eps])
fname = 'flux__offset{:d}__elem{:d}'
savefig(fname.format(op.offset, tp.num_cells[-1][0]), plot_dir, extensions=extensions)
