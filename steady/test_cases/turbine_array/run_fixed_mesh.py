from thetis import *

import argparse
import matplotlib.pyplot as plt
import matplotlib.patches as ptch
import os

from adapt_utils.steady.test_cases.turbine_array.options import TurbineArrayOptions
from adapt_utils.steady.swe.turbine.solver import AdaptiveSteadyTurbineProblem
from adapt_utils.plotting import *


# --- Parse arguments

parser = argparse.ArgumentParser()

# Problem setup
parser.add_argument('-level', help="""
    Number of uniform refinements to apply to the initial mesh (default 0)""")
parser.add_argument('-offset', help="""
    Number of turbine diameters by which to offset turbines in y-direction.
    'Aligned' configuration given by offset=0, 'Offset' configuration given by offset=1.
    (Default 0)""")

# I/O and debugging
parser.add_argument('-plot_pdf', help="Save plots to .pdf (default False).")
parser.add_argument('-plot_png', help="Save plots to .png (default False).")
parser.add_argument('-plot_pvd', help="Save plots to .pvd (default False).")
parser.add_argument('-plot_all', help="Plot to .pdf, .png and .pvd (default False).")
parser.add_argument('-debug', help="Toggle debugging mode (default False).")
parser.add_argument('-debug_mode', help="""
    Choose debugging mode from 'basic' and 'full' (default 'basic').""")

args = parser.parse_args()


# --- Set parameters

plot_pdf = bool(args.plot_pdf or False)
plot_png = bool(args.plot_png or False)
if bool(args.plot_all or False):
    plot_pdf = plot_png = True
extensions = []
if plot_pdf:
    extensions.append('pdf')
if plot_png:
    extensions.append('png')
plot_any = len(extensions) > 0
kwargs = {
    'approach': 'fixed_mesh',

    # Problem setup
    'level': int(args.level or 0),
    'offset': int(args.offset or 0),

    # I/O and debugging
    'plot_pvd': bool(args.plot_pvd or False),
    'debug': bool(args.debug or False),
    'debug_mode': args.debug_mode or 'basic',
}
discrete_turbines = True
# discrete_turbines = False
op = TurbineArrayOptions(**kwargs)
num_cells = op.default_mesh.num_cells()

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
        "colors": ["C0", "C2", "C1"],
    },
}
tricontourf_kwargs = {
    'vmin': 3.5,
    'vmax': 5.2,
    'cmap': 'coolwarm',
    'levels': 50,
}
fontsizes = {
    'legend': 20,
    'tick': 24,
    'cbar': 20,
}
plot_dir = create_directory(os.path.join(os.path.dirname(__file__), 'screenshots'))


# --- Plot initial mesh

if plot_any:
    fig, axes = plt.subplots(figsize=(12, 5))
    triplot(op.default_mesh, axes=axes, **triplot_kwargs)
    axes.set_xlim([0, op.domain_length])
    axes.set_ylim([0, op.domain_width])
    for axis in (axes.xaxis, axes.yaxis):
        for tick in axis.get_major_ticks():
            tick.label.set_fontsize(fontsizes['tick'])

    # Annotate with turbine array
    loc = op.region_of_interest
    D = op.turbine_diameter
    turbine1 = ptch.Rectangle((loc[0][0]-D/2, loc[0][1]-D/2), D, D, **patch_kwargs)
    turbine2 = ptch.Rectangle((loc[1][0]-D/2, loc[1][1]-D/2), D, D, **patch_kwargs)
    axes.add_patch(turbine1)
    axes.add_patch(turbine2)
    handles, labels = axes.get_legend_handles_labels()
    handles.append(turbine1)
    labels = ['Inflow', 'Outflow', 'Walls', 'Turbines']
    axes.legend(handles, labels, loc='upper right', fontsize=fontsizes['legend'])

    # Save to file
    plt.tight_layout()
    fname = os.path.join(plot_dir, 'inital_mesh__offset{:d}__elem{:d}.{:3s}')
    for ext in extensions:
        plt.savefig(fname.format(op.offset, num_cells, ext))


# --- Solve forward problem

tp = AdaptiveSteadyTurbineProblem(op, discrete_turbines=discrete_turbines)
tp.solve_forward()
op.print_debug("Power output: {:.4e}kW".format(tp.quantity_of_interest()/1000))  # TODO: MegaWatts?


# --- Plot fluid speed

if plot_any:

    # Compute fluid speed
    u, eta = tp.fwd_solution.split()
    spd = interpolate(sqrt(dot(u, u)), tp.P1[0])

    # Plot
    fig, axes = plt.subplots(figsize=(12, 5))
    cbar = fig.colorbar(tricontourf(spd, axes=axes, **tricontourf_kwargs), ax=axes)
    cbar.ax.set_yticklabels(cbar.ax.get_yticklabels(), fontsize=fontsizes['cbar'])
    cbar.set_label(r'Fluid speed [$m\,s^{-1}$]', fontsize=fontsizes['cbar'])
    axes.set_xlim([0, op.domain_length])
    axes.set_ylim([0, op.domain_width])
    for axis in (axes.xaxis, axes.yaxis):
        for tick in axis.get_major_ticks():
            tick.label.set_fontsize(fontsizes['tick'])

    # Save to file
    plt.tight_layout()
    fname = os.path.join(plot_dir, 'fluid_speed__offset{:d}__elem{:d}.{:3s}')
    for ext in extensions:
        plt.savefig(fname.format(op.offset, num_cells, ext))
