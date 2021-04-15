from thetis import *

import argparse
import matplotlib.pyplot as plt
import matplotlib.patches as ptch
import os

from adapt_utils.io import load_mesh
from adapt_utils.plotting import *  # NOQA
from adapt_utils.unsteady.test_cases.turbine_array.options import TurbineArrayOptions


# --- Parse arguments

parser = argparse.ArgumentParser()
parser.add_argument("-load_mesh", help="Mesh number to load from a previous run")
parser.add_argument("-fpath", help="Filepath for loading meshes")
parser.add_argument("-plot_pdf", help="Toggle plotting to .pdf")
parser.add_argument("-plot_png", help="Toggle plotting to .png")
parser.add_argument("-plot_all", help="Toggle plotting to .pdf, .png and .pvd")
args = parser.parse_args()


# --- Set parameters

plot_pdf = bool(args.plot_pdf or False)
plot_png = bool(args.plot_png or False)
plot_all = bool(args.plot_all or False)
if plot_all:
    plot_pvd = plot_pdf = plot_png = True
plot_any = plot_pdf or plot_png
extensions = []
if plot_pdf:
    extensions.append('pdf')
if plot_png:
    extensions.append('png')
kwargs = {
    "interior_kw": {
        "linewidth": 0.1,
    },
    "boundary_kw": {
        "color": "k",
    },
}
patch_kwargs = {
    "facecolor": "none",
    "linewidth": 2,
}
op = TurbineArrayOptions(3.0)
L = op.domain_length
W = op.domain_width
l = 15


# --- Get mesh

fname = None if args.load_mesh is None else "plex_{:s}".format(args.load_mesh)
fpath = args.fpath
if fname is not None:
    if fpath is None:
        raise ValueError("Please provide a directory to load the mesh from.")
    mesh = load_mesh(fname, fpath)
    fname = "mesh_{:s}".format(args.load_mesh)
    zoom_lim = ([-775, 775], [-260, 260])
    zoom_ticks = (np.linspace(-750, 750, 7), np.linspace(-225, 225, 7))
else:
    mesh = op.default_mesh
    fname = "mesh"
    zoom_lim = ([-625, 625], [-210, 210])
    zoom_ticks = (np.linspace(-600, 600, 7), np.linspace(-200, 200, 9))


# --- Get mesh stats

stats = """
Mesh stats
==========
Element count:  {:9d}
Vertex count:   {:9d}
Min. cell size: {:9.4f} m
Max. cell size: {:9.4f} m
Min. angle:     {:9.4f} degrees
"""
cell_sizes = project(CellSize(mesh), FunctionSpace(mesh, "DG", 0))
with cell_sizes.dat.vec_ro as cs:
    with get_minimum_angles_2d(mesh).dat.vec_ro as ma:
        stats = stats.format(
            mesh.num_cells(), mesh.num_vertices(), cs.min()[1], cs.max()[1], ma.min()[1],
        )
print_output(stats)
di = fpath or create_directory(os.path.join(os.path.dirname(__file__), 'plots'))
if args.load_mesh is not None:
    with open(os.path.join(di, 'log_{:s}'.format(args.load_mesh)), 'w+') as log:
        log.write(stats)


# --- Plot whole mesh

fig, axes = plt.subplots(figsize=(12, 6))
triplot(mesh, axes=axes, **kwargs)
axes.legend().remove()
axes.set_xlim([-L/2-l, L/2+l])
axes.set_ylim([-W/2-l, W/2+l])
axes.set_xlabel(r"$x$-coordinate $[\mathrm m]$")
axes.set_ylabel(r"$y$-coordinate $[\mathrm m]$")
axes.set_yticks(np.linspace(-W/2, W/2, 5))
plt.tight_layout()
for i, loc in enumerate(op.region_of_interest):
    patch_kwargs["edgecolor"] = "C{:d}".format(i // 3)
    centre = (loc[0]-loc[2]/2, loc[1]-loc[3]/2)
    axes.add_patch(ptch.Rectangle(centre, loc[2], loc[3], **patch_kwargs))
savefig(fname, di, extensions=extensions)


# --- Zoom in on array region

axes.set_xlim(zoom_lim[0])
axes.set_ylim(zoom_lim[1])
axes.set_xticks(zoom_ticks[0])
axes.set_yticks(zoom_ticks[1])
savefig(fname + "_zoom", di, extensions=extensions)
