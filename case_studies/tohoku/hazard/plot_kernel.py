from thetis import *

import argparse
import matplotlib.pyplot as plt
import numpy as np

from adapt_utils.case_studies.tohoku.hazard.options import TohokuHazardOptions
from adapt_utils.plotting import *
from adapt_utils.swe.tsunami.solver import AdaptiveTsunamiProblem


# --- Parse arguments

parser = argparse.ArgumentParser(prog="plot_kernel")
parser.add_argument("-level", help="(Integer) mesh resolution (default 0)")
parser.add_argument("-locations", help="""
    Locations of interest, separated by commas. Choose from {'Fukushima Daiichi', 'Onagawa',
    'Fukushima Daini', 'Tokai', 'Hamaoka', 'Tohoku', 'Tokyo', 'Ogasawara'}. (Default 'Fukushima Daiichi')
    """)
parser.add_argument("-radius", help="Radius of interest (default 100km)")
parser.add_argument("-kernel_shape", help="""
    Choose kernel shape from {'gaussian', 'circular_bump', 'ball'}.
    """)
args = parser.parse_args()


# --- Set parameters

locations = ['Fukushima Daiichi'] if args.locations is None else args.locations.split(',')
radius = float(args.radius or 100.0e+03)
kwargs = {
    'level': int(args.level or 0),
    'radius': radius,
    'locations': locations,
    'kernel_shape': args.kernel_shape or 'gaussian',
}
op = TohokuHazardOptions(**kwargs)
plot_dir = create_directory('plots')


# --- Set kernel

swp = AdaptiveTsunamiProblem(op)
swp.get_qoi_kernels(0)
k_u, k_eta = swp.kernels[0].split()
kernel = Function(swp.P1[0], name="QoI kernel")
kernel.interpolate(k_eta)
num_cells = swp.mesh.num_cells()
num_vertices = swp.mesh.num_vertices()
vol = assemble(k_eta*dx)
print_output("elem {:d} vert {:d} vol {:.4f} km^3".format(num_cells, num_vertices, vol/1.0e+09))


# --- Plot
fig, axes = plt.subplots(figsize=(5, 5))
cax = fig.add_axes([0.05, 0.55, 0.1, 0.4])
cax.axis(False)
tc = tricontourf(k_eta, axes=axes, levels=np.linspace(0, 1.05, 50), cmap='coolwarm')
cbar = fig.colorbar(tc, ax=cax)
cbar.set_ticks(np.linspace(0, 1, 6))
axes.axis(False)
op.annotate_plot(axes, markercolour='C2', textcolour='C2')
loc = ('_'.join(locations)).replace(' ', '_').lower()
savefig("{:s}_{:s}_{:d}".format(loc, op.kernel_shape, op.level), plot_dir, extensions=["jpg"], tight=False)
