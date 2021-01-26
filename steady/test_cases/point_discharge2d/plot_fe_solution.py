from thetis import *

import argparse
import matplotlib.pyplot as plt
import numpy as np
import os

from adapt_utils.io import initialise_field, load_mesh
from adapt_utils.plotting import *
from adapt_utils.steady.test_cases.point_discharge2d.options import PointDischarge2dOptions


# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('approach', help="Mesh adaptation approach")
parser.add_argument('family', help="Finite element family")
parser.add_argument('-stabilisation', help="Stabilisation method to use")
parser.add_argument('-anisotropic_stabilisation', help="Use anisotropic cell size measure?")
parser.add_argument('-debug', help="Toggle debugging mode.")
args = parser.parse_args()

# Set parameters
assert args.family in ('cg', 'dg')
kwargs = {
    'approach': args.approach,
    'plot_pvd': False,
    'debug': bool(args.debug or False),
}
op = PointDischarge2dOptions(**kwargs)
op.tracer_family = args.family
op.stabilisation_tracer = args.stabilisation
op.anisotropic_stabilisation = bool(args.anisotropic_stabilisation or False)
op.di = os.path.join(op.di, args.stabilisation or args.family)

# Load from HDF5
op.plot_pvd = False
Q = FunctionSpace(load_mesh("myplex", fpath=op.di), op.tracer_family.upper(), op.degree_tracer)
approx = initialise_field(Q, "Tracer", "finite_element", fpath=op.di, op=op)

# Plot
fig, axes = plt.subplots(figsize=(8, 3))
levels = np.linspace(0, 3, 50)
tc = tricontourf(approx, axes=axes, levels=levels, cmap='coolwarm')
cbar = fig.colorbar(tc, ax=axes, orientation="horizontal", pad=0.1)
cbar.set_ticks(np.linspace(0, 3, 7))
axes.set_xticks(np.linspace(0, 50, 6))
axes.xaxis.tick_top()
axes.set_yticks(np.linspace(0, 10, 3))
savefig("finite_element_solution", op.di, extensions=["jpg"])
