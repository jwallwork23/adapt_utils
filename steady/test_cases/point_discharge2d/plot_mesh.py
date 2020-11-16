from thetis import *

import argparse
import matplotlib.pyplot as plt
import numpy as np
import os

from adapt_utils.io import load_mesh
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
mesh = load_mesh("mesh", fpath=op.di)

# Plot
fig, axes = plt.subplots(figsize=(8, 2.5))
levels = np.linspace(0, 3, 50)
triplot(mesh, axes=axes, interior_kw={"linewidth": 0.1}, boundary_kw={"color": "k"})
axes.axis(False)
axes.set_xlim([0, 50])
axes.set_ylim([0, 10])
savefig("mesh", op.di, extensions=["png"])
