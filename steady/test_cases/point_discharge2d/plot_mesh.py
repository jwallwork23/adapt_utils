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
parser.add_argument('-family', help="Finite element family")
parser.add_argument('-stabilisation', help="Stabilisation method to use")
parser.add_argument('-anisotropic_stabilisation', help="Use anisotropic cell size measure?")
parser.add_argument('-enrichment_method', help="Choose from {'GE_hp', 'GE_h', 'GE_p', 'PR', 'DQ'}.")
parser.add_argument('-debug', help="Toggle debugging mode.")
args = parser.parse_args()

# Set parameters
family = args.family or 'cg'
assert family in ('cg', 'dg')
kwargs = {
    'approach': args.approach,
    'enrichment_method': args.enrichment_method or 'DQ',
    'plot_pvd': False,
    'debug': bool(args.debug or False),
}
op = PointDischarge2dOptions(**kwargs)
op.tracer_family = family
op.stabilisation_tracer = args.stabilisation or 'supg'
op.anisotropic_stabilisation = False if args.anisotropic_stabilisation == "0" else True
op.di = os.path.join(op.di, op.stabilisation_tracer or family, op.enrichment_method)

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
