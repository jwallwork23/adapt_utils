from thetis import *

import argparse
import matplotlib.pyplot as plt
import numpy as np
import os

from adapt_utils.io import initialise_field, load_mesh
from adapt_utils.plotting import *
from adapt_utils.steady.solver import AdaptiveSteadyProblem
from adapt_utils.steady.test_cases.point_discharge2d.options import PointDischarge2dOptions


# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('-level', help="Number of uniform refinements to apply to the initial mesh.")
parser.add_argument('-family', help="Finite element family")
parser.add_argument('-stabilisation', help="Stabilisation method to use")
parser.add_argument('-debug', help="Toggle debugging mode.")
args = parser.parse_args()

# Set parameters
family = args.family or 'cg'
assert family in ('cg', 'dg')
kwargs = {
    'level': int(args.level or 0),
    'plot_pvd': False,
    'debug': bool(args.debug or False),
    'stabilisation': args.stabilisation,
}
op = PointDischarge2dOptions(approach='fixed_mesh', **kwargs)
op.tracer_family = family
op.stabilisation = args.stabilisation
op.di = os.path.join(op.di, args.stabilisation or args.family)

# Get analytical solution
tp = AdaptiveSteadyProblem(op)
analytical = op.analytical_solution(tp.Q[0])

# Plot
fig, axes = plt.subplots(figsize=(8, 3))
levels = np.linspace(0, 3, 50)
tc = tricontourf(analytical, axes=axes, levels=levels, cmap='coolwarm')
cbar = fig.colorbar(tc, ax=axes, orientation="horizontal", pad=0.1)
cbar.set_ticks(np.linspace(0, 3, 7))
axes.set_xticks(np.linspace(0, 50, 6))
axes.xaxis.tick_top()
axes.set_yticks(np.linspace(0, 10, 3))
savefig("analytical_solution", op.di, extensions=["png"])
