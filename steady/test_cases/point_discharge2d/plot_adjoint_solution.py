from thetis import *

import argparse
import matplotlib.pyplot as plt
import numpy as np
import os

from adapt_utils.io import initialise_field
from adapt_utils.plotting import *
from adapt_utils.steady.test_cases.point_discharge2d.options import PointDischarge2dOptions


# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('-level', help="Number of uniform refinements to apply to the initial mesh.")
parser.add_argument('-family', help="Finite element family")
parser.add_argument('-stabilisation', help="Stabilisation method to use")
parser.add_argument('-anisotropic_stabilisation', help="Use anisotropic cell size measure?")
parser.add_argument('-offset', help="Toggle between aligned or offset region of interest.")
parser.add_argument('-debug', help="Toggle debugging mode.")
args = parser.parse_args()

# Set parameters
family = args.family or 'cg'
assert family in ('cg', 'dg')
offset = bool(args.offset or False)
kwargs = {
    'level': int(args.level or 0),
    'plot_pvd': False,
    'aligned': not offset,
    'debug': bool(args.debug or False),
}
op = PointDischarge2dOptions(approach='fixed_mesh', **kwargs)
op.tracer_family = family
op.stabilisation_tracer = args.stabilisation or 'supg'
op.anisotropic_stabilisation = False if args.anisotropic_stabilisation == "0" else True
alignment = 'offset' if offset else 'aligned'
op.di = os.path.join(op.di, op.stabilisation_tracer or family, alignment)

# Plot
Q = FunctionSpace(op.default_mesh, op.tracer_family.upper(), op.degree_tracer)
solutions = []
for approach in ('continuous', 'discrete'):

    # Load from HDF5
    fname = "_".join([approach, "adjoint"])
    adj = initialise_field(Q, "Adjoint tracer", fname, fpath=op.di, op=op)
    solutions.append(adj)

    # Plot
    fig, axes = plt.subplots(figsize=(8, 3))
    eps = 1.0e-05
    levels = np.linspace(-eps, 0.75 + eps, 20)
    tc = tricontourf(adj, axes=axes, levels=levels, cmap='coolwarm')
    cbar = fig.colorbar(tc, ax=axes, orientation="horizontal", pad=0.1)
    cbar.set_ticks(np.linspace(0, 0.75, 4))
    axes.set_xticks(np.linspace(0, 50, 6))
    axes.xaxis.tick_top()
    axes.set_yticks(np.linspace(0, 10, 3))
    savefig(fname, op.di, extensions=["jpg"])

# Compute L2 error
print_output("L2 'error': {:.4f}%".format(100*errornorm(*solutions)/norm(solutions[1])))

# Plot difference
fig, axes = plt.subplots(figsize=(8, 3))
eps = 1.0e-05
minpower = -14
maxpower = -2
minvalue = 10**minpower
maxvalue = 10**maxpower
diff = solutions[0].copy(deepcopy=True)
diff -= solutions[1]
diff.interpolate(min_value(max_value(abs(diff), minvalue), maxvalue))
diff.dat.data[:] = np.log10(diff.dat.data)
powers = np.linspace(minpower, maxpower, 50)
tc = tricontourf(diff, axes=axes, levels=powers, cmap='coolwarm')
cbar = fig.colorbar(tc, ax=axes, orientation="horizontal", pad=0.1)
powers = np.linspace(minpower, maxpower, int(np.floor((maxpower-minpower)/2))+1)
cbar.set_ticks(powers)
cbar.set_ticklabels([r"$10^{{{:d}}}$".format(int(i)) for i in powers])
axes.set_xticks(np.linspace(0, 50, 6))
axes.xaxis.tick_top()
axes.set_yticks(np.linspace(0, 10, 3))
savefig("difference", op.di, extensions=["jpg"])
