from firedrake import *

import argparse
import matplotlib.pyplot as plt
import numpy as np

from adapt_utils.io import create_directory
from adapt_utils.plotting import *
from adapt_utils.steady.solver import AdaptiveSteadyProblem
from adapt_utils.steady.test_cases.point_discharge2d.options import PointDischarge2dOptions


# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('enrichment_method', help="Choose from {'GE_hp', 'GE_h', 'GE_p', 'DQ'}.")
parser.add_argument('-level', help="Mesh resolution level")
parser.add_argument('-offset', help="Toggle between aligned or offset region of interest.")
parser.add_argument('-plot_colourbar', help="Plot colourbar (separately).")
parser.add_argument('-debug', help="Toggle debugging mode.")
args = parser.parse_args()

# Set parameters
offset = bool(args.offset or False)
alignment = 'offset' if offset else 'aligned'
kwargs = {
    'approach': 'dwr',
    'level': int(args.level or 1),
    'aligned': not offset,
    'plot_pvd': False,
    'debug': bool(args.debug or 0),
}
methods = ('GE_hp', 'GE_h', 'GE_p', 'DQ')
assert args.enrichment_method in methods
plot_dir = create_directory('plots')
op = PointDischarge2dOptions(**kwargs)
op.tracer_family = 'cg'
op.stabilisation_tracer = 'supg'
op.anisotropic_stabilisation = True
op.use_automatic_sipg_parameter = False
op.enrichment_method = args.enrichment_method

# Evaluate error indicator field
tp = AdaptiveSteadyProblem(op, print_progress=False)
tp.solve_forward()
tp.solve_adjoint()
tp.indicate_error('tracer')
minpower = -15
maxpower = -3
minvalue = 1.0001*10**minpower
maxvalue = 0.9999*10**maxpower
indicator = interpolate(min_value(max_value(tp.indicator[op.enrichment_method], minvalue), maxvalue), tp.P0[0])
indicator.dat.data[:] = np.log10(indicator.dat.data)
powers = np.linspace(minpower, maxpower, 50)

# Plot indicator field
fig, axes = plt.subplots(figsize=(8, 2.5))
tc = tricontourf(indicator, axes=axes, levels=powers, cmap='coolwarm')
axes.set_xticks([])
axes.set_yticks([])
savefig("indicator_{:s}_{:s}".format(op.enrichment_method, alignment), plot_dir, extensions=["jpg"])

# Plot colourbar
if bool(args.plot_colourbar or False):
    axes.set_visible(False)
    cbar = fig.colorbar(tc, ax=axes, orientation='horizontal', pad=0.2, fraction=0.25)
    powers = np.linspace(minpower, maxpower, int(np.floor((maxpower-minpower)/2))+1)
    cbar.set_ticks(powers)
    cbar.set_ticklabels([r"$10^{{{:d}}}$".format(int(i)) for i in powers])
    savefig("indicator_cbar", plot_dir, extensions=["jpg"])
