from firedrake import tricontourf, interpolate, max_value

import argparse
import matplotlib.pyplot as plt
import numpy as np

from adapt_utils.io import create_directory
from adapt_utils.plotting import *
from adapt_utils.steady.solver import AdaptiveSteadyProblem
from adapt_utils.steady.test_cases.point_discharge2d.options import PointDischarge2dOptions


# --- Parse arguments

parser = argparse.ArgumentParser()
parser.add_argument('enrichment_method', help="Choose from {'GE_hp', 'GE_h', 'GE_p', 'DQ'}.")
parser.add_argument('-offset', help="Toggle between aligned or offset region of interest.")
parser.add_argument('-debug', help="Toggle debugging mode.")
args = parser.parse_args()


# --- Set parameters

offset = bool(args.offset or False)
alignment = 'offset' if offset else 'aligned'
kwargs = {
    'approach': 'dwr',
    'aligned': not offset,
    'plot_pvd': False,
    'debug': bool(args.debug or 0),
}
level = 0


# --- Loop over enrichment methods

methods = ('GE_hp', 'GE_h', 'GE_p', 'DQ')
assert args.enrichment_method in methods
plot_dir = create_directory('plots')

op = PointDischarge2dOptions(level=level, **kwargs)
op.tracer_family = 'cg'
op.stabilisation_tracer = 'supg'
op.anisotropic_stabilisation = True
op.use_automatic_sipg_parameter = False
op.normalisation = 'complexity'
op.enrichment_method = args.enrichment_method

tp = AdaptiveSteadyProblem(op, print_progress=False)
tp.solve_forward()
tp.solve_adjoint()
tp.indicate_error('tracer')
minpower = -14
minvalue = 10**minpower
indicator = interpolate(max_value(tp.indicator[op.enrichment_method], minvalue), tp.P0[0])
indicator.dat.data[:] = np.log10(indicator.dat.data)
powers = np.linspace(minpower, 0, 50)

fig, axes = plt.subplots(figsize=(8, 3))
tc = tricontourf(indicator, axes=axes, levels=powers, cmap='coolwarm')
cbar = fig.colorbar(tc, ax=axes, orientation='horizontal', pad=0.2, fraction=0.2)
powers = np.linspace(minpower, 0, int(np.floor(-minpower/2))+1)
cbar.set_ticks(powers)
cbar.set_ticklabels([r"$10^{{{:d}}}$".format(int(i)) for i in powers])
axes.set_xticks(np.linspace(0, 50, 6))
axes.xaxis.tick_top()
axes.set_yticks(np.linspace(0, 10, 3))
savefig("indicator_{:s}_{:s}".format(op.enrichment_method, alignment), plot_dir, extensions=["jpg"])
