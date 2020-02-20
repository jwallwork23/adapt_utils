from thetis import *

import argparse

from adapt_utils.test_cases.steady_turbine.options import *
from adapt_utils.swe.turbine.solver import *


parser = argparse.ArgumentParser()
parser.add_argument('-level', help="Number of uniform refinements to apply to the initial mesh.")
parser.add_argument('-debug', help="Toggle debugging mode.")
parser.add_argument('-offset', help="""
    Number of turbine diameters by which to offset turbines in y-direction.
    'Aligned' configuration given by offset=0, 'Offset' configuration given by offset=1.""")
args = parser.parse_args()

level = int(args.level or 4)
kwargs = {'offset': int(args.offset or 0), 'plot_pvd': True, 'debug': bool(args.debug)}

tp = SteadyTurbineProblem(Steady2TurbineOptions(**kwargs), levels=level)
for i in range(level):
    tp = tp.tp_enriched
tp.solve()
