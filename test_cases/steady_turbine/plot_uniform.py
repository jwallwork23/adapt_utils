from thetis import *

import argparse

from adapt_utils.test_cases.steady_turbine.options import *
from adapt_utils.swe.turbine.solver import *


parser = argparse.ArgumentParser()
parser.add_argument('-level', help="Number of uniform refinements to apply to the initial mesh.")
parser.add_argument('-offset', help="Choose offset or aligned configuration.")
parser.add_argument('-debug', help="Toggle debugging mode.")
args = parser.parse_args()

level = int(args.level or 4)
offset = bool(args.offset)
kwargs = {'plot_pvd': True, 'debug': bool(args.debug)}

data = {}
op = Steady2TurbineOffsetOptions(**kwargs) if offset else Steady2TurbineOptions(**kwargs)
tp = SteadyTurbineProblem(op, levels=level)
for i in range(level+1):
    if i < level:
        tp = tp.tp_enriched
tp.solve()
