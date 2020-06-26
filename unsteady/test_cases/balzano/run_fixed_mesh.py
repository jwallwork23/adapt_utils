from adapt_utils.test_cases.balzano.options import BalzanoOptions
from adapt_utils.adapt.solver import AdaptiveProblem

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-bathymetry_type")
args = parser.parse_args()

bathy_type = int(args.bathymetry_type or 1)

op = BalzanoOptions(plot_timeseries=True, bathymetry_type=bathy_type)
swp = AdaptiveProblem(op)
swp.solve()
