from adapt_utils.test_cases.inundated_beach.options import BalzanoOptions
from adapt_utils.swe.solver import UnsteadyShallowWaterProblem

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-bathymetry_type")
args = parser.parse_args()

bathy_type = int(args.bathymetry_type or 1)

op = BalzanoOptions(plot_timeseries=True, bathymetry_type=bathy_type)
swp = UnsteadyShallowWaterProblem(op, levels=0)
swp.solve()
