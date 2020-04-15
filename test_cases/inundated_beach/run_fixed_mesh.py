from thetis import print_output

from adapt_utils.test_cases.inundated_beach.options import BalzanoOptions
from adapt_utils.swe.solver import UnsteadyShallowWaterProblem

import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-bathymetry_type")
args = parser.parse_args()

bathy_type = int(args.bathymetry_type or 1)

op = BalzanoOptions(plot_timeseries=True, bathymetry_type=bathy_type)
op.qoi_mode = 'inundation_volume'
swp = UnsteadyShallowWaterProblem(op, levels=0)
swp.solve()
print_output("QoI: {:.1f} km^3 h".format(swp.callbacks["qoi"].get_val()/1.0e+9))
op.plot()
