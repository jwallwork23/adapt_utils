from thetis import *

from adapt_utils.turbine.solver import UnsteadyTurbineProblem
from adapt_utils.test_cases.unsteady_turbine.options import Unsteady15TurbineOptions

import argparse


parser = argparse.ArgumentParser()
parser.add_argument("-approach", help="Mesh adaptation strategy")
parser.add_argument("-debug", help="Activate debugging mode for more verbose output to screen")
parser.add_argument("-just_plot")
args = parser.parse_args()

if args.debug is not None and bool(args.debug):
    set_log_level(DEBUG)

approach = 'fixed_mesh' if args.approach is None else args.approach
just_plot = False if args.just_plot is None else bool(args.just_plot)

op = Unsteady15TurbineOptions(approach=approach)
op.plot_pvd = True
op.end_time = 300.0  # TODO: temp

tp = UnsteadyTurbineProblem(op=op)
if not just_plot:
    tp.solve()
    print("Total power output of array: {:.1f}W".format(tp.quantity_of_interest()))
tp.plot_power_timeseries()
