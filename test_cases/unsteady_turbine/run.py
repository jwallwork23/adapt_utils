from thetis import *

from adapt_utils.turbine.solver import UnsteadyTurbineProblem
from adapt_utils.test_cases.unsteady_turbine.options import Unsteady15TurbineOptions

import argparse


parser = argparse.ArgumentParser()
parser.add_argument("-approach", help="Mesh adaptation strategy")
parser.add_argument("-debug", help="Activate debugging mode for more verbose output to screen")
args = parser.parse_args()

if args.debug is not None and bool(args.debug):
    set_log_level(DEBUG)

approach = 'fixed_mesh' if args.approach is None else args.approach

op = Unsteady15TurbineOptions(approach=approach)
op.plot_pvd = True

op.end_time = op.dt_per_export*op.dt  # TODO: temporary

tp = UnsteadyTurbineProblem(op=op)
tp.solve()
print("Total power output of array: {:.1f}kW".format(tp.quantity_of_interest()))
