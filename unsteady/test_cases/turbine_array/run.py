from adapt_utils.unsteady.swe.turbine.solver import UnsteadyTurbineProblem
from adapt_utils.unsteady.test_cases.turbine_array.options import TurbineArrayOptions

import argparse


parser = argparse.ArgumentParser()
parser.add_argument("-approach", help="Mesh adaptation strategy")
parser.add_argument("-just_plot")
args = parser.parse_args()

approach = args.approach or 'fixed_mesh'
just_plot = bool(args.just_plot or False)

op = TurbineArrayOptions(approach=approach)
op.plot_pvd = True
op.save_hdf5 = True
op.end_time = 60.0  # TODO: temp

tp = UnsteadyTurbineProblem(op=op, load_index=1)
if not just_plot:
    tp.solve()
    print("Total power output of array: {:.1f}W".format(tp.quantity_of_interest()))
tp.plot_power_timeseries()
