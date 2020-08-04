from adapt_utils.unsteady.swe.turbine.solver import AdaptiveTurbineProblem
from adapt_utils.unsteady.test_cases.turbine_array.options import TurbineArrayOptions

import argparse


parser = argparse.ArgumentParser()
parser.add_argument("-approach", help="Mesh adaptation strategy")
args = parser.parse_args()

approach = args.approach or 'fixed_mesh'

kwargs = {
    'plot_pvd': True,
    # 'save_hdf5': True,
    'save_hdf5': False,
}

op = TurbineArrayOptions(approach=approach)

tp = AdaptiveTurbineProblem(op)
tp.solve()
print("Total power output of array: {:.1f}W".format(tp.quantity_of_interest()))
# tp.plot_power_timeseries()
