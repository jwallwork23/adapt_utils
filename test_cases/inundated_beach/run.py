from thetis import *

from adapt_utils.test_cases.inundated_beach.options import BalzanoOptions
from adapt_utils.swe.solver import UnsteadyShallowWaterProblem

op = BalzanoOptions(plot_timeseries=True)
swp = UnsteadyShallowWaterProblem(op, levels=0)
swp.solve()
op.get_timeseries_plot()
