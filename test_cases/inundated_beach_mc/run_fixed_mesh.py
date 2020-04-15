from thetis import *

from adapt_utils.test_cases.inundated_beach_mc.options import BalzanoOptions
from adapt_utils.swe.solver import UnsteadyShallowWaterProblem

op = BalzanoOptions(plot_timeseries=True)
op.qoi_mode = 'inundation_volume'
swp = UnsteadyShallowWaterProblem(op, levels=0)
swp.solve()
