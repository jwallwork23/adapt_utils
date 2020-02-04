from adapt_utils.test_cases.inundated_beach.options import BalzanoOptions
from adapt_utils.swe.tsunami.solver import TsunamiProblem

import matplotlib.pyplot as plt

op = BalzanoOptions(plot_timeseries=True)
op.qoi_mode = 'inundation_volume'
tp = TsunamiProblem(op, levels=0)
tp.solve()
op.plot()
plt.show()
