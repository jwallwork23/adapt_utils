from thetis import print_output

from adapt_utils.test_cases.inundated_beach.options import BalzanoOptions
from adapt_utils.swe.tsunami.solver import TsunamiProblem

import matplotlib.pyplot as plt

op = BalzanoOptions(plot_timeseries=True)
op.qoi_mode = 'inundation_volume'
tp = TsunamiProblem(op, levels=0)
tp.solve()
print_output("QoI: {:.4e}".format(tp.callbacks["qoi"].get_val()))
op.plot()
plt.show()
