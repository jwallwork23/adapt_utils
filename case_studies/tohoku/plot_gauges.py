from adapt_utils.case_studies.tohoku.options import TohokuOptions
from adapt_utils.swe.tsunami.solver import TsunamiProblem

import matplotlib.pyplot as plt

op = TohokuOptions(utm=True)
swp = TsunamiProblem(op, levels=0)
for g in op.gauges:
    op.plot_timeseries(g)
plt.show()
