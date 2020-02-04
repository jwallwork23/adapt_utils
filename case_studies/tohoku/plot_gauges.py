from adapt_utils.case_studies.tohoku.options import TohokuOptions
from adapt_utils.swe.tsunami.solver import TsunamiProblem

import matplotlib.pyplot as plt

op = TohokuOptions(utm=True)
swp = TsunamiProblem(op, levels=0)
op.plot_timeseries("P02")
op.plot_timeseries("P06")
plt.show()
