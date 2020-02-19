from adapt_utils.case_studies.tohoku.options import TohokuOptions
from adapt_utils.swe.tsunami.solver import TsunamiProblem

op = TohokuOptions(utm=True)
swp = TsunamiProblem(op, levels=0)
for g in op.gauges:
    op.end_time = 1500.0 if "P0" in g else 2100.0
    op.plot_timeseries(g, cutoff=35)
