from adapt_utils.case_studies.tohoku.options import TohokuOptions


op = TohokuOptions()
for g in op.gauges:
    op.end_time = 1500.0 if "P0" in g else 2100.0
    op.plot_timeseries(g, cutoff=35)
