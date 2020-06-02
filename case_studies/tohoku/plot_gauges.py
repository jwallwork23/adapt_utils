from adapt_utils.case_studies.tohoku.options import TohokuOptions


op = TohokuOptions()
for g in op.gps_gauges:
    op.plot_timeseries(g, sample=30)
for g in op.pressure_gauges:
    op.plot_timeseries(g, sample=60)
