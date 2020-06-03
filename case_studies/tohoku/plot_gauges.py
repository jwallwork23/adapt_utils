from adapt_utils.case_studies.tohoku.options import TohokuOptions


op = TohokuOptions()
for g in ("801", "806", "P02", "P06")
    op.plot_timeseries(g, sample=30)
