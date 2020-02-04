from adapt_utils.case_studies.tohoku.options import TohokuOptions
from adapt_utils.swe.tsunami.solver import TsunamiProblem

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-num_initial_adapt")
parser.add_argument("-n")
args = parser.parse_args()

n = int(args.n or 40)
num_adapt = int(args.num_initial_adapt or 0)

op = TohokuOptions(utm=True, plot_pvd=True, n=n, offset=0)  # TODO: Use offset

# NOTE: wd alpha current = 1.5

swp = TsunamiProblem(op, levels=0)
ext = None
if num_adapt > 0:
    swp.initialise_mesh(adapt_field='bathymetry', num_adapt=num_adapt)  # FIXME
    ext = "bathymetry_{:d}".format(num_adapt)
swp.solve()
op.plot_timeseries("P02", extension=ext)
op.plot_timeseries("P06", extension=ext)
