from thetis import *

from adapt_utils.case_studies.tohoku.options import TohokuOptions
from adapt_utils.swe.tsunami.solver import TsunamiProblem

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-num_initial_adapt")
parser.add_argument("-n")
parser.add_argument("-end_time")
args = parser.parse_args()

n = int(args.n or 40)
num_adapt = int(args.num_initial_adapt or 0)

op = TohokuOptions(utm=True, plot_pvd=True, n=n, offset=0)  # TODO: Use offset
op.end_time = float(args.end_time or op.end_time)


# Set wetting and drying parameter
# op.wetting_and_drying_alpha.assign(0.5)
h = CellSize(op.default_mesh)
b = op.bathymetry
P0 = FunctionSpace(op.default_mesh, "DG", 0)  # NOTE: alpha is enormous in this approach (O(km))
op.wetting_and_drying_alpha = interpolate(h*sqrt(dot(grad(b), grad(b))), P0)

swp = TsunamiProblem(op, levels=0)
# swp.nonlinear = False
ext = None
if num_adapt > 0:
    swp.initialise_mesh(adapt_field='bathymetry', num_adapt=num_adapt)  # FIXME
    ext = "bathymetry_{:d}".format(num_adapt)
swp.solve()
op.plot_timeseries("P02", extension=ext)
op.plot_timeseries("P06", extension=ext)
