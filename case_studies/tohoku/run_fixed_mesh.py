from thetis import *

import argparse

from adapt_utils.case_studies.tohoku.options import TohokuOptions
from adapt_utils.swe.tsunami.solver import TsunamiProblem

parser = argparse.ArgumentParser()
parser.add_argument("-num_initial_adapt")
parser.add_argument("-n")
args = parser.parse_args()

n = int(args.n) or 40
op = TohokuOptions(utm=True, plot_pvd=True, num_adapt=4, n=n)

swp = TsunamiProblem(op, levels=0)
if args.num_initial_adapt is not None:
    swp.initialise_mesh(adapt_field='bathymetry', num_adapt=int(args.num_initial_adapt))  # FIXME
swp.solve()

# TODO: Plot gauge measurements
