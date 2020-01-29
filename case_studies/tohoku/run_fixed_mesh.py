from thetis import *

import argparse

from adapt_utils.case_studies.tohoku.options import TohokuOptions
from adapt_utils.tsunami.solver import TsunamiProblem

parser = argparse.ArgumentParser()
parser.add_argument("-initial_adapt")
args = parser.parse_args()

op = TohokuOptions(utm=True, plot_pvd=True, num_adapt=4)

if args.initial_adapt is not None:
    op.adapt_to_bathymetry_hessian()

swp = TsunamiProblem(op, levels=0)
swp.solve()

# TODO: Plot gauge measurements
