from thetis import *

import argparse

from adapt_utils.case_studies.tohoku.options import TohokuOptions
from adapt_utils.swe.tsunami.solver import TsunamiProblem

parser = argparse.ArgumentParser()
parser.add_argument("-initial_adapt")
args = parser.parse_args()

op = TohokuOptions(utm=True, plot_pvd=True, num_adapt=4)

swp = TsunamiProblem(op, levels=0)
if args.initial_adapt is not None:  # TODO: neaten this
    swp.approach = 'hessian'
    swp.op.approach = 'hessian'
    swp.op.adapt_field = 'bathymetry'
    swp.initialise_mesh()
swp.solve()

# TODO: Plot gauge measurements
