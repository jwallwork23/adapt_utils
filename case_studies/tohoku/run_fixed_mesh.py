from thetis import Constant

from adapt_utils.case_studies.tohoku.options import TohokuOptions
from adapt_utils.swe.tsunami.solver import TsunamiProblem

import argparse

parser = argparse.ArgumentParser(prog="run_fixed_mesh")
parser.add_argument("-end_time", help="End time of simulation")
parser.add_argument("-approach", help="Mesh adaptation approach")
parser.add_argument("-target", help="Target mesh complexity/error for metric based methods")
parser.add_argument("-debug", help="Print all debugging statements")
args = parser.parse_args()

# Read parameters
approach = args.approach or 'fixed_mesh'
debug = bool(args.debug or False)

# Set parameters for fixed mesh run
op = TohokuOptions(approach=approach, plot_pvd=True, debug=debug)
op.end_time = float(args.end_time or op.end_time)
op.sipg_parameter = Constant(10.0)

# Solve
swp = TsunamiProblem(op, levels=0)
swp.solve()
op.plot_timeseries("P02")
op.plot_timeseries("P06")
