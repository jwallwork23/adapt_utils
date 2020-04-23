from thetis import Constant, print_output

from adapt_utils.case_studies.tohoku.options import TohokuOptions
from adapt_utils.swe.tsunami.solver import TsunamiProblem

import argparse


parser = argparse.ArgumentParser(prog="run_fixed_mesh")
parser.add_argument("-end_time", help="End time of simulation")
parser.add_argument("-level", help="(Integer) mesh resolution")
parser.add_argument("-debug", help="Print all debugging statements")
args = parser.parse_args()

# Set parameters for fixed mesh run
op = TohokuOptions(
    level=int(args.level or 0),
    approach='fixed_mesh',
    plot_pvd=True,
    debug=bool(args.debug or False),
)
op.end_time = float(args.end_time or op.end_time)

# Solve
swp = TsunamiProblem(op, levels=0)
swp.solve()
print_output("Quantity of interest: {:.4e}".format(swp.callbacks["qoi"].get_value()))
op.plot_timeseries("P02")
op.plot_timeseries("P06")
