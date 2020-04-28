from thetis import print_output

import argparse

from adapt_utils.case_studies.tohoku.options import TohokuOptions
from adapt_utils.swe.tsunami.solver import AdaptiveTsunamiProblem


parser = argparse.ArgumentParser(prog="run_fixed_mesh")
parser.add_argument("-level", help="(Integer) mesh resolution")
parser.add_argument("-end_time", help="End time of simulation")
parser.add_argument("-num_meshes", help="Number of meshes to consider")
parser.add_argument("-debug", help="Print all debugging statements")
args = parser.parse_args()

# Set parameters for fixed mesh run
op = TohokuOptions(
    level=int(args.level or 0),
    approach='fixed_mesh',
    plot_pvd=True,
    debug=bool(args.debug or False),
    num_meshes=int(args.num_meshes or 1),
)
op.end_time = float(args.end_time or op.end_time)

# Solve
swp = AdaptiveTsunamiProblem(op)
swp.solve_forward()
print_output("Quantity of interest: {:.4e}".format(swp.quantity_of_interest()))
op.plot_timeseries("P02")
op.plot_timeseries("P06")
