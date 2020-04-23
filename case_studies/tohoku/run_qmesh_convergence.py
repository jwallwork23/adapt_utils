from thetis import Constant, print_output

from adapt_utils.case_studies.tohoku.options import TohokuOptions
from adapt_utils.swe.tsunami.solver import TsunamiProblem

import argparse


parser = argparse.ArgumentParser(prog="run_qmesh_convergence")
parser.add_argument("-levels", help="Number of mesh levels to consider")
parser.add_argument("-end_time", help="End time of simulation")
parser.add_argument("-debug", help="Print all debugging statements")
args = parser.parse_args()

# Read parameters
debug = bool(args.debug or False)
levels = int(args.levels or 10)

qois = []
num_cells = []
for level in range(levels):

    # Set parameters
    op = TohokuOptions(level=level, approach='fixed_mesh', plot_pvd=True, debug=debug)
    op.end_time = float(args.end_time or op.end_time)

    # Solve
    swp = TsunamiProblem(op, levels=0)
    swp.solve()
    qoi = swp.callbacks["qoi"].get_value()
    print_output("Quantity of interest: {:.4e}".format(qoi))
    qois.append(qoi)
    num_cells.append(swp.mesh.num_cells())

# Print results
for level in range(levels):
    print_output("{:2d}: elements {:6d} qoi {:.4e}".format(level+1, num_cells[level], qois[level]))
