from thetis import print_output

import argparse
import os

from adapt_utils.case_studies.tohoku.options import TohokuOptions
from adapt_utils.swe.tsunami.solver import AdaptiveTsunamiProblem


parser = argparse.ArgumentParser(prog="run_qmesh_convergence")
parser.add_argument("-levels", help="Number of mesh levels to consider")
parser.add_argument("-end_time", help="End time of simulation")
parser.add_argument("-num_meshes", help="Number of meshes to consider")
parser.add_argument("-debug", help="Print all debugging statements")
args = parser.parse_args()

# Read parameters
kwargs = {
    'approach': 'fixed_mesh',
    'debug': bool(args.debug or False),
    'num_meshes': int(args.num_meshes or 1),
    'plot_pvd': True,
}
levels = int(args.levels or 10)

qois = []
num_cells = []
for level in range(levels):
    kwargs['level'] = level

    # Set parameters
    op = TohokuOptions(**kwargs)
    op.end_time = float(args.end_time or op.end_time)

    # Solve
    swp = AdaptiveTsunamiProblem(op)
    swp.solve_forward()
    qoi = swp.quantity_of_interest()
    print_output("Quantity of interest: {:.4e}".format(qoi))
    qois.append(qoi)
    num_cells.append(swp.num_cells[0][0])

# Print/log results
logfile = open(os.path.join(swp.di, 'qmesh_convergence_log'), 'w')
for level in range(levels):
    print_output("{:2d}: elements {:6d} qoi {:.4e}".format(level+1, num_cells[level], qois[level]))
    logfile.write("{:6d} {:.4e}".format(num_cells[level], qois[level]))
logfile.close()
