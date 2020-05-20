from thetis import print_output, create_directory

import argparse
import os

from adapt_utils.case_studies.tohoku.options import TohokuOptions
from adapt_utils.swe.tsunami.solver import AdaptiveTsunamiProblem


parser = argparse.ArgumentParser(prog="run_qmesh_convergence")

# Space-time domain
parser.add_argument("-end_time", help="End time of simulation")
parser.add_argument("-num_meshes", help="Number of meshes to consider")

# Solver
parser.add_argument("-family", help="Element family for mixed FE space")

# Outer loop
parser.add_argument("-levels", help="Number of mesh levels to consider")

# Misc
parser.add_argument("-debug", help="Print all debugging statements")
args = parser.parse_args()

# Read parameters
kwargs = {
    'num_meshes': int(args.num_meshes or 1),
    'end_time': float(args.end_time or 1440.0),
    'plot_pvd': True,
    'debug': bool(args.debug or False),
}
levels = int(args.levels or 10)
di = create_directory(os.path.join(os.path.dirname(__file__), 'outputs/qmesh'))

qois = []
num_cells = []
for level in range(levels):

    # Set parameters
    kwargs['level'] = level
    op = TohokuOptions(approach='fixed_mesh')
    op.update(kwargs)

    # Solve
    swp = AdaptiveTsunamiProblem(op)
    swp.solve_forward()
    qoi = swp.quantity_of_interest()
    print_output("Quantity of interest: {:.4e}".format(qoi))
    qois.append(qoi)
    num_cells.append(swp.num_cells[0][0])

# Print/log results
with open(os.path.join(os.path.dirname(__file__), '../../.git/logs/HEAD'), 'r') as gitlog:
    for line in gitlog:
        words = line.split()
    kwargs['adapt_utils git commit'] = words[1]
logstr = 80*'*' + '\n' + 33*' ' + 'PARAMETERS\n' + 80*'*' + '\n'
for key in kwargs:
    logstr += "    {:32s}: {:}\n".format(key, kwargs[key])
logstr += 80*'*' + '\n' + 35*' ' + 'SUMMARY\n' + 80*'*' + '\n'
logstr += "{:8s}    {:7s}\n".format('Elements', 'QoI')
for level in range(levels):
    logstr += "{:8d}    {:7.4e}\n".format(num_cells[level], qois[level])
with open(os.path.join(di, 'qmesh_convergence_log'), 'w') as logfile:
    logfile.write(logstr)
print_output(logstr)
