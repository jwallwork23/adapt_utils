from thetis import print_output, create_directory

import datetime
import argparse
import os

from adapt_utils.case_studies.tohoku.options import TohokuOptions
from adapt_utils.swe.tsunami.solver import AdaptiveTsunamiProblem


parser = argparse.ArgumentParser(prog="run_qmesh_convergence")

# Space-time domain
parser.add_argument("-end_time", help="End time of simulation in seconds (default 1440s i.e. 24min)")
parser.add_argument("-num_meshes", help="Number of meshes to consider (for testing, default 1)")

# Solver
parser.add_argument("-family", help="Element family for mixed FE space (default 'dg-cg')")
parser.add_argument("-nonlinear", help="Toggle nonlinear equations (default False)")

# Outer loop
parser.add_argument("-levels", help="Number of mesh levels to consider (default 10)")

# QoI
parser.add_argument("-start_time", help="""
Start time of period of interest in seconds (default 1200s i.e. 20min)""")
parser.add_argument("-locations", help="""
Locations of interest, separated by commas. Choose from {'Fukushima Daiichi', 'Onagawa',
'Fukushima Daini', 'Tokai', 'Hamaoka', 'Tohoku', 'Tokyo'}. (Default 'Fukushima Daiichi')
""")
parser.add_argument("-radii", help="Radii of interest, separated by commas (default 100km)")

# Misc
parser.add_argument("-debug", help="Print all debugging statements")
args = parser.parse_args()

# Collect locations and radii
if args.locations is None:
    locations = ['Fukushima Daiichi', ]
else:
    locations = args.locations.split(',')
if args.radii is None:
    radii = [100.0e+03 for l in locations]
else:
    radii = [float(r) for r in args.radii.split(',')]
if len(locations) != len(radii):
    msg = "Number of locations ({:d}) and radii ({:d}) do not match."
    raise ValueError(msg.format(len(locations), len(radii)))

# Read parameters
kwargs = {

    # Space-time domain
    'num_meshes': int(args.num_meshes or 1),
    'end_time': float(args.end_time or 1440.0),

    # Solver
    'family': args.family or 'dg-cg',

    # QoI
    'start_time': float(args.start_time or 1200.0),
    'radii': radii,
    'locations': locations,

    # Misc
    'plot_pvd': True,
    'debug': bool(args.debug or False),
}
levels = int(args.levels or 10)
nonlinear = bool(args.nonlinear or False)
di = create_directory(os.path.join(os.path.dirname(__file__), 'outputs/qmesh'))

qois = []
num_cells = []
for level in range(levels):

    # Set parameters
    op = TohokuOptions(approach='fixed_mesh', level=level)
    op.update(kwargs)

    # Solve
    swp = AdaptiveTsunamiProblem(op, nonlinear=nonlinear)
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
kwargs['nonlinear'] = nonlinear
logstr = 80*'*' + '\n' + 33*' ' + 'PARAMETERS\n' + 80*'*' + '\n'
for key in kwargs:
    logstr += "    {:32s}: {:}\n".format(key, kwargs[key])
logstr += 80*'*' + '\n' + 35*' ' + 'SUMMARY\n' + 80*'*' + '\n'
logstr += "{:8s}    {:7s}\n".format('Elements', 'QoI')
for level in range(levels):
    logstr += "{:8d}    {:7.4e}\n".format(num_cells[level], qois[level])
date = datetime.date.today()
date = '{:d}-{:d}-{:d}'.format(date.year, date.month, date.day)
j = 0
while True:
    logdir = os.path.join(di, '{:s}-run-{:d}'.format(date, j))
    if not os.path.exists(logdir):
        create_directory(logdir)
        break
    j += 1
with open(os.path.join(logdir, 'log'), 'w') as logfile:
    logfile.write(logstr)
print_output(logstr)
