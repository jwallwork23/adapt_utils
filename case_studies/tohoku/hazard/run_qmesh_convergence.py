from thetis import print_output, create_directory

import argparse
import datetime
import os

from adapt_utils.case_studies.tohoku.options.options import TohokuOptions
from adapt_utils.unsteady.swe.tsunami.solver import AdaptiveTsunamiProblem


# --- Parse arguments

parser = argparse.ArgumentParser(prog="run_qmesh_convergence")

# Space-time domain
parser.add_argument("-end_time", help="End time of simulation in seconds (default 1440s i.e. 24min)")
parser.add_argument("-num_meshes", help="Number of meshes to consider (for testing, default 1)")

# Solver
parser.add_argument("-family", help="Element family for mixed FE space (default 'dg-cg')")
parser.add_argument("-nonlinear", help="Toggle nonlinear equations (default False)")
parser.add_argument("-stabilisation", help="Stabilisation method to use (default None)")

# Outer loop
parser.add_argument("-levels", help="Number of mesh levels to consider (default 5)")

# QoI
parser.add_argument("-start_time", help="""
    Start time of period of interest in seconds (default zero)""")
parser.add_argument("-locations", help="""
    Locations of interest, separated by commas. Choose from {'Fukushima Daiichi', 'Onagawa',
    'Fukushima Daini', 'Tokai', 'Hamaoka', 'Tohoku', 'Tokyo'}. (Default 'Fukushima Daiichi')
    """)
parser.add_argument("-radius", help="Radius of interest (default 100km)")

# I/O and debugging
parser.add_argument("-debug", help="Print all debugging statements")
parser.add_argument("-debug_mode", help="Choose debugging mode from 'basic' and 'full'")

args = parser.parse_args()


# --- Set parameters

if args.locations is None:  # TODO: Parse as list
    locations = ['Fukushima Daiichi', ]
else:
    locations = args.locations.split(',')
radius = float(args.radius or 100.0e+03)
family = args.family or 'cg-cg'
nonlinear = bool(args.nonlinear or False)
stabilisation = args.stabilisation or 'lax_friedrichs'
if stabilisation == 'none' or family == 'cg-cg' or not nonlinear:
    stabilisation = None
kwargs = {
    'approach': 'fixed_mesh',

    # Space-time domain
    'num_meshes': int(args.num_meshes or 1),
    'end_time': float(args.end_time or 24*60.0),

    # Physics
    'bathymetry_cap': 30.0,  # FIXME

    # Solver
    'family': family,
    'stabilisation': stabilisation,
    'use_wetting_and_drying': False,

    # QoI
    'start_time': float(args.start_time or 0.0),
    'radius': radius,
    'locations': locations,

    # I/O and debugging
    'debug': bool(args.debug or False),
}
levels = int(args.levels or 4)
di = create_directory(os.path.join(os.path.dirname(__file__), 'outputs', 'qmesh'))


# --- Loop over mesh hierarchy

qois = []
num_cells = []
for level in range(levels):
    print_output("Running qmesh convergence on level {:d}".format(level))

    # Set parameters
    kwargs['level'] = level
    op = TohokuOptions(**kwargs)

    # Solve
    swp = AdaptiveTsunamiProblem(op, nonlinear=nonlinear)
    swp.solve_forward()
    qoi = swp.quantity_of_interest()
    print_output("Quantity of interest: {:.4e}".format(qoi))

    # Diagnostics
    qois.append(qoi)
    num_cells.append(swp.num_cells[0][0])


# --- Log results

with open(os.path.join(os.path.dirname(__file__), '../../../.git/logs/HEAD'), 'r') as gitlog:
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
print_output(logdir)
