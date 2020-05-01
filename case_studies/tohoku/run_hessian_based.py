from thetis import *

import argparse
import datetime

from adapt_utils.case_studies.tohoku.options import TohokuOptions
from adapt_utils.swe.tsunami.solver import AdaptiveTsunamiProblem


parser = argparse.ArgumentParser(prog="run_hessian_based")

# Timstepping
parser.add_argument("-end_time", help="End time of simulation (default 25 minutes)")

# Initial mesh
parser.add_argument("-level", help="(Integer) mesh resolution (default 0)")

# Mesh adaptation
parser.add_argument("-num_meshes", help="Number of meshes to consider (default 5)")
parser.add_argument("-norm_order", help="p for Lp normalisaton (default 1)")
parser.add_argument("-adapt_field", help="Field to construct metric w.r.t")
parser.add_argument("-target", help="Target space-time complexity (default 1.0e+03)")
parser.add_argument("-h_min", help="Minimum tolerated element size (default 100m)")
parser.add_argument("-h_max", help="Maximum tolerated element size (default 1000km)")

# Outer loop
parser.add_argument("-num_adapt", help="Maximum number of adaptation loop iterations (default 35)")
parser.add_argument("-element_rtol", help="Relative tolerance for element count (default 0.005)")
parser.add_argument("-qoi_rtol", help="Relative tolerance for quantity of interest (default 0.005)")

# Misc
parser.add_argument("-debug", help="Print all debugging statements")
args = parser.parse_args()

# Order for spatial Lp normalisation
p = 1
if args.norm_order is not None:
    if p == 'inf':
        p = None
    else:
        p = float(args.norm_order)

kwargs = {

    # Timestepping
    'end_time': float(args.end_time or 1500.0),

    # Initial mesh
    'level': int(args.level or 0),

    # Mesh adaptation
    'num_meshes': int(args.num_meshes or 5),
    'adapt_field': args.adapt_field or 'elevation',
    'normalisation': 'complexity',
    'norm_order': p,
    'target': float(args.target or 5.0e+03),
    'h_min': float(args.h_min or 1.0e+02),
    'h_max': float(args.h_max or 1.0e+06),
    'plot_pvd': True,

    # Outer loop
    'element_rtol': float(args.element_rtol or 0.005),
    'qoi_rtol': float(args.qoi_rtol or 0.005),
    'num_adapt': int(args.num_adapt or 35),

    # Misc
    'debug': bool(args.debug or False),
}
logstr = 80*'*' + '\n' + 33*' ' + 'PARAMETERS\n' + 80*'*' + '\n'
for key in kwargs:
    logstr += "    {:20s}: {:}\n".format(key, kwargs[key])
logstr += 80*'*' + '\n'
print_output(logstr)

# Create parameter class and problem object
op = TohokuOptions(approach='hessian')
op.update(kwargs)
swp = AdaptiveTsunamiProblem(op)
swp.run_hessian_based()

# Print summary / logging  # TODO: Save meshes to disk
logstr += 35*' ' + 'SUMMARY\n' + 80*'*' + '\n'
logstr += "Mesh iteration  1: qoi {:.4e}\n".format(swp.qois[0])
msg = "Mesh iteration {:2d}: qoi {:.4e} space-time complexity {:.4e}\n"
for n in range(1, len(swp.qois)):
    logstr += msg.format(n+1, swp.qois[n], swp.st_complexities[n])
logstr += 80*'*' + '\n' + 30*' ' + 'FINAL ELEMENT COUNTS\n' + 80*'*' + '\n'
l = op.end_time/op.num_meshes
for i, num_cells in enumerate(swp.num_cells[-1]):
    logstr += "Time window ({:7.1f},{:7.1f}]: {:7d}\n".format(i*l, (i+1)*l, num_cells)
logstr += 80*'*' + '\n'
print_output(logstr)
date = datetime.date.today()
date = '{:d}-{:d}-{:d}'.format(date.year, date.month, date.day)
j = 0
while True:
    fname = os.path.join(op.di, '{:s}-run-{:d}'.format(date, j))
    if not os.path.exists(fname):
        break
    j += 1
with open(fname, 'w') as f:
    f.write(logstr)
print_output(fname)
