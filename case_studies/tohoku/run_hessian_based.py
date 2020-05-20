from thetis import *

import argparse
import datetime

from adapt_utils.case_studies.tohoku.options import TohokuOptions
from adapt_utils.swe.tsunami.solver import AdaptiveTsunamiProblem


parser = argparse.ArgumentParser(prog="run_hessian_based")

# Space-time domain
parser.add_argument("-end_time", help="End time of simulation (default 24 minutes)")
parser.add_argument("-level", help="(Integer) mesh resolution (default 0)")
parser.add_argument("-num_meshes", help="Number of meshes to consider (default 12)")

# Mesh adaptation
parser.add_argument("-norm_order", help="p for Lp normalisation (default 1)")
parser.add_argument("-normalisation", help="Normalisation method (default 'complexity')")
parser.add_argument("-adapt_field", help="Field to construct metric w.r.t")
parser.add_argument("-time_combine", help="Method for time-combining Hessians (default integrate)")
parser.add_argument("-hessian_lag", help="Compute Hessian every n timesteps (default 6)")
parser.add_argument("-target", help="Target space-time complexity (default 1.0e+03)")
parser.add_argument("-h_min", help="Minimum tolerated element size (default 100m)")
parser.add_argument("-h_max", help="Maximum tolerated element size (default 1000km)")

# Outer loop
parser.add_argument("-num_adapt", help="Maximum number of adaptation loop iterations (default 35)")
parser.add_argument("-element_rtol", help="Relative tolerance for element count (default 0.005)")
parser.add_argument("-qoi_rtol", help="Relative tolerance for quantity of interest (default 0.005)")

# Misc
parser.add_argument("-save_plex", help="Save final set of mesh DMPlexes to disk")
parser.add_argument("-debug", help="Print all debugging statements")
args = parser.parse_args()
p = args.norm_order

kwargs = {

    # Timestepping
    'end_time': float(args.end_time or 1440.0),

    # Space-time domain
    'level': int(args.level or 0),
    'num_meshes': int(args.num_meshes or 12),

    # Mesh adaptation
    'adapt_field': args.adapt_field or 'elevation',
    'hessian_time_combination': args.time_combine or 'integrate',
    'hessian_timestep_lag': float(args.hessian_lag or 6),
    'normalisation': args.normalisation or 'complexity',
    'norm_order': 1 if p is None else None if p == 'inf' else float(p),
    'target': float(args.target or 5.0e+03),
    'h_min': float(args.h_min or 1.0e+02),
    'h_max': float(args.h_max or 1.0e+06),

    # Outer loop
    'element_rtol': float(args.element_rtol or 0.005),
    'qoi_rtol': float(args.qoi_rtol or 0.005),
    'num_adapt': int(args.num_adapt or 35),

    # Misc
    'debug': bool(args.debug or False),
    'plot_pvd': True,
}
save_plex = bool(args.save_plex or False)
logstr = 80*'*' + '\n' + 33*' ' + 'PARAMETERS\n' + 80*'*' + '\n'
for key in kwargs:
    logstr += "    {:24s}: {:}\n".format(key, kwargs[key])
print_output(logstr + 80*'*' + '\n')

# Create parameter class and problem object
op = TohokuOptions(approach='hessian')
op.update(kwargs)
swp = AdaptiveTsunamiProblem(op)  # TODO: Option to load plexes
swp.run_hessian_based()

# Print summary / logging
with open(os.path.join(os.path.dirname(__file__), '../../.git/logs/HEAD'), 'r') as gitlog:
    for line in gitlog:
        words = line.split()
    kwargs['adapt_utils git commit'] = words[1]
logstr += 80*'*' + '\n' + 35*' ' + 'SUMMARY\n' + 80*'*' + '\n'
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
    di = os.path.join(op.di, '{:s}-run-{:d}'.format(date, j))
    if not os.path.exists(di):
        create_directory(di)
        break
    j += 1
with open(os.path.join(di, 'log'), 'w') as logfile:
    logfile.write(logstr)
if save_plex:
    swp.store_plexes(di=di)
print_output(di)
