from thetis import print_output

import argparse

from adapt_utils.case_studies.tohoku.options import TohokuOptions
from adapt_utils.swe.tsunami.solver import AdaptiveTsunamiProblem


parser = argparse.ArgumentParser(prog="run_fixed_mesh")

# Space-time domain
parser.add_argument("-end_time", help="End time of simulation in seconds (default 1440s i.e. 24min)")
parser.add_argument("-level", help="(Integer) mesh resolution (default 0)")
parser.add_argument("-num_meshes", help="Number of meshes to consider (for testing, default 1)")

# Solver
parser.add_argument("-family", help="Element family for mixed FE space (default 'dg-cg')")
parser.add_argument("-nonlinear", help="Toggle nonlinear equations (default False)")

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
parser.add_argument("-just_plot", help="Only plot gauge timeseries and errors")
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

# Set parameters for fixed mesh run
kwargs = {

    # Space-time domain
    'num_meshes': int(args.num_meshes or 1),
    'end_time': float(args.end_time or 1440.0),

    # Physics
    'bathymetry_cap': 30.0,  # FIXME

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
level = int(args.level or 0)
nonlinear = bool(args.nonlinear or False)
ext = '{:s}linear_level{:d}'.format('non' if nonlinear else '', level)
op = TohokuOptions(approach='fixed_mesh', level=level)
op.update(kwargs)

# Solve
just_plot = bool(args.just_plot or False)
if not just_plot:
    swp = AdaptiveTsunamiProblem(op, nonlinear=nonlinear, extension=ext)
    swp.solve_forward()
    print_output("Quantity of interest: {:.4e}".format(swp.quantity_of_interest()))
op.plot_all_timeseries()
