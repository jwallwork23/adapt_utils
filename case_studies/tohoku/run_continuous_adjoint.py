from firedrake import project

import argparse

from adapt_utils.swe.tsunami.solver import AdaptiveTsunamiProblem
from adapt_utils.case_studies.tohoku.options import TohokuOptions


parser = argparse.ArgumentParser(prog="run_continuous_adjoint")

# Space-time domain
parser.add_argument("-end_time", help="""
End time of simulation in seconds (default 1440s, i.e. 24mins)""")
parser.add_argument("-level", help="(Integer) resolution for initial mesh (default 2)")
parser.add_argument("-num_meshes", help="Number of meshes to consider (for testing, default 1)")

# Solver
parser.add_argument("-family", help="Element family for mixed FE space (default 'dg-cg')")

# QoI
parser.add_argument("-start_time", help="""
Start time of period of interest in seconds (default 1200s, i.e. 20mins)""")
parser.add_argument("-locations", help="""
Locations of interest, separated by commas. Choose from {'Fukushima Daiichi', 'Onagawa',
'Fukushima Daini', 'Tokai', 'Hamaoka', 'Tohoku', 'Tokyo'}. (Default 'Fukushima Daiichi')""")
parser.add_argument("-radii", help="Radii of interest, separated by commas (default 100km)")

# Misc
parser.add_argument("-debug", help="Print all debugging statements")
parser.add_argument("-plot_kernel", help="Just plot kernel functions (do not solve equations)")
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

# Set parameters for fixed mesh adjoint run
op = TohokuOptions(
    start_time=float(args.start_time or 0.0),
    end_time=float(args.end_time or 1200.0),
    family=args.family or 'dg-cg',
    level=int(args.level or 2),
    approach='fixed_mesh',
    plot_pvd=True,
    debug=bool(args.debug or False),
    num_meshes=int(args.num_meshes or 1),
    radii=radii,
    locations=locations,
)
assert op.start_time >= 0.0
assert op.start_time <= op.end_time

# Setup problem object
swp = AdaptiveTsunamiProblem(op)

# Take a look at the smoothed kernel function(s) *or* solve adjoint equation
plot_kernels = bool(args.plot_kernel) or False
if plot_kernels:
    for i, P1 in enumerate(swp.P1):
        swp.get_qoi_kernels(i)
        k_eta_proj = project(swp.kernels[i].split()[1], P1)
        k_eta_proj.rename("Kernel function for elevation space")
        swp.indicator_file.write(k_eta_proj)
else:
    swp.solve_adjoint()
