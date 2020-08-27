import argparse

from adapt_utils.case_studies.tohoku.options.hazard_options import TohokuHazardOptions
from adapt_utils.io import TimeDependentAdaptationLogger
from adapt_utils.unsteady.swe.tsunami.solver import AdaptiveTsunamiProblem


# --- Parse arguments

parser = argparse.ArgumentParser(prog="run_hessian_based")

# Space-time domain
parser.add_argument("-end_time", help="End time of simulation in seconds (default 1440s i.e. 24min)")
parser.add_argument("-level", help="(Integer) mesh resolution (default 0)")
parser.add_argument("-num_meshes", help="Number of meshes to consider (default 12)")

# Solver
parser.add_argument("-family", help="Element family for mixed FE space (default dg-cg)")
parser.add_argument("-nonlinear", help="Toggle nonlinear equations (default False)")
parser.add_argument("-stabilisation", help="Stabilisation method to use (default None)")

# Mesh adaptation
parser.add_argument("-norm_order", help="p for Lp normalisation (default 1)")
parser.add_argument("-normalisation", help="Normalisation method (default 'complexity')")
parser.add_argument("-adapt_field", help="Field to construct metric w.r.t")
parser.add_argument("-time_combine", help="Method for time-combining Hessians (default integrate)")
parser.add_argument("-hessian_lag", help="Compute Hessian every n timesteps (default 6)")
parser.add_argument("-target", help="Target space-time complexity (default 1.0e+03)")
parser.add_argument("-h_min", help="Minimum tolerated element size (default 100m)")
parser.add_argument("-h_max", help="Maximum tolerated element size (default 1000km)")

# QoI
parser.add_argument("-start_time", help="""
    Start time of period of interest in seconds (default zero)""")
parser.add_argument("-locations", help="""
    Locations of interest, separated by commas. Choose from {'Fukushima Daiichi', 'Onagawa',
    'Fukushima Daini', 'Tokai', 'Hamaoka', 'Tohoku', 'Tokyo'}. (Default 'Fukushima Daiichi')
    """)
parser.add_argument("-radius", help="Radius of interest (default 100km)")

# Outer loop
parser.add_argument("-num_adapt", help="Maximum number of adaptation loop iterations (default 35)")
parser.add_argument("-element_rtol", help="Relative tolerance for element count (default 0.005)")
parser.add_argument("-qoi_rtol", help="Relative tolerance for quantity of interest (default 0.005)")

# I/O and debugging
parser.add_argument("-save_meshes", help="Save final set of mesh DMPlexes to disk")
parser.add_argument("-plot_pvd", help="Toggle plotting to .pvd")
parser.add_argument("-debug", help="Print all debugging statements")
parser.add_argument("-debug_mode", help="Choose debugging mode from 'basic' and 'full'")

args, unknown = parser.parse_known_args()
p = args.norm_order


# --- Set parameters

plot_pvd = bool(args.plot_pvd or False)
if args.locations is None:
    locations = ['Fukushima Daiichi', ]
else:
    locations = args.locations.split(',')
radius = float(args.radius or 100.0e+03)
family = args.family or 'cg-cg'  # FIXME: what's wrong with dg-cg?
nonlinear = bool(args.nonlinear or False)
stabilisation = args.stabilisation or 'lax_friedrichs'
if stabilisation == 'none' or family == 'cg-cg' or not nonlinear:
    stabilisation = None
kwargs = {
    'approach': 'hessian',

    # Space-time domain
    'level': int(args.level or 0),
    'num_meshes': int(args.num_meshes or 12),
    'end_time': float(args.end_time or 24*60.0),

    # Physics
    'bathymetry_cap': 30.0,  # FIXME

    # Solver
    'family': family,
    'stabilisation': stabilisation,
    'use_wetting_and_drying': False,

    # Mesh adaptation
    'adapt_field': args.adapt_field or 'elevation',
    'hessian_time_combination': args.time_combine or 'integrate',
    'hessian_timestep_lag': float(args.hessian_lag or 1),
    'normalisation': args.normalisation or 'complexity',
    'norm_order': 1 if p is None else None if p == 'inf' else float(p),
    'target': float(args.target or 5.0e+03),
    'h_min': float(args.h_min or 1.0e+02),
    'h_max': float(args.h_max or 1.0e+06),

    # QoI
    'start_time': float(args.start_time or 0.0),
    'radius': radius,
    'locations': locations,

    # Outer loop
    'element_rtol': float(args.element_rtol or 0.005),
    'qoi_rtol': float(args.qoi_rtol or 0.005),
    'num_adapt': int(args.num_adapt or 35),

    # I/O and debugging
    'plot_pvd': plot_pvd,
    'debug': bool(args.debug or False),
    'debug_mode': args.debug_mode or 'basic'
}
assert 0.0 <= kwargs['start_time'] <= kwargs['end_time']
save_meshes = bool(args.save_meshes or False)


# --- Solve

op = TohokuHazardOptions(**kwargs)
swp = AdaptiveTsunamiProblem(op, nonlinear=nonlinear)  # TODO: Option to load meshes
logger = TimeDependentAdaptationLogger(swp, nonlinear=nonlinear, **kwargs)
swp.run_hessian_based()
logger.log(*unknown, fpath=op.di, save_meshes=save_meshes)
