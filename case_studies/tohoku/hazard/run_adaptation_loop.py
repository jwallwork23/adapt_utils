import argparse

from adapt_utils.case_studies.tohoku.hazard.options import TohokuHazardOptions
from adapt_utils.io import TimeDependentAdaptationLogger
from adapt_utils.swe.tsunami.solver import AdaptiveTsunamiProblem


# --- Parse arguments

parser = argparse.ArgumentParser(prog="run_adapt")

# Space-time domain
parser.add_argument("-end_time", help="End time of simulation in seconds (default 1400s, i.e. 24min)")
parser.add_argument("-level", help="(Integer) resolution for initial mesh (default 0)")
parser.add_argument("-num_meshes", help="Number of meshes to consider (default 24)")

# Physics
parser.add_argument("-base_viscosity", help="Base viscosity (default 1.0e-03)")

# Solver
parser.add_argument("-family", help="Element family for mixed FE space (default cg-cg)")
parser.add_argument("-nonlinear", help="Toggle nonlinear equations (default False)")
parser.add_argument("-stabilisation", help="Stabilisation method to use (default None)")

# Mesh adaptation
parser.add_argument("-approach", help="Mesh adaptation approach")
parser.add_argument("-norm_order", help="p for Lp normalisation (default 1)")
parser.add_argument("-normalisation", help="Normalisation method (default 'complexity')")
parser.add_argument("-adapt_field", help="Field to construct metric w.r.t")
parser.add_argument("-time_combine", help="Method for time-combining Hessians (default 'integrate')")
parser.add_argument("-hessian_lag", help="Compute Hessian every n timesteps (default 1)")
parser.add_argument("-target", help="Target space-time complexity (default 5.0e+03)")
parser.add_argument("-h_min", help="Minimum tolerated element size (default 100m)")
parser.add_argument("-h_max", help="Maximum tolerated element size (default 1000km)")

# QoI
parser.add_argument("-start_time", help="""
    Start time of period of interest in seconds (default 1200s, i.e. 20min)""")
parser.add_argument("-locations", help="""
    Locations of interest, separated by commas. Choose from {'Fukushima Daiichi', 'Onagawa',
    'Fukushima Daini', 'Tokai', 'Hamaoka', 'Tohoku', 'Tokyo'}. (Default 'Fukushima Daiichi')
    """)
parser.add_argument("-radius", help="Radius of interest (default 100km)")

# Outer loop
parser.add_argument("-max_adapt", help="Maximum number of adaptation loop iterations (default 35)")
parser.add_argument("-element_rtol", help="Relative tolerance for element count (default 0.005)")
parser.add_argument("-qoi_rtol", help="Relative tolerance for quantity of interest (default 0.005)")
parser.add_argument("-target_base")
parser.add_argument("-iterations")

# I/O and debugging
parser.add_argument("-save_meshes", help="Save final set of mesh DMPlexes to disk")
parser.add_argument("-plot_pvd", help="Toggle plotting to .pvd")
parser.add_argument("-debug", help="Print all debugging statements")
parser.add_argument("-debug_mode", help="Choose debugging mode from 'basic' and 'full'")

args, unknown = parser.parse_known_args()
p = args.norm_order


# --- Set parameters

plot_pvd = False if args.plot_pvd == '0' else True
if args.locations is None:
    locations = ['Fukushima Daiichi']
else:
    locations = args.locations.split(',')
radius = float(args.radius or 100.0e+03)
family = args.family or 'cg-cg'
nonlinear = bool(args.nonlinear or False)
stabilisation = args.stabilisation or 'lax_friedrichs'
if stabilisation == 'none' or family == 'cg-cg' or not nonlinear:
    stabilisation = None
approach = args.approach or 'hessian'
kwargs = {
    'approach': approach,

    # Space-time domain
    'level': int(args.level or 0),
    'end_time': float(args.end_time or 1440.0),
    'num_meshes': int(args.num_meshes or 24),

    # Timestepping
    'dt_per_export': 1,

    # Physics
    'bathymetry_cap': 30.0,
    'base_viscosity': Constant(args.base_viscosity or 1.0e-03),

    # Solver
    'family': family,
    'stabilisation': stabilisation,
    'use_wetting_and_drying': False,

    # Mesh adaptation
    'adapt_field': args.adapt_field or 'all_avg',
    'hessian_time_combination': args.time_combine or 'integrate',
    'hessian_timestep_lag': int(args.hessian_lag or 1),  # NOTE: Not used in weighted Hessian
    'normalisation': args.normalisation or 'complexity',
    'norm_order': 1 if p is None else None if p == 'inf' else float(p),
    'min_adapt': 3,
    'h_min': float(args.h_min or 1.0e+02),
    'h_max': float(args.h_max or 1.0e+06),

    # QoI
    'start_time': float(args.start_time or 1200.0),
    'radius': radius,
    'locations': locations,

    # Outer loop
    'element_rtol': float(args.element_rtol or 0.005),
    'qoi_rtol': float(args.qoi_rtol or 0.005),
    'max_adapt': int(args.max_adapt or 5),  # As recommended in [Belme et al. 2012]
    'target': float(args.target or 24*4000),
    'target_base': float(args.target_base or 2.0),
    'outer_iterations': int(args.iterations or 5),

    # Misc
    'plot_pvd': plot_pvd,
    'debug': bool(args.debug or False),
    'debug_mode': args.debug_mode or 'basic'
}
assert 0.0 <= kwargs['start_time'] <= kwargs['end_time']
save_meshes = bool(args.save_meshes or False)
op = TohokuHazardOptions(**kwargs)


# --- Solve

for i in range(op.outer_iterations):
    kwargs['target'] = op.target*op.target_base**i
    swp = AdaptiveTsunamiProblem(op, nonlinear=nonlinear)
    logger = TimeDependentAdaptationLogger(swp, nonlinear=nonlinear, **kwargs)
    swp.run()
    logger.log(*unknown, fpath=op.di, save_meshes=save_meshes)
