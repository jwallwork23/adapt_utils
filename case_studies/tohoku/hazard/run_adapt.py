import argparse
import matplotlib.pyplot as plt
import os
from time import perf_counter

from adapt_utils.case_studies.tohoku.hazard.options import TohokuHazardOptions
from adapt_utils.io import TimeDependentAdaptationLogger
from adapt_utils.swe.tsunami.solver import AdaptiveTsunamiProblem


# --- Parse arguments

parser = argparse.ArgumentParser(prog="run_adapt")

# Space-time domain
parser.add_argument("-end_time", help="End time of simulation in seconds (default 1400s, i.e. 24min)")
parser.add_argument("-level", help="(Integer) resolution for initial mesh (default 0)")
parser.add_argument("-num_meshes", help="Number of meshes to consider (default 24)")
parser.add_argument("-dt", help="Timestep")

# Physics
parser.add_argument("-base_viscosity", help="Base viscosity (default 1.0e-03)")

# Solver
parser.add_argument("-family", help="Element family for mixed FE space (default cg-cg)")
parser.add_argument("-nonlinear", help="Toggle nonlinear equations (default False)")
parser.add_argument("-stabilisation", help="Stabilisation method to use (default None)")
parser.add_argument("-lu", help="Hit both forward and adjoint systems with LU")

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
    Start time of period of interest in seconds (default 1220s, i.e. 20min)""")
parser.add_argument("-locations", help="""
    Locations of interest, separated by commas. Choose from {'Fukushima Daiichi', 'Onagawa',
    'Fukushima Daini', 'Tokai', 'Hamaoka', 'Tohoku', 'Tokyo'}. (Default 'Fukushima Daiichi')
    """)
parser.add_argument("-radius", help="Radius of interest (default 100km)")
parser.add_argument("-kernel_shape", help="""
    Choose kernel shape from {'gaussian', 'circular_bump', 'ball'}.
    """)

# Outer loop
parser.add_argument("-min_adapt", help="Minimum number of adaptation loop iterations (default 3)")
parser.add_argument("-max_adapt", help="Maximum number of adaptation loop iterations (default 5)")
parser.add_argument("-element_rtol", help="Relative tolerance for element count (default 0.005)")
parser.add_argument("-qoi_rtol", help="Relative tolerance for quantity of interest (default 0.005)")

# I/O and debugging
parser.add_argument("-save_meshes", help="Save final set of mesh DMPlexes to disk")
parser.add_argument("-plot_pdf", help="Toggle plotting to .pdf")
parser.add_argument("-plot_png", help="Toggle plotting to .png")
parser.add_argument("-plot_pvd", help="Toggle plotting to .pvd")
parser.add_argument("-plot_all", help="Toggle plotting to .pdf, .png and .pvd")
parser.add_argument("-debug", help="Print all debugging statements")
parser.add_argument("-debug_mode", help="Choose debugging mode from 'basic' and 'full'")

args, unknown = parser.parse_known_args()
p = args.norm_order


# --- Set parameters

plot_pdf = bool(args.plot_pdf or False)
plot_png = bool(args.plot_png or False)
plot_pvd = False if args.plot_pvd == '0' else True
plot_all = bool(args.plot_all or False)
if plot_all:
    plot_pvd = plot_pdf = plot_png = True
plot_any = plot_pdf or plot_png
extensions = []
if plot_pdf:
    extensions.append('pdf')
if plot_png:
    extensions.append('png')
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
    'inversion_level': 2,  # TODO: Avoid this hard-code

    # Space-time domain
    'level': int(args.level or 0),
    'end_time': float(args.end_time or 1440.0),
    'num_meshes': int(args.num_meshes or 24),

    # Physics
    'bathymetry_cap': 30.0,
    'base_viscosity': float(args.base_viscosity or 1.0e-03),

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
    'target': float(args.target or 5.0e+03),
    'h_min': float(args.h_min or 1.0e+02),
    'h_max': float(args.h_max or 1.0e+06),

    # QoI
    'start_time': float(args.start_time or 1200.0),
    'radius': radius,
    'locations': locations,
    'kernel_shape': args.kernel_shape or 'ball',

    # Outer loop
    'element_rtol': float(args.element_rtol or 0.005),
    'qoi_rtol': float(args.qoi_rtol or 0.005),
    'min_adapt': int(args.min_adapt or 3),
    'max_adapt': int(args.max_adapt or 5),  # As recommended in [Belme et al. 2012]

    # Misc
    'plot_pvd': plot_pvd,
    'debug': bool(args.debug or False),
    'debug_mode': args.debug_mode or 'basic'
}
assert 0.0 <= kwargs['start_time'] <= kwargs['end_time']
save_meshes = bool(args.save_meshes or False)
op = TohokuHazardOptions(**kwargs)
if args.dt is not None:
    op.dt = float(args.dt)
op.dt_per_export = 1

if bool(args.lu or False):
    from adapt_utils.params import lu_params
    op.solver_parameters['shallow_water'] = lu_params
    op.adjoint_solver_parameters['shallow_water'] = lu_params
op.solver_parameters['shallow_water']['ksp_converged_reason'] = None
op.adjoint_solver_parameters['shallow_water']['ksp_converged_reason'] = None


# --- Solve

swp = AdaptiveTsunamiProblem(op, nonlinear=nonlinear)
logger = TimeDependentAdaptationLogger(swp, nonlinear=nonlinear, **kwargs)
tic = perf_counter()
swp.run()
logger.logstr += logger.msg.format("CPU time", perf_counter() - tic)
logger.log(*unknown, fpath=op.di, save_meshes=save_meshes)


# --- Plotting

if plot_any:
    N = swp.num_cells[-1]
    plt.pie(N, labels=["Mesh {:d} ({:d})".format(i, n) for i, n in enumerate(N)])
    plt.title("Element counts for {:s} adaptation".format(approach.upper()))
    for ext in extensions:
        plt.savefig(os.path.join(op.di, 'pie' + ext))
    plt.show()
