import argparse
import os

from adapt_utils.io import create_directory, File, save_mesh
from adapt_utils.steady.solver import AdaptiveSteadyProblem
from adapt_utils.steady.test_cases.point_discharge2d.options import PointDischarge2dOptions


# --- Parse arguments

parser = argparse.ArgumentParser()
parser.add_argument('approach', help="Mesh adaptation approach.")

# Solver
parser.add_argument('-level', help="Number of uniform refinements to apply to the initial mesh.")
parser.add_argument('-family', help="Finite element family.")
parser.add_argument('-stabilisation', help="Stabilisation method to use.")
parser.add_argument('-anisotropic_stabilisation', help="Use anisotropic cell size measure?")

# Mesh adaptation
parser.add_argument('-target', help="Target complexity.")
parser.add_argument('-normalisation', help="Metric normalisation strategy.")
parser.add_argument('-norm_order', help="Metric normalisation order.")
parser.add_argument('-convergence_rate', help="Convergence rate for anisotropic DWR.")
parser.add_argument('-min_adapt', help="Minimum number of mesh adaptations.")
parser.add_argument('-max_adapt', help="Maximum number of mesh adaptations.")
parser.add_argument('-enrichment_method', help="Choose from {'GE_hp', 'GE_h', 'GE_p', 'PR', 'DQ'}.")

# I/O and debugging
parser.add_argument('-plot_indicator', help="Plot error indicator to file.")
parser.add_argument('-offset', help="Toggle between aligned or offset region of interest.")
parser.add_argument('-debug', help="Toggle debugging mode.")
args = parser.parse_args()
p = 'inf' if args.norm_order == 'inf' else float(args.norm_order or 1)
alpha = float(args.convergence_rate or 2)


# --- Set parameters

family = args.family or 'cg'
assert family in ('cg', 'dg')
kwargs = {
    'level': int(args.level or 0),

    # QoI
    'aligned': not bool(args.offset or False),

    # Mesh adaptation
    'approach': args.approach,
    'target': float(args.target or 5.0e+02),
    'norm_order': p,
    'convergence_rate': alpha,
    'min_adapt': int(args.min_adapt or 3),
    'max_adapt': int(args.max_adapt or 35),
    'enrichment_method': args.enrichment_method or 'GE_h',

    # I/O and debugging
    'plot_pvd': True,
    'debug': bool(args.debug or 0),
}
op = PointDischarge2dOptions(**kwargs)
op.tracer_family = family
stabilisation = args.stabilisation or 'supg'
op.stabilisation_tracer = None if stabilisation == 'none' else stabilisation
op.anisotropic_stabilisation = False if args.anisotropic_stabilisation == '0' else True
op.use_automatic_sipg_parameter = op.tracer_family == 'dg'
op.di = create_directory(os.path.join(op.di, op.stabilisation_tracer or family, op.enrichment_method))
op.normalisation = args.normalisation or 'complexity'  # FIXME: error
op.print_debug(op)

# --- Solve

tp = AdaptiveSteadyProblem(op)
tp.run()

if bool(args.plot_indicator or False):
    indicator_file = File(os.path.join(op.di, "indicator.pvd"))
    indicator_file.write(tp.indicator[op.enrichment_method])

# Export to HDF5
save_mesh(tp.mesh, "mesh", fpath=op.di)
