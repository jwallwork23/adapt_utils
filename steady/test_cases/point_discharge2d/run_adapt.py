import argparse

from adapt_utils.steady.solver import AdaptiveSteadyProblem
from adapt_utils.steady.test_cases.point_discharge2d.options import PointDischarge2dOptions


# --- Parse arguments

parser = argparse.ArgumentParser()

# Solver
parser.add_argument('-level', help="Number of uniform refinements to apply to the initial mesh.")
parser.add_argument('-family', help="Finite element family.")
parser.add_argument('-stabilisation', help="Stabilisation method to use.")

# Mesh adaptation
parser.add_argument('-approach', help="Mesh adaptation approach.")
parser.add_argument('-target', help="Target complexity.")
parser.add_argument('-normalisation', help="Metric normalisation strategy.")
parser.add_argument('-min_adapt', help="Minimum number of mesh adaptations.")
parser.add_argument('-max_adapt', help="Maximum number of mesh adaptations.")

# I/O and debugging
parser.add_argument('-offset', help="Toggle between aligned or offset region of interest.")
parser.add_argument('-debug', help="Toggle debugging mode.")
args = parser.parse_args()


# --- Set parameters

family = args.family or 'cg'
assert family in ('cg', 'dg')
kwargs = {
    'level': int(args.level or 0),

    # QoI
    'aligned': not bool(args.offset or False),

    # Mesh adaptation
    'approach': args.approach or 'dwr',
    'target': float(args.target or 1.0e+03),
    'norm_order': 1,
    'min_adapt': int(args.min_adapt or 0),
    'max_adapt': int(args.max_adapt or 35),

    # I/O and debugging
    'plot_pvd': True,
    'debug': bool(args.debug or 0),
}
op = PointDischarge2dOptions(**kwargs)
op.tracer_family = family
op.stabilisation = args.stabilisation
op.use_automatic_sipg_parameter = op.tracer_family == 'dg'
op.normalisation = args.normalisation or 'complexity'  # FIXME: error
op.print_debug(op)


# --- Solve

tp = AdaptiveSteadyProblem(op)
tp.run()
