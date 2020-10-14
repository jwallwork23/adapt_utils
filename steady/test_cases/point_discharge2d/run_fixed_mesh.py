import argparse
import os

from adapt_utils.steady.solver import AdaptiveSteadyProblem
from adapt_utils.steady.test_cases.point_discharge2d.options import PointDischarge2dOptions


# --- Parse arguments

parser = argparse.ArgumentParser()
parser.add_argument('-level', help="Number of uniform refinements to apply to the initial mesh.")
parser.add_argument('-family', help="Finite element family")
parser.add_argument('-stabilisation', help="Stabilisation method to use")
parser.add_argument('-use_automatic_sipg_parameter')
parser.add_argument('-offset', help="Toggle between aligned or offset region of interest.")
parser.add_argument('-debug', help="Toggle debugging mode.")
args = parser.parse_args()


# --- Set parameters

family = args.family
assert family in ('cg', 'dg')
kwargs = {
    'level': int(args.level or 0),
    'aligned': not bool(args.offset or False),
    'plot_pvd': True,
    'debug': bool(args.debug or False),
}
op = PointDischarge2dOptions(approach='fixed_mesh', **kwargs)
op.tracer_family = family
op.stabilisation = args.stabilisation
op.di = os.path.join(op.di, args.stabilisation or family)
if op.tracer_family == 'cg':
    op.use_automatic_sipg_parameter = False
else:
    auto_sipg = bool(args.use_automatic_sipg_parameter or False)
    op.use_automatic_sipg_parameter = auto_sipg
    if auto_sipg:
        op.di += '_sipg'

# TODO: Limiters?

# --- Solve

tp = AdaptiveSteadyProblem(op)
tp.solve_forward()
