import argparse

from adapt_utils.steady.solver import AdaptiveSteadyProblem
from adapt_utils.steady.test_cases.point_discharge2d.options import PointDischarge2dOptions


# --- Parse arguments

parser = argparse.ArgumentParser()
parser.add_argument('-approach', help="Mesh adaptation approach.")
parser.add_argument('-level', help="Number of uniform refinements to apply to the initial mesh.")
parser.add_argument('-offset', help="Toggle between aligned or offset region of interest.")
parser.add_argument('-debug', help="Toggle debugging mode.")
args = parser.parse_args()


# --- Set parameters

kwargs = {
    'approach': args.approach or 'fixed_mesh',
    'aligned': not bool(args.offset or False),
    'plot_pvd': True,
    'debug': bool(args.debug or 0),
    'level': int(args.level or 0),
}
op = PointDischarge2dOptions(**kwargs)


# --- Solve

tp = AdaptiveSteadyProblem(op, levels=0 if op.approach == 'fixed_mesh' else 1)
if op.approach == 'fixed_mesh':
    tp.solve_forward()
else:
    tp.adaptation_loop()  # TODO
