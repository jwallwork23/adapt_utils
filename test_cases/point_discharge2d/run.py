import argparse

from adapt_utils.test_cases.point_discharge2d.options import *
from adapt_utils.tracer.solver2d import *


parser = argparse.ArgumentParser()
parser.add_argument('-approach', help="Mesh adaptation approach.")
parser.add_argument('-level', help="Number of uniform refinements to apply to the initial mesh.")
parser.add_argument('-centred', help="Toggle between centred or offset region of interest.")
parser.add_argument('-debug', help="Toggle debugging mode.")
args = parser.parse_args()

kwargs = {
    'approach': args.approach or 'fixed_mesh',
    'centred': bool(args.centred or 1),
    'plot_pvd': True,
    'debug': bool(args.debug or 0),
    'n': int(args.level or 0),
}


op = TelemacOptions(**kwargs)
tp = SteadyTracerProblem2d(op, levels=0 if op.approach == 'fixed_mesh' else 1)
if op.approach == 'fixed_mesh':
    tp.solve()
else:
    tp.adaptation_loop()
tp.plot_solution()
