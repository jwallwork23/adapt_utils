from firedrake import *
from thetis import print_output

import argparse

from adapt_utils.test_cases.ale_trivial.options import ALEAdvectionOptions
from adapt_utils.tracer.solver2d import UnsteadyTracerProblem2d
from adapt_utils.adapt.r import MeshMover


parser = argparse.ArgumentParser()
parser.add_argument("-interpretation", help="Choose from {'eulerian', 'lagrangian'}.")
args = parser.parse_args()

interpretation = args.interpretation or 'eulerian'
approach = 'fixed_mesh' if interpretation == 'eulerian' else 'ale'

op = ALEAdvectionOptions(approach=approach, prescribed_velocity='fluid')
tp = UnsteadyTracerProblem2d(op)
init_norm = norm(tp.solution)
if approach == 'fixed_mesh':
    tp.solve()
elif approach == 'ale':
    tp.solve_ale()
else:
    raise NotImplementedError
final_norm = norm(tp.solution)
print_output("Initial norm:   {:.4e}".format(init_norm))
print_output("Final norm:     {:.4e}".format(final_norm))
print_output("Relative error: {:.2f}%".format(100*abs(1.0 - final_norm/init_norm)))
