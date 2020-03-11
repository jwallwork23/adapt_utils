from firedrake import norm
from thetis import print_output

import argparse

from adapt_utils.test_cases.bubble_shear.options import BubbleOptions
# from adapt_utils.tracer.solver2d_thetis import UnsteadyTracerProblem2d_Thetis
from adapt_utils.tracer.solver2d import UnsteadyTracerProblem2d


parser = argparse.ArgumentParser()
parser.add_argument("interpretation", help="Choose from {'eulerian', 'lagrangian'}.")
parser.add_argument("-family", help="Choose from {'cg', 'dg'}.")
parser.add_argument("-n", help="Resolution of initial mesh.")
parser.add_argument("-num_adapt", help="Number of initial mesh adaptations.")
parser.add_argument("-debug", help="Toggle debugging mode.")
args = parser.parse_args()

approach = 'fixed_mesh' if args.interpretation == 'eulerian' else 'ale'
kwargs = {
    'approach': approach,
    'family': args.family or 'dg',  # FIXME: CG not working
    'n': int(args.n or 1),
    'debug': bool(args.debug or False),
    'prescribed_velocity': 'fluid',
    'num_adapt': 1,
    'nonlinear_method': 'quasi_newton',
    'r_adapt_rtol': 1.0e-3,
}
initialisation_kwargs = {
    'approach': 'monge_ampere',
    'num_adapt': int(args.num_adapt or 1),
    'adapt_field': 'solution_frobenius',
    'alpha': 0.1,
}

op = BubbleOptions(**kwargs)
# tp = UnsteadyTracerProblem2d_Thetis(op)
tp = UnsteadyTracerProblem2d(op)
if approach == 'ale':
    tp.initialise_mesh(**initialisation_kwargs)
    tp.set_start_condition()
tp.setup_solver_forward()
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
