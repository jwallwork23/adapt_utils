from thetis import print_output, norm

import argparse

from adapt_utils.test_cases.bubble_shear.options import BubbleOptions
from adapt_utils.solver import AdaptiveProblem


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
    'tracer_family': args.family or 'dg',  # FIXME: CG not working
    'n': int(args.n or 1),
    'debug': bool(args.debug or False),
    'prescribed_velocity': 'fluid',
    'num_adapt': 1,
    'nonlinear_method': 'quasi_newton',
    'r_adapt_rtol': 1.0e-3,
}
# initialisation_kwargs = {
#     'approach': 'monge_ampere',
#     'num_adapt': int(args.num_adapt or 1),
#     'adapt_field': 'solution_frobenius',
#     'alpha': 0.1,
# }

op = BubbleOptions(**kwargs)
tp = AdaptiveProblem(op)
tp.set_initial_condition()
init_norm = norm(tp.fwd_solutions_tracer[0])
if approach != 'fixed_mesh':
    raise NotImplementedError  # TODO
else:
    tp.solve_forward()
final_norm = norm(tp.fwd_solutions_tracer[0])
print_output("Initial norm:   {:.4e}".format(init_norm))
print_output("Final norm:     {:.4e}".format(final_norm))
print_output("Relative error: {:.2f}%".format(100*abs(1.0 - final_norm/init_norm)))
