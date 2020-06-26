from thetis import print_output, norm

import argparse

from adapt_utils.unsteady.test_cases.bubble_shear.options import BubbleOptions
from adapt_utils.unsteady.solver import AdaptiveProblem


parser = argparse.ArgumentParser()
parser.add_argument("interpretation", help="Choose from {'eulerian', 'lagrangian'}.")
parser.add_argument("-n", help="Resolution of initial mesh.")
parser.add_argument("-num_adapt", help="Number of initial mesh adaptations.")
parser.add_argument("-conservative", help="Toggle conservative tracer equation")
parser.add_argument("-limiters", help="Toggle limiters for tracer equation")
parser.add_argument("-stabilisation", help="Stabilisation method")
parser.add_argument("-family", help="Choose finite element from 'cg' and 'dg'")
parser.add_argument("-debug", help="Toggle debugging mode.")
args = parser.parse_args()

approach = 'fixed_mesh' if args.interpretation == 'eulerian' else 'ale'
kwargs = {
    'approach': approach,
    'n': int(args.n or 1),

    # Spatial discretisation
    'tracer_family': args.family or 'dg',
    'stabilisation': args.stabilisation or 'no',
    'use_automatic_sipg_parameter': False,
    'use_limiter_for_tracers': bool(args.limiters or True),
    'use_tracer_conservative_form': bool(args.conservative or False),

    # Mesh movement
    'prescribed_velocity': 'fluid',
    'nonlinear_method': 'quasi_newton',
    'r_adapt_rtol': 1.0e-3,

    # Misc
    'debug': bool(args.debug or False),
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
