from firedrake import norm, errornorm
from thetis import print_output

import numpy as np
import argparse

from adapt_utils.unsteady.test_cases.ale_trivial.options import ALEAdvectionOptions
from adapt_utils.unsteady.solver import AdaptiveProblem


# Collect user specified arguments
parser = argparse.ArgumentParser()
parser.add_argument("interpretation", help="Choose from {'eulerian', 'lagrangian'}.")
parser.add_argument("-conservative", help="Toggle conservative tracer equation")
parser.add_argument("-limiters", help="Toggle limiters for tracer equation")
parser.add_argument("-stabilisation", help="Stabilisation method")
parser.add_argument("-family", help="Choose finite element from 'cg' and 'dg'")
parser.add_argument("-debug", help="Toggle debugging mode")
args = parser.parse_args()

approach = 'fixed_mesh' if args.interpretation == 'eulerian' else 'lagrangian'

# Setup
kwargs = {
    'approach': approach,

    # Discretisation
    'tracer_family': args.family or 'dg',
    'stabilisation': args.stabilisation,
    'use_automatic_sipg_parameter': False,  # We have an inviscid problem
    'use_limiter_for_tracers': bool(args.limiters or True),
    'use_tracer_conservative_form': bool(args.conservative or False),

    # Misc
    'debug': bool(args.debug or False),
}
op = ALEAdvectionOptions(**kwargs)
tp = AdaptiveProblem(op)
tp.set_initial_condition()

# Get initial solution and coordinates
init_sol = tp.fwd_solutions_tracer[0].copy(deepcopy=True)
init_norm = norm(init_sol)
init_coords = tp.meshes[0].coordinates.dat.data.copy()

# Solve PDE
tp.solve_forward()
final_sol = tp.fwd_solutions_tracer[-1]
final_norm = norm(final_sol)
final_coords = tp.meshes[-1].coordinates.dat.data

# Check final coords match initial coords
if approach == 'lagrangian':
    final_coords[:] -= [10.0, 0.0]  # TODO: Implement periodicity
try:
    assert np.allclose(init_coords, final_coords)
except AssertionError:
    raise ValueError("Initial and final mesh coordinates do not match")

# Compute relative errors
print_output("Initial norm:        {:.4e}".format(init_norm))
print_output("Final norm:          {:.4e}".format(final_norm))
print_output("Relative error:      {:.2f}%".format(100*abs(errornorm(init_sol, final_sol)/init_norm)))
