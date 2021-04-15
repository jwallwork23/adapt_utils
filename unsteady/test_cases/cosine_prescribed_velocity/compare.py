from firedrake import norm, errornorm
from thetis import print_output

import numpy as np
import argparse

from adapt_utils.unsteady.test_cases.cosine_prescribed_velocity.options import *
from adapt_utils.unsteady.solver import AdaptiveProblem


# --- Parse arguments

parser = argparse.ArgumentParser()
parser.add_argument("-conservative", help="Toggle conservative tracer equation")
parser.add_argument("-limiters", help="Toggle limiters for tracer equation")
parser.add_argument("-stabilisation", help="Stabilisation method")
parser.add_argument("-family", help="Choose finite element from 'cg' and 'dg'")
parser.add_argument("-debug", help="Toggle debugging mode")
args = parser.parse_args()

# --- Set parameters

kwargs = {
    'approach': 'fixed_mesh',

    # Discretisation
    'tracer_family': args.family or 'dg',
    'stabilisation': args.stabilisation,
    'use_automatic_sipg_parameter': False,  # We have an inviscid problem
    'use_limiter_for_tracers': bool(args.limiters or True),
    'use_tracer_conservative_form': bool(args.conservative or False),

    # Misc
    'debug': bool(args.debug or False),
}


# --- Create solver and copy initial solution

ep = AdaptiveProblem(CosinePrescribedVelocityOptions(**kwargs))
ep.set_initial_condition()
init_sol = ep.fwd_solutions_tracer[0].copy(deepcopy=True)
init_norm = norm(init_sol)


# --- Eulerian interpretation

ep.solve_forward()
final_sol_eulerian = ep.fwd_solutions_tracer[-1]
relative_error_eulerian = abs(errornorm(init_sol, final_sol_eulerian)/init_norm)
print_output("Relative error in Eulerian case:   {:.2f}%".format(100*relative_error_eulerian))


# --- Lagrangian interpretation

kwargs['approach'] = 'lagrangian'
lp = AdaptiveProblem(CosinePrescribedVelocityOptions(**kwargs))
lp.set_initial_condition()
init_sol = lp.fwd_solutions_tracer[0].copy(deepcopy=True)
init_coords = lp.meshes[0].coordinates.dat.data.copy()
lp.solve_forward()
final_sol_lagrangian = lp.fwd_solutions_tracer[-1]

final_coords = lp.meshes[-1].coordinates.dat.data
final_coords[:] -= [10.0, 0.0]  # TODO: Implement periodicity
if not np.allclose(init_coords, final_coords):  # FIXME
    raise ValueError("Initial and final mesh coordinates do not match")


# --- Comparison

relative_error_lagrangian = abs(errornorm(init_sol, final_sol_lagrangian)/init_norm)
print_output("Relative error in Lagrangian case: {:.2f}%".format(100*relative_error_lagrangian))
assert relative_error_lagrangian < relative_error_eulerian
