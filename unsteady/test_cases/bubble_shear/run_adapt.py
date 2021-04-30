import argparse

from adapt_utils.norms import *
from adapt_utils.unsteady.solver import AdaptiveProblem
from adapt_utils.unsteady.test_cases.bubble_shear.options import BubbleOptions


# --- Parse arguments

parser = argparse.ArgumentParser()
parser.add_argument("-n", help="Resolution of initial mesh.")
parser.add_argument("-dt", help="Timestep.")
parser.add_argument("-end_time", help="Simulation end time.")

parser.add_argument("-limiters", help="Toggle limiters for tracer equation")
parser.add_argument("-stabilisation", help="Stabilisation method")
parser.add_argument("-family", help="Choose finite element from 'cg' and 'dg'")

parser.add_argument("-approach", help="Mesh adaptation approach")
parser.add_argument("-target", help="Target complexity")
parser.add_argument("-num_meshes", help="Number of meshes in the sequence")
parser.add_argument("-max_adapt", help="Maximum number of adaptation steps")
parser.add_argument("-hessian_time_combination")

parser.add_argument("-debug", help="Toggle debugging mode.")
args = parser.parse_args()


# --- Set parameters

kwargs = {

    # Solver
    'tracer_family': args.family or 'cg',
    'stabilisation_tracer': args.stabilisation or 'supg',
    'use_limiter_for_tracers': bool(args.limiters or False),

    # Mesh adaptation
    'approach': args.approach or 'hessian',
    'num_meshes': int(args.num_meshes or 50),
    'max_adapt': int(args.max_adapt or 3),
    'hessian_time_combination': args.hessian_time_combination or 'integrate',
    'target': float(args.target or 4000),
    'norm_order': 1,
    'normalisation': 'complexity',

    # Debugging
    'debug': bool(args.debug or False),
}
op = BubbleOptions(approach='hessian', n=int(args.n or 1))
op.update(kwargs)
if args.dt is not None:
    op.dt = float(args.dt)
if args.end_time is not None:
    op.end_time = float(args.end_time)

# --- Solve the tracer transport problem

tp = AdaptiveProblem(op)
tp.run()

final_sol = tp.fwd_solutions_tracer[-1].copy(deepcopy=True)
final_l1_norm = norm(final_sol, norm_type='L1')
final_l2_norm = norm(final_sol, norm_type='L2')
tp.set_initial_condition(i=-1)
init_sol = tp.fwd_solutions_tracer[-1].copy(deepcopy=True)
init_l1_norm = norm(init_sol, norm_type='L1')
init_l2_norm = norm(init_sol, norm_type='L2')
abs_l2_error = errornorm(init_sol, final_sol, norm_type='L2')
print("Conservation error: {:.2f}%".format(100*abs(init_l1_norm-final_l1_norm)/init_l1_norm))
print("Relative L2 error:  {:.2f}%".format(100*abs_l2_error/init_l2_norm))
