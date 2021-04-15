from thetis import *

import argparse

from adapt_utils.unsteady.solver import AdaptiveProblem
from adapt_utils.unsteady.test_cases.bubble_shear.options import BubbleOptions


# --- Parse arguments

parser = argparse.ArgumentParser()
parser.add_argument("-n", help="Resolution of initial mesh.")
parser.add_argument("-dt", help="Timestep.")
parser.add_argument("-end_time", help="Simulation end time.")

parser.add_argument("-family", help="Choose finite element from 'cg' and 'dg'")
parser.add_argument("-conservative", help="Toggle conservative tracer equation")
parser.add_argument("-limiters", help="Toggle limiters for tracer equation")
parser.add_argument("-stabilisation", help="Stabilisation method")
parser.add_argument("-anisotropic_stabilisation", help="Use anisotropic stabilisation.")

parser.add_argument("-debug", help="Toggle debugging mode.")
args = parser.parse_args()


# --- Set parameters

kwargs = {
    'tracer_family': args.family or 'cg',
    'stabilisation_tracer': args.stabilisation or 'supg',
    'anisotropic_stabilisation': False if args.anisotropic_stabilisation == "0" else True,
    'use_automatic_sipg_parameter': False,  # We have an inviscid problem
    'use_limiter_for_tracers': bool(args.limiters or False),
    'use_tracer_conservative_form': bool(args.conservative or False),  # FIXME?
    'debug': bool(args.debug or False),
}
if os.getenv('REGRESSION_TEST') is not None:
    kwargs['dt'] = 0.015
    kwargs['end_time'] = 1.5
op = BubbleOptions(approach='fixed_mesh', n=int(args.n or 1))
op.update(kwargs)
if args.dt is not None:
    op.dt = float(args.dt)
if args.end_time is not None:
    op.end_time = float(args.end_time)


# --- Solve the tracer transport problem

tp = AdaptiveProblem(op)
tp.set_initial_condition()
init_sol = tp.fwd_solutions_tracer[0].copy(deepcopy=True)
init_l1_norm = norm(init_sol, norm_type='L1')
init_l2_norm = norm(init_sol, norm_type='L2')
tp.solve_forward()

# Compare initial and final tracer concentrations
final_sol = tp.fwd_solutions_tracer[0].copy(deepcopy=True)
final_l1_norm = norm(final_sol, norm_type='L1')
final_l2_norm = norm(final_sol, norm_type='L2')
abs_l2_error = errornorm(init_sol, final_sol, norm_type='L2')
print_output("Conservation error: {:.2f}%".format(100*abs(init_l1_norm-final_l1_norm)/init_l1_norm))
print_output("Relative L2 error:  {:.2f}%".format(100*abs_l2_error/init_l2_norm))
