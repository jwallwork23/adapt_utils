from thetis import *

import argparse

from adapt_utils.unsteady.solver import AdaptiveProblem
from adapt_utils.unsteady.test_cases.bubble_shear.options import BubbleOptions


# --- Parse arguments

parser = argparse.ArgumentParser()
parser.add_argument("-n", help="Resolution of initial mesh.")
parser.add_argument("-conservative", help="Toggle conservative tracer equation")
parser.add_argument("-limiters", help="Toggle limiters for tracer equation")
parser.add_argument("-stabilisation", help="Stabilisation method")
parser.add_argument("-family", help="Choose finite element from 'cg' and 'dg'")

parser.add_argument("-target", help="Target complexity")
parser.add_argument("-num_meshes", help="Number of meshes in the sequence")

parser.add_argument("-debug", help="Toggle debugging mode.")
args = parser.parse_args()


# --- Set parameters

kwargs = {

    # Solver
    'tracer_family': args.family or 'dg',
    'stabilisation_tracer': args.stabilisation or 'lax_friedrichs',
    'use_automatic_sipg_parameter': False,  # We have an inviscid problem
    # 'use_limiter_for_tracers': bool(args.limiters or False),
    'use_limiter_for_tracers': False if args.limiters == "0" else True,
    'use_tracer_conservative_form': bool(args.conservative or False),  # FIXME?

    # Mesh adaptation
    'num_meshes': int(args.num_meshes or 200),
    'max_adapt': 8,
    'hessian_time_combination': 'intersect',
    'target': float(args.target or 4000),
    'norm_order': 1,
    'normalisation': 'complexity',

    # Debugging
    'debug': bool(args.debug or False),
}
op = BubbleOptions(approach='hessian', n=int(args.n or 1))
op.update(kwargs)


# --- Solve the tracer transport problem

tp = AdaptiveProblem(op)
tp.run()
