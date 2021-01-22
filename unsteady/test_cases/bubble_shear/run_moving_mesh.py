from thetis import *

import argparse

from adapt_utils.adapt.r import MeshMover
from adapt_utils.adapt.recovery import *
from adapt_utils.unsteady.solver import AdaptiveProblem
from adapt_utils.unsteady.test_cases.bubble_shear.options import BubbleOptions


# --- Parse arguments

parser = argparse.ArgumentParser()
parser.add_argument("-n", help="Resolution of initial mesh.")
parser.add_argument("-conservative", help="Toggle conservative tracer equation")
parser.add_argument("-limiters", help="Toggle limiters for tracer equation")
parser.add_argument("-stabilisation", help="Stabilisation method")
parser.add_argument("-family", help="Choose finite element from 'cg' and 'dg'")
parser.add_argument("-dt_per_mesh_movement", help="Mesh movement frequency")
parser.add_argument("-alpha", help="Prominence of refined region.")
parser.add_argument("-debug", help="Toggle debugging mode.")
args = parser.parse_args()


# --- Set parameters

kwargs = {
    'tracer_family': args.family or 'dg',
    'stabilisation_tracer': args.stabilisation or 'lax_friedrichs',
    'use_automatic_sipg_parameter': False,  # We have an inviscid problem
    # 'use_limiter_for_tracers': bool(args.limiters or False),
    'use_limiter_for_tracers': False if args.limiters == "0" else True,
    'use_tracer_conservative_form': bool(args.conservative or False),  # FIXME?

    'nonlinear_method': 'relaxation',
    'r_adapt_rtol': 5.0e-02,
    'dt_per_mesh_movement': int(args.dt_per_mesh_movement or 1),
    'debug': bool(args.debug or False),
}
op = BubbleOptions(approach='monge_ampere', n=int(args.n or 1))
op.update(kwargs)
alpha = 5.0
eps = 1.0e-03  # Parameter controlling width of refined region


# --- Initialise mesh

tp = AdaptiveProblem(op)


def monitor(mesh):
    x, y = SpatialCoordinate(mesh)
    x0, y0, r = op.source_loc[0]
    return conditional(le(abs((x-x0)**2 + (y-y0)**2 - r**2), eps), alpha, 1.0)


mesh_mover = MeshMover(tp.meshes[0], monitor, method='monge_ampere', op=op)
mesh_mover.adapt()
tp.__init__(op, meshes=[Mesh(mesh_mover.x)])


# --- Solve the tracer transport problem

op.r_adapt_rtol = 1.0e-03
alpha = Constant(float(args.alpha or 10.0))
tp.set_initial_condition()


def monitor(mesh):
    P1 = FunctionSpace(mesh, "CG", 1)
    sol = project(tp.fwd_solutions_tracer[0], P1)
    g = recover_gradient(sol, op=op)
    gnorm = sqrt(dot(g, g))
    g_max = interpolate(gnorm, P1).vector().gather().max()
    return 1.0 + alpha*gnorm/g_max
    # frobenius = sqrt(H[0, 0]**2 + H[0, 1]**2 + H[1, 0]**2 + H[1, 1]**2)
    # frobenius_max = interpolate(frobenius, P1).vector().gather().max()
    # return 1.0 + alpha*frobenius/frobenius_max


tp.set_monitor_functions(monitor)
tp.solve_forward()
