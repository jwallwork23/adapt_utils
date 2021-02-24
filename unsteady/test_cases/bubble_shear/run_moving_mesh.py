from thetis import *

import argparse

from adapt_utils.adapt.r import MeshMover
from adapt_utils.adapt.recovery import *
from adapt_utils.unsteady.solver import AdaptiveProblem
from adapt_utils.unsteady.test_cases.bubble_shear.options import BubbleOptions


# --- Parse arguments

parser = argparse.ArgumentParser()
parser.add_argument("-n", help="Resolution of initial mesh.")
parser.add_argument("-dt", help="Timestep.")
parser.add_argument("-end_time", help="Simulation end time.")

parser.add_argument("-limiters", help="Toggle limiters for tracer equation")
parser.add_argument("-stabilisation", help="Stabilisation method")
parser.add_argument("-anisotropic_stabilisation", help="Toggle anisotropic stabilisation")
parser.add_argument("-family", help="Choose finite element from 'cg' and 'dg'")

parser.add_argument("-dt_per_mesh_movement", help="Mesh movement frequency")
parser.add_argument("-alpha", help="Prominence of refined region.")
parser.add_argument("-debug", help="Toggle debugging mode.")
args = parser.parse_args()


# --- Set parameters

kwargs = {
    'tracer_family': args.family or 'cg',
    'stabilisation_tracer': args.stabilisation or 'supg',
    'use_automatic_sipg_parameter': False,  # We have an inviscid problem
    'anisotropic_stabilisation': False if args.anisotropic_stabilisation == "0" else True,
    'use_limiter_for_tracers': bool(args.limiters or False),

    'nonlinear_method': 'relaxation',
    'r_adapt_rtol': 5.0e-02,
    'dt_per_mesh_movement': int(args.dt_per_mesh_movement or 1),
    'debug': bool(args.debug or False),
    'debug_mode': 'full',
}
op = BubbleOptions(approach='monge_ampere', n=int(args.n or 1))
op.update(kwargs)
if args.dt is not None:
    op.dt = float(args.dt)
if args.end_time is not None:
    op.end_time = float(args.end_time)
alpha = 5.0
eps = 1.0e-03  # Parameter controlling width of refined region


# --- Initialise mesh

tp = AdaptiveProblem(op)

ring = lambda x, y: conditional(le(abs((x-x0)**2 + (y-y0)**2 - r**2), eps), alpha, 1.0)
x0, y0, r = op.source_loc[0]


def ring_monitor(mesh):
    return ring(*SpatialCoordinate(mesh))


mesh_mover = MeshMover(tp.meshes[0], ring_monitor, method='monge_ampere', op=op)
mesh_mover.adapt()
tp.__init__(op, meshes=[Mesh(mesh_mover.x)])


# --- Solve the tracer transport problem

op.r_adapt_rtol = 1.0e-03
alpha = Constant(float(args.alpha or 10.0))
tp.set_initial_condition()
tp.simulation_time = 0.0
dt = Constant(op.dt)
theta = Constant(1.0)  # Implicit Euler
dx = dx(degree=3)


def static_gradient_monitor(mesh):
    """
    Simple monitor function based on the gradient.
    """
    P1 = FunctionSpace(mesh, "CG", 1)
    sol = project(tp.fwd_solutions_tracer[0], P1)
    g = recover_gradient(sol, op=op)
    gnorm = sqrt(dot(g, g))
    g_max = interpolate(gnorm, P1).vector().gather().max()
    return 1.0 + alpha*gnorm/g_max


def static_hessian_monitor(mesh):
    P1 = FunctionSpace(mesh, "CG", 1)
    sol = project(tp.fwd_solutions_tracer[0], P1)
    H = recover_hessian(sol, op=op)
    frobenius = sqrt(H[0, 0]**2 + H[0, 1]**2 + H[1, 0]**2 + H[1, 1]**2)
    frobenius_max = interpolate(frobenius, P1).vector().gather().max()
    return 1.0 + alpha*frobenius/frobenius_max


def advected_monitor(monitor):  # FIXME
    def wrapper(mesh):
        t = tp.simulation_time
        coords = mesh.coordinates
        P1 = FunctionSpace(mesh, "CG", 1)
        monitor_old = static_gradient_monitor(mesh)
        u = op.get_velocity(coords, t)
        u_new = op.get_velocity(coords, t+op.dt)

        trial, test = TrialFunction(P1), TestFunction(P1)
        a = trial*test*dx - theta*dt*dot(u_new, grad(trial))*test*dx
        L = monitor_old*test*dx + (1-theta)*dt*dot(u, grad(monitor_old))*test*dx

        monitor_new = Function(P1)
        params = {
            'mat_type': 'aij',
            'ksp_type': 'preonly',
            'pc_type': 'lu',
            'pc_type_factor_mat_solver_type': 'mumps',
        }
        solve(a == L, monitor_new, solver_parameters=params)
        # return 0.5*(monitor_new + monitor_old)
        return monitor_new
    return wrapper


tp.set_monitor_functions(advected_monitor(static_hessian_monitor))
tp.solve_forward()
