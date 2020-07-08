from thetis import *

import numpy as np

from adapt_utils.unsteady.test_cases.trench_tracer.options import TrenchTracerOptions
from adapt_utils.unsteady.solver import AdaptiveProblem
from adapt_utils.adapt import recovery
from adapt_utils.norms import local_frobenius_norm

kwargs = {
    'approach': 'monge_ampere',
    'nx': 1,
    'ny': 1,
    'plot_pvd': True,
    'debug': True,

    'nonlinear_method': 'relaxation',
    'r_adapt_rtol': 1.0e-2,

    'family': 'dg-dg',
    'stabilisation': 'lax_friedrichs',
    'use_automatic_sipg_parameter': True,
}

op = TrenchTracerOptions(**kwargs)
assert op.num_meshes == 1
swp = AdaptiveProblem(op)
# swp.shallow_water_options[0]['mesh_velocity'] = swp.mesh_velocities[0]
swp.shallow_water_options[0]['mesh_velocity'] = None


def gradient_interface_monitor(mesh, alpha=100.0):

    """
    Monitor function focused around the steep_gradient (budd acta numerica)

    NOTE: Defined on the *computational* mesh.

    """
    P1 = FunctionSpace(mesh, "CG", 1)

    # eta = swp.solution.split()[1]
    b = swp.bathymetry[0]
    bath_hess = recovery.construct_hessian(b, op = swp.op)
    frob_bath_hess = Function(b.function_space()).project(local_frobenius_norm(bath_hess))

    norm_two_proj = project(frob_bath_hess, P1)

    H = Function(P1)
    tau = TestFunction(P1)
    n = FacetNormal(mesh)

    mon_init = project(sqrt(1.0 + alpha * norm_two_proj), P1)

    K = 10*(0.4**2)/4
    a = (inner(tau, H)*dx)+(K*inner(grad(tau), grad(H))*dx) - (K*(tau*inner(grad(H), n)))*ds
    a -= inner(tau, mon_init)*dx
    solve(a == 0, H)

    return H


swp.set_monitor_functions(gradient_interface_monitor)
swp.solve_forward()
