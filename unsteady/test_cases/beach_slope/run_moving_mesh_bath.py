from firedrake.petsc import PETSc
import firedrake as fire
from thetis import *

import datetime
import numpy as np
import pandas as pd
import time

from adapt_utils.adapt import recovery
from adapt_utils.io import initialise_fields, export_final_state
from adapt_utils.norms import local_frobenius_norm, local_norm
from adapt_utils.unsteady.solver import AdaptiveProblem
from adapt_utils.unsteady.test_cases.beach_slope.options import BeachOptions


nx = 0.2
ny = 0.5

alpha = 3
beta = 0
gamma = 1

kappa = 200

ts = time.time()
st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
outputdir = 'outputs' + st

inputdir = 'hydrodynamics_beach_l_sep_nx_' + str(int(nx*220))
print(inputdir)
kwargs = {
    'approach': 'monge_ampere',
    'nx': nx,
    'ny': ny,
    'plot_pvd': True,
    'input_dir': inputdir,
    'output_dir': outputdir,
    'nonlinear_method': 'relaxation',
    'r_adapt_rtol': 1.0e-3,
    # Spatial discretisation
    'family': 'dg-dg',
    'stabilisation': None,
    'use_automatic_sipg_parameter': True,
    'friction': 'manning'
}

op = BeachOptions(**kwargs)
assert op.num_meshes == 1
swp = AdaptiveProblem(op)
# swp.shallow_water_options[0]['mesh_velocity'] = swp.mesh_velocities[0]
swp.shallow_water_options[0]['mesh_velocity'] = None

def gradient_interface_monitor(mesh, alpha=alpha, beta=beta, gamma=gamma, K = kappa):

    """
    Monitor function focused around the steep_gradient (budd acta numerica)

    NOTE: Defined on the *computational* mesh.

    """
    P1 = FunctionSpace(mesh, "CG", 1)

    # eta = swp.solution.split()[1]
    b = swp.fwd_solutions_bathymetry[0]
    bath_gradient = recovery.construct_gradient(b)
    bath_hess = recovery.construct_hessian(b, op=op)
    frob_bath_hess = Function(b.function_space()).project(local_frobenius_norm(bath_hess))

    if max(abs(frob_bath_hess.dat.data[:]))<1e-10:
        frob_bath_norm = Function(b.function_space()).project(frob_bath_hess)
    else:
        frob_bath_norm = Function(b.function_space()).project(frob_bath_hess/max(frob_bath_hess.dat.data[:]))

    current_mesh = b.function_space().mesh()
    bath_grad2 = Function(bath_gradient.function_space()).project(bath_gradient)
    l2_bath_grad = Function(b.function_space()).project(abs(local_norm(bath_gradient)))

    bath_dx_l2_norm = Function(b.function_space()).interpolate(l2_bath_grad/max(l2_bath_grad.dat.data[:]))
    # comp = interpolate(alpha*bath_dx_l2_norm, b.function_space())
    comp = interpolate(conditional(alpha*beta*bath_dx_l2_norm > alpha*gamma*frob_bath_norm, alpha*beta*bath_dx_l2_norm, alpha*gamma*frob_bath_norm), b.function_space())
    comp_new = project(comp, P1)
    comp_new2 = interpolate(conditional(comp_new > Constant(0.0), comp_new, Constant(0.0)), P1)
    mon_init = project(Constant(1.0) + comp_new2, P1)

    H = Function(P1)
    tau = TestFunction(P1)

    a = (inner(tau, H)*dx)+(K*inner(tau.dx(1), H.dx(1))*dx) - inner(tau, mon_init)*dx
    solve(a == 0, H)

    # H = Function(P1)
    # tau = TestFunction(P1)

    # n = FacetNormal(mesh)

    # a = (inner(tau, H)*dx)+(K*inner(grad(tau), grad(H))*dx) - (K*(tau*inner(grad(H), n)))*ds
    # a -= inner(tau, mon_init)*dx
    # solve(a == 0, H)

    return H

swp.set_monitor_functions(gradient_interface_monitor)

t1 = time.time()
swp.solve_forward()
t2 = time.time()

print(t2-t1)

print(nx)
print(alpha)
print(beta)
print(gamma)

new_mesh = RectangleMesh(880, 20, 220, 10)

bath = Function(FunctionSpace(new_mesh, "CG", 1)).project(swp.fwd_solutions_bathymetry[0])

export_final_state("adapt_output/hydrodynamics_beach_bath_mov_"+ str(op.dt_per_export) + "_" + str(int(nx*220))+"_" + str(alpha) +'_' + str(beta) + '_' + str(gamma), bath)

bath_real = initialise_fields(new_mesh, 'fixed_output/hydrodynamics_beach_bath_fixed_440_1')

print('L2')
print(fire.errornorm(bath, bath_real))
print(kappa)

V = FunctionSpace(new_mesh, 'CG', 1)

x,y = SpatialCoordinate(new_mesh)

bath_mod = Function(V).interpolate(conditional(x > 70, bath, Constant(0.0)))
bath_real_mod = Function(V).interpolate(conditional(x > 70, bath_real, Constant(0.0)))

print('subdomain')

print(fire.errornorm(bath_mod, bath_real_mod))
