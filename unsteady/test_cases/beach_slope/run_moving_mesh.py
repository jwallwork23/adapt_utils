"""
Beach Profile Test case
=======================

Solves the hydro-morphodynamic simulation of a beach profile using moving mesh methods

"""

import firedrake as fire
from thetis import *

import datetime
import os
import time

from adapt_utils.adapt import recovery
from adapt_utils.io import initialise_bathymetry, export_bathymetry
from adapt_utils.norms import local_frobenius_norm, local_norm
from adapt_utils.unsteady.solver import AdaptiveProblem
from adapt_utils.unsteady.test_cases.beach_slope.options import BeachOptions

fac_x = 0.2
fac_y = 0.5

dt_exp = 72

alpha = 5
beta = 0
gamma = 1

kappa = 200

ts = time.time()
st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
di = os.path.dirname(__file__)
outputdir = os.path.join(di, 'outputs' + st)

# to create the input hydrodynamics directiory please run beach_tidal_hydro.py
# setting fac_x and fac_y to be the same values as above

# we have included the hydrodynamics input dir for fac_x = 0.2 and fac_y = 0.5 as an example

# Note to recreate subdomain errors in options.py self.dt_per_mesh_movement = 72 and for whole
# domain errors self.dt_per_mesh_movement = 648

inputdir = os.path.join(di, 'hydrodynamics_beach_l_sep_nx_' + str(int(fac_x*220)) + '_' + str(int(fac_y*10)))

tol_value = 1e-3

kwargs = {
    'approach': 'monge_ampere',
    'dt_exp': dt_exp,
    'nx': fac_x,
    'ny': fac_y,
    'plot_pvd': True,
    'input_dir': inputdir,
    'output_dir': outputdir,
    'nonlinear_method': 'relaxation',
    'r_adapt_rtol': tol_value,
    # Spatial discretisation
    'family': 'dg-dg',
    'stabilisation': None,
    'stabilisation_sediment': None,
    'use_automatic_sipg_parameter': True,
    'friction': 'manning'
}

op = BeachOptions(**kwargs)
assert op.num_meshes == 1
swp = AdaptiveProblem(op)


def gradient_interface_monitor(mesh, alpha=alpha, beta=beta, gamma=gamma, K=kappa):

    """
    Monitor function focused around the steep_gradient (budd acta numerica)

    NOTE: Defined on the *computational* mesh.

    """
    P1 = FunctionSpace(mesh, "CG", 1)

    b = swp.fwd_solutions_bathymetry[0]
    bath_gradient = recovery.construct_gradient(b)
    bath_hess = recovery.construct_hessian(b, op=op)
    frob_bath_hess = Function(b.function_space()).project(local_frobenius_norm(bath_hess))

    if max(abs(frob_bath_hess.dat.data[:])) < 1e-10:
        frob_bath_norm = Function(b.function_space()).project(frob_bath_hess)
    else:
        frob_bath_norm = Function(b.function_space()).project(frob_bath_hess/max(frob_bath_hess.dat.data[:]))

    l2_bath_grad = Function(b.function_space()).project(abs(local_norm(bath_gradient)))

    bath_dx_l2_norm = Function(b.function_space()).interpolate(l2_bath_grad/max(l2_bath_grad.dat.data[:]))
    comp = interpolate(conditional(alpha*beta*bath_dx_l2_norm > alpha*gamma*frob_bath_norm, alpha*beta*bath_dx_l2_norm, alpha*gamma*frob_bath_norm), b.function_space())
    comp_new = project(comp, P1)
    comp_new2 = interpolate(conditional(comp_new > Constant(0.0), comp_new, Constant(0.0)), P1)
    mon_init = project(Constant(1.0) + comp_new2, P1)

    H = Function(P1)
    tau = TestFunction(P1)

    a = (inner(tau, H)*dx)+(K*inner(tau.dx(1), H.dx(1))*dx) - inner(tau, mon_init)*dx
    solve(a == 0, H)

    return H


swp.set_monitor_functions(gradient_interface_monitor)

t1 = time.time()
swp.solve_forward()
t2 = time.time()

print(t2-t1)

new_mesh = RectangleMesh(880, 20, 220, 10)

bath = Function(FunctionSpace(new_mesh, "CG", 1)).project(swp.fwd_solutions_bathymetry[0])

# export final bathymetry to readable format
fpath = "hydrodynamics_beach_bath_mov_{:d}_{:d}_{:d}_{:d}_{:d}"
fpath = fpath.format(op.dt_per_export, int(fac_x*220), alpha, beta, gamma)
export_bathymetry(bath, os.path.join("adapt_output", fpath), op=op)

bath_real = initialise_bathymetry(new_mesh, 'fixed_output/hydrodynamics_beach_bath_fixed_440_10')

print('L2')
print(fire.errornorm(bath, bath_real))

V = FunctionSpace(new_mesh, 'CG', 1)

x, y = SpatialCoordinate(new_mesh)

bath_mod = Function(V).interpolate(conditional(x > 70, bath, Constant(0.0)))
bath_real_mod = Function(V).interpolate(conditional(x > 70, bath_real, Constant(0.0)))

print('subdomain')

print(fire.errornorm(bath_mod, bath_real_mod))