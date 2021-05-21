"""
Migrating Trench 2D Test case
=======================

Solves the hydro-morphodynamic simulation of a 2D migrating trench using moving mesh methods

"""

from thetis import *
import firedrake as fire

import datetime
import time

from adapt_utils.adapt import recovery
from adapt_utils.io import initialise_bathymetry, export_bathymetry
from adapt_utils.norms import local_frobenius_norm, local_norm
from adapt_utils.unsteady.test_cases.trench_slant.options import TrenchSlantOptions
from adapt_utils.unsteady.solver import AdaptiveProblem

ts = time.time()
st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
outputdir = 'outputs' + st

fac_x = 0.5
fac_y = 0.5
alpha = 10
beta = 1
gamma = 1

# to create the input hydrodynamics directiory please run hydro_trench_slant.py
# setting fac_x and fac_y to be the same values as above

# We have included the hydrodynamics input dir for fac_x = 0.5 and fac_y = 0.5 as an example

# Note for fac_x=fac_y=1 and fac_x=fac_y=1.6 self.dt should be changed to 0.125
# and self.dt_per_mesh_movement should be changed to 80 in options.py

inputdir = 'hydrodynamics_trench_slant_' + str(fac_x)

kwargs = {
    'approach': 'monge_ampere',
    'nx': fac_x,
    'ny': fac_y,
    'plot_pvd': True,
    'input_dir': inputdir,
    'output_dir': outputdir,
    'nonlinear_method': 'relaxation',
    'r_adapt_rtol': 1.0e-3,
    # Spatial discretisation
    'family': 'dg-dg',
    'stabilisation': 'lax_friedrichs',
    'stabilisation_sediment': 'lax_friedrichs',
}


op = TrenchSlantOptions(**kwargs)
assert op.num_meshes == 1
swp = AdaptiveProblem(op)

def gradient_interface_monitor(mesh, alpha=alpha, beta=beta, gamma=gamma):

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
    frob_bath_norm = Function(b.function_space()).project(frob_bath_hess/max(frob_bath_hess.dat.data[:]))
    current_mesh = b.function_space().mesh()
    l2_bath_grad = Function(b.function_space()).project(local_norm(bath_gradient))
    bath_dx_l2_norm = Function(b.function_space()).interpolate(l2_bath_grad/max(l2_bath_grad.dat.data[:]))

    comp = interpolate(conditional(alpha*beta*bath_dx_l2_norm > alpha*gamma*frob_bath_norm, alpha*beta*bath_dx_l2_norm, alpha*gamma*frob_bath_norm), b.function_space())
    comp_new = project(comp, P1)
    comp_new2 = interpolate(conditional(comp_new > Constant(0.0), comp_new, Constant(0.0)), P1)
    mon_init = project(Constant(1.0) + comp_new2, P1)

    return mon_init

swp.set_monitor_functions(gradient_interface_monitor)

t1 = time.time()
swp.solve_forward()
t2 = time.time()

new_mesh = RectangleMesh(16*5*4, 5*4, 16, 1.1)

bath = Function(FunctionSpace(new_mesh, "CG", 1)).project(swp.fwd_solutions_bathymetry[0])

export_bathymetry(bath, "adapt_output/hydrodynamics_trench_slant_bath_"+str(alpha) + "_" + str(beta) + '_' + str(gamma) + '-' + str(nx))

bath_real = initialise_bathymetry(new_mesh, 'hydrodynamics_trench_slant_bath_new_4.0')

print(nx)
print(ny)
print(alpha)
print('L2')
print(fire.errornorm(bath, bath_real))

print("total time: ")
print(t2-t1)

print(beta)
print(gamma)
