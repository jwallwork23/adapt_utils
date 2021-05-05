"""
Migrating Trench Test case
=======================

Testing the adjoint method for the hydro-morphodynamic simulation of a migrating trench using mesh movement methods
"""
from firedrake_adjoint import *
from thetis import *
import firedrake as fire

import pandas as pd
import datetime
import time

from adapt_utils.adapt import recovery
from adapt_utils.io import initialise_bathymetry, export_bathymetry
from adapt_utils.norms import local_frobenius_norm, local_norm
from adapt_utils.unsteady.test_cases.trench_1d_der.options import TrenchSedimentOptions
from adapt_utils.unsteady.solver import AdaptiveProblem

ts = time.time()
st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
outputdir = 'outputs' + st

res = 0.5

fric_coeff = Constant(0.025)

# to create the input hydrodynamics directiory please run hydro_trench_slant.py
# setting fac_x and fac_y to be the same values as above

# --- Set parameters

ts = time.time()
st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
di = os.path.dirname(__file__)
outputdir = os.path.join(di, 'outputs' + st)
inputdir = os.path.join(di, 'hydrodynamics_trench_'+ str(res))
print(inputdir)
kwargs = {
    'approach': 'monge_ampere',
    'nx': res,
    'ny': 1 if res < 4 else 2,
    'fric_coeff': fric_coeff,
    'plot_pvd': True,
    'input_dir': inputdir,
    'output_dir': outputdir,
    'nonlinear_method': 'relaxation',
    'r_adapt_rtol': 1e-3,
    # Spatial discretisation
    'family': 'dg-dg',
    'stabilisation': 'lax_friedrichs',
    'use_automatic_sipg_parameter': True,
}

op = TrenchSedimentOptions(**kwargs)
assert op.num_meshes == 1
swp = AdaptiveProblem(op)

alpha = Constant(1)
beta = Constant(0.5)
gamma = Constant(1)

def gradient_interface_monitor(mesh, alpha=alpha, beta=beta, gamma=gamma, x=None):
    """
    Monitor function focused around the steep_gradient (budd acta numerica)
    NOTE: Defined on the *computational* mesh.
    """

    P1 = FunctionSpace(mesh, "CG", 1)

    # eta = swp.solution.split()[1]
    b = swp.fwd_solutions_bathymetry[0]
    bath_gradient = recovery.recover_gradient(b)
    bath_hess = recovery.recover_hessian(b, op=op)
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

# --- Simulation and analysis

# Solve forward problem

t1 = time.time()
swp.solve_forward()
t2 = time.time()

J = assemble(swp.fwd_solutions_bathymetry[0]*dx)
rf = ReducedFunctional(J, Control(diff_coeff))

print(J)
print(rf(Constant(0.15)))

h2 = Constant(5e-3)
conv_rate = taylor_test(rf, diff_coeff, h2)

