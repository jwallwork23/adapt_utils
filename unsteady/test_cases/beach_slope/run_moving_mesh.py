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
import pandas as pd

from adapt_utils.adapt import recovery
from adapt_utils.io import initialise_bathymetry, export_bathymetry
from adapt_utils.norms import local_frobenius_norm, local_norm
from adapt_utils.unsteady.solver import AdaptiveProblem
from adapt_utils.unsteady.test_cases.beach_slope.options import BeachOptions

# number of mesh elements
fac_x = 0.2
fac_y = 0.5

# mesh movement frequency
dt_exp = 16

# monitor function parameters
alpha = 5
beta = 0
gamma = 1

# set output directory name
ts = time.time()
st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
di = os.path.dirname(__file__)
outputdir = os.path.join(di, 'outputs' + st)

tol_value = 1e-3

kwargs = {
    'approach': 'monge_ampere',
    'dt_exp': dt_exp,
    'nx': fac_x,
    'ny': fac_y,
    'plot_pvd': True,
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


def gradient_interface_monitor(mesh, alpha=alpha, beta=beta, gamma=gamma):

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

    return mon_init

swp.set_monitor_functions(gradient_interface_monitor)

t1 = time.time()
# run model
swp.solve_forward()
t2 = time.time()

print(t2-t1)

print(dt_exp)

print(fac_x)
print(fac_y)
print(alpha)
print(beta)
print(gamma)

# export full bathymetry
new_mesh = RectangleMesh(1400, 20, 350, 10)

bath = Function(FunctionSpace(new_mesh, "CG", 1)).project(swp.fwd_solutions_bathymetry[0])

fpath = "hydrodynamics_beach_bath_mov_{:d}_{:d}_{:d}_{:d}_{:d}"
fpath = fpath.format(op.dt_per_export, int(fac_x*350), alpha, beta, gamma)
export_bathymetry(bath, os.path.join("adapt_output", fpath), op=op)

bath_real = initialise_bathymetry(new_mesh, 'fixed_output/hydrodynamics_beach_bath_fixed_700_2')

print('L2')
print(fire.errornorm(bath, bath_real))

# export bathymetry along central y-axis
xaxisthetis1 = []
baththetis1 = []

for i in np.linspace(0, 349, 350):
    xaxisthetis1.append(i)
    baththetis1.append(-bath.at([i, 5]))
df = pd.concat([pd.DataFrame(xaxisthetis1, columns = ['x']), pd.DataFrame(baththetis1, columns = ['bath'])], axis = 1)
df.to_csv("adapt_output/final_result_check_nx" + str(alpha) + '_' + str(beta) + '_'  + str(gamma) + '_' +  str(fac_x) + "_ny" + str(fac_y) + ".csv", index = False)

df_real = pd.read_csv('fixed_output/final_result_check_nx2_ny2.csv')

error = sum([(df['bath'][i] - df_real['bath'][i])**2 for i in range(len(df))])

print(error)
