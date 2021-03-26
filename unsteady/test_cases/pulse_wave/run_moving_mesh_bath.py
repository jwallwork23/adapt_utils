import firedrake as fire
from thetis import *

import datetime
import pandas as pd
import os
import time

from adapt_utils.adapt import recovery
from adapt_utils.io import initialise_bathymetry, export_bathymetry
from adapt_utils.norms import local_frobenius_norm, local_norm
from adapt_utils.unsteady.solver import AdaptiveProblem
from adapt_utils.unsteady.test_cases.beach_pulse_wave.options import BeachOptions

ts = time.time()
st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
outputdir = 'outputs' + st

nx = 0.5
ny = 0.5

alpha = 2
beta = 0
gamma = 1

kappa = 20

ts = time.time()
st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
outputdir = 'outputs' + st

kwargs = {
    'approach': 'monge_ampere',
    'nx': nx,
    'ny': ny,
    'plot_pvd': True,
    'output_dir': outputdir,
    'nonlinear_method': 'relaxation',
    'r_adapt_rtol': 1.0e-3,
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
    bath_gradient = recovery.recover_gradient(b)
    bath_hess = recovery.recover_hessian(b, op=op)
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

print(nx)
print(alpha)
print(beta)
print(gamma)

new_mesh = RectangleMesh(880, 20, 220, 10)

bath = Function(FunctionSpace(new_mesh, "CG", 1)).project(swp.fwd_solutions_bathymetry[0])

fpath = "hydrodynamics_beach_bath_new_{:d}_{:d}_{:d}_{:d}".format(int(nx*220), alpha, beta, gamma)
export_bathymetry(bath, os.path.join("adapt_output", fpath), op=op)

xaxisthetis1 = []
baththetis1 = []

for i in np.linspace(0, 219, 220):
    xaxisthetis1.append(i)
    baththetis1.append(-bath.at([i, 5]))
df = pd.concat([pd.DataFrame(xaxisthetis1, columns=['x']), pd.DataFrame(baththetis1, columns=['bath'])], axis=1)
df.to_csv("final_result_nx" + str(nx) + "_" + str(alpha) + '_' + str(beta) + '_' + str(gamma) + ".csv", index=False)

df_real = pd.read_csv('final_result_nx3_ny1.csv')
print("Mesh error: ")
print(sum([(df['bath'][i] - df_real['bath'][i])**2 for i in range(len(df_real))]))

bath_real = initialise_bathymetry(new_mesh, 'hydrodynamics_beach_bath_new_660')

print('L2')
print(fire.errornorm(bath, bath_real))
print(kappa)
print('diff y')
