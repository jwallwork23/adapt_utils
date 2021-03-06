from thetis import *
import firedrake as fire

import datetime
import numpy as np
import os
import pandas as pd
import time

from adapt_utils.adapt import recovery
from adapt_utils.io import initialise_bathymetry, export_bathymetry
from adapt_utils.norms import local_norm
from adapt_utils.unsteady.solver import AdaptiveProblem
from adapt_utils.unsteady.test_cases.beach_wall.options import BeachOptions


nx = 1
ny = 1

alpha = 1
beta = 1
gamma = 0

kappa = 100

ts = time.time()
st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
di = os.path.dirname(__file__)
outputdir = os.path.join(di, 'outputs' + st)

# to create the input hydrodynamics directiory please run beach_tidal_hydro.py
# setting nx and ny to be the same values as above

# we have included the hydrodynamics input dir for nx = 1 and ny = 1 as an example

inputdir = os.path.join(di, 'hydrodynamics_beach_l_sep_nx_' + str(int(nx*220)))
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
    'stabilisation_sediment': None,
    'use_automatic_sipg_parameter': True,
    'friction': 'manning'
}

op = BeachOptions(**kwargs)
assert op.num_meshes == 1
swp = AdaptiveProblem(op)


def velocity_monitor(mesh, alpha=alpha, beta=beta, gamma=gamma, K=kappa):
    P1 = FunctionSpace(mesh, "CG", 1)

    uv, elev = swp.fwd_solutions[0].split()
    horizontal_velocity = Function(elev.function_space()).project(uv[0])
    abs_horizontal_velocity = Function(elev.function_space()).project(abs(uv[0]))
    abs_hor_vel_norm = Function(elev.function_space()).project(conditional(abs(elev) > 10**(-5), abs(abs_horizontal_velocity - np.mean(abs_horizontal_velocity.dat.data[:])), Constant(0.0)))

    uv_gradient = recovery.recover_gradient(horizontal_velocity)
    frob_uv_hess = Function(elev.function_space()).project(local_norm(uv_gradient))

    if max(abs(frob_uv_hess.dat.data[:])) < 1e-4:
        div_uv_star = Function(elev.function_space()).project(frob_uv_hess)
    else:
        div_uv_star = Function(elev.function_space()).project(frob_uv_hess/max(frob_uv_hess.dat.data[:]))

    if max(abs_horizontal_velocity.dat.data[:]) < 1e-4:
        abs_uv_star = Function(elev.function_space()).project(abs_hor_vel_norm)
    else:
        abs_uv_star = Function(elev.function_space()).project(abs_hor_vel_norm/max(abs_hor_vel_norm.dat.data[:]))

    comp = interpolate(conditional(beta*abs_uv_star > gamma*div_uv_star, beta*abs_uv_star, gamma*div_uv_star), elev.function_space())
    comp_new = project(comp, P1)
    comp_new2 = interpolate(conditional(comp_new > Constant(0.0), comp_new, Constant(0.0)), P1)
    mon_init = project(1.0 + alpha * comp_new2, P1)

    H = Function(P1)
    tau = TestFunction(P1)

    a = (inner(tau, H)*dx)+(K*inner(tau.dx(1), H.dx(1))*dx) - inner(tau, mon_init)*dx
    solve(a == 0, H)

    return H


swp.set_monitor_functions(velocity_monitor)

t1 = time.time()
swp.solve_forward()
t2 = time.time()

print(t2-t1)

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

bath_real = initialise_bathymetry(new_mesh, 'hydrodynamics_beach_bath_new_440')

print('L2')
print(fire.errornorm(bath, bath_real))

df_real = pd.read_csv('final_result_nx2_ny1.csv')
print("Mesh error: ")
print(sum([(df['bath'][i] - df_real['bath'][i])**2 for i in range(len(df_real))]))

print(alpha)
print(beta)
print(gamma)
