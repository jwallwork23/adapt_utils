from thetis import *
from firedrake.petsc import PETSc
import firedrake as fire

import pylab as plt
import pandas as pd
import numpy as np
import time
import datetime

from adapt_utils.unsteady.test_cases.beach_bedload_uniform.options import BeachOptions
from adapt_utils.unsteady.solver import AdaptiveProblem
from adapt_utils.norms import local_frobenius_norm, local_norm
from adapt_utils.adapt import recovery

def export_final_state(inputdir, bathymetry_2d):
    """
    Export fields to be used in a subsequent simulation
    """
    if not os.path.exists(inputdir):
        os.makedirs(inputdir)
    print_output("Exporting fields for subsequent simulation")

    chk = DumbCheckpoint(inputdir + "/bathymetry", mode=FILE_CREATE)
    chk.store(bathymetry_2d, name="bathymetry")
    File(inputdir + '/bathout.pvd').write(bathymetry_2d)
    chk.close()

    plex = bathymetry_2d.function_space().mesh()._plex
    viewer = PETSc.Viewer().createHDF5(inputdir + '/myplex.h5', 'w')
    viewer(plex)

def initialise_fields(mesh2d, inputdir):
    """
    Initialise simulation with results from a previous simulation
    """
    V = FunctionSpace(mesh2d, 'CG', 1)
    # elevation
    with timed_stage('initialising bathymetry'):
        chk = DumbCheckpoint(inputdir + "/bathymetry", mode=FILE_READ)
        bath = Function(V, name="bathymetry")
        chk.load(bath)
        chk.close()

    return bath

nx = 0.5
ny = 0.5

alpha = 5
beta = 1
gamma = 1

kappa = 100 #12.5

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
swp.shallow_water_options[0]['mesh_velocity'] = None

def velocity_monitor(mesh, alpha=alpha, beta=beta, gamma=gamma, K = kappa):
    P1 = FunctionSpace(mesh, "CG", 1)
    b = swp.fwd_solutions_bathymetry[0]
    uv, elev = swp.fwd_solutions[0].split()
    horizontal_velocity = Function(elev.function_space()).project(uv[0])
    abs_horizontal_velocity = Function(elev.function_space()).project(abs(uv[0]))
    abs_hor_vel_norm = Function(elev.function_space()).project(conditional(b > -0.1, abs_horizontal_velocity - np.mean(abs_horizontal_velocity.dat.data[:]), Constant(0.0)))

    uv_gradient = recovery.construct_gradient(horizontal_velocity)
    frob_uv_hess = Function(elev.function_space()).project(local_norm(uv_gradient))

    if max(abs(frob_uv_hess.dat.data[:])) < 1e-10:
        div_uv_star = Function(elev.function_space()).project(frob_uv_hess)
    else:
        div_uv_star = Function(elev.function_space()).project(frob_uv_hess/max(frob_uv_hess.dat.data[:]))

    if max(abs_horizontal_velocity.dat.data[:])<1e-10:
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

export_final_state("adapt_output/hydrodynamics_beach_bath_new_"+str(int(nx*220))+"_" + str(alpha) +'_' + str(beta) + '_' + str(gamma), bath)


xaxisthetis1 = []
baththetis1 = []

for i in np.linspace(0, 219, 220):
    xaxisthetis1.append(i)
    baththetis1.append(-bath.at([i, 5]))
df = pd.concat([pd.DataFrame(xaxisthetis1, columns = ['x']), pd.DataFrame(baththetis1, columns = ['bath'])], axis = 1)
df.to_csv("final_result_nx" + str(nx) +"_" + str(alpha) +'_' + str(beta) + '_' + str(gamma) + ".csv", index = False)

bath_real = initialise_fields(new_mesh, 'hydrodynamics_beach_bath_new_440')

print('L2')
print(fire.errornorm(bath, bath_real))

df_real = pd.read_csv('final_result_nx2_ny1.csv')
print("Mesh error: ")
print(sum([(df['bath'][i] - df_real['bath'][i])**2 for i in range(len(df_real))]))

print(alpha)
print(beta)
print(gamma)
