from firedrake.petsc import PETSc
import firedrake as fire
from thetis import *

import numpy as np

from adapt_utils.unsteady.test_cases.beach_suspended_uniform.options import BeachOptions
from adapt_utils.unsteady.solver import AdaptiveProblem
from adapt_utils.adapt import recovery
from adapt_utils.norms import local_frobenius_norm, local_norm

import pandas as pd
import time
import datetime

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

ts = time.time()
st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
outputdir = 'outputs' + st

nx = 0.25
ny = 0.5

alpha = 2
beta = 1
gamma = 0.5

kappa = 50

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
    bath_grad2 = Function(bath_gradient.function_space()).project(bath_gradient+as_vector((0.025, 0.0)))
    l2_bath_grad = Function(b.function_space()).project(abs(local_norm(bath_gradient)))

    bath_dx_l2_norm = Function(b.function_space()).interpolate(l2_bath_grad/max(l2_bath_grad.dat.data[:]))
    #comp = interpolate(alpha*bath_dx_l2_norm, b.function_space())
    comp = interpolate(conditional(alpha*beta*bath_dx_l2_norm > alpha*gamma*frob_bath_norm, alpha*beta*bath_dx_l2_norm, alpha*gamma*frob_bath_norm), b.function_space())
    comp_new = project(comp, P1)
    comp_new2 = interpolate(conditional(comp_new > Constant(0.0), comp_new, Constant(0.0)), P1)
    mon_init = project(Constant(1.0) + comp_new2, P1)

    #H = Function(P1)
    #tau = TestFunction(P1)

    #a = (inner(tau, H)*dx)+(K*inner(tau.dx(1), H.dx(1))*dx) - inner(tau, mon_init)*dx
    #solve(a == 0, H)

    H = Function(P1)
    tau = TestFunction(P1)

    n = FacetNormal(mesh)

    a = (inner(tau, H)*dx)+(K*inner(grad(tau), grad(H))*dx) - (K*(tau*inner(grad(H), n)))*ds
    a -= inner(tau, mon_init)*dx
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

export_final_state("adapt_output/hydrodynamics_beach_bath_new_"+str(int(nx*220))+"_" + str(alpha) +'_' + str(beta) + '_' + str(gamma), bath)


xaxisthetis1 = []
baththetis1 = []

for i in np.linspace(0, 219, 220):
    xaxisthetis1.append(i)
    baththetis1.append(-bath.at([i, 5]))
df = pd.concat([pd.DataFrame(xaxisthetis1, columns = ['x']), pd.DataFrame(baththetis1, columns = ['bath'])], axis = 1)
df.to_csv("final_result_nx" + str(nx) +"_" + str(alpha) +'_' + str(beta) + '_' + str(gamma) + ".csv", index = False)

df_real = pd.read_csv('final_result_nx3_ny1.csv')
print("Mesh error: ")
print(sum([(df['bath'][i] - df_real['bath'][i])**2 for i in range(len(df_real))]))

bath_real = initialise_fields(new_mesh, 'hydrodynamics_beach_bath_new_660')

print('L2')
print(fire.errornorm(bath, bath_real))
print(kappa)