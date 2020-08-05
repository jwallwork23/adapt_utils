from thetis import *
from firedrake.petsc import PETSc
import firedrake as fire

import pylab as plt
import pandas as pd
import numpy as np
import time
import datetime

from adapt_utils.unsteady.test_cases.beach_hydro.options import BeachOptions
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

    try:
        plex = bathymetry_2d.function_space().mesh()._plex
    except AttributeError:
        plex = bathymetry_2d.function_space().mesh()._topology_dm
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

alpha = 1
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

    # Mesh movement
    'nonlinear_method': 'relaxation',
    'r_adapt_rtol': 1.0e-3,

    # Spatial discretisation
    'family': 'dg-dg',
    'stabilisation': None,
    'use_automatic_sipg_parameter': True,
    'friction': 'manning'

    # I/O
    'plot_pvd': True,
    'input_dir': inputdir,
    'output_dir': outputdir,
    'plot_bathymetry': True
}

op = BeachOptions(**kwargs)
assert op.num_meshes == 1
swp = AdaptiveProblem(op)
swp.shallow_water_options[0]['mesh_velocity'] = None

def velocity_monitor(mesh, alpha=alpha, beta=beta, gamma=gamma, K = kappa):
    P1 = FunctionSpace(mesh, "CG", 1)
    b = swp.fwd_solutions_bathymetry[0]

    if b is not None:
    	abs_hor_vel_norm = Function(b.function_space()).project(conditional(b > 0.0, Constant(1.0), Constant(0.0)))
    else:
        abs_hor_vel_norm = Function(swp.bathymetry[0].function_space()).project(conditional(swp.bathymetry[0] > 0.0, Constant(2.0), Constant(0.0)))
    comp_new = project(abs_hor_vel_norm, P1)
    mon_init = project(1.0 + alpha * comp_new, P1)
    return mon_init

swp.set_monitor_functions(velocity_monitor)

t1 = time.time()
swp.solve_forward()
t2 = time.time()

print(t2-t1)

#new_mesh = RectangleMesh(880, 20, 220, 10)

#bath = Function(FunctionSpace(new_mesh, "CG", 1)).project(swp.fwd_solutions_bathymetry[0])

export_final_state("hydrodynamics_beach_bath_new_"+str(int(nx*220))+"_test", swp.fwd_solutions_bathymetry[0])

"""
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
"""
