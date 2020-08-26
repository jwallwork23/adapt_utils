from thetis import *
from firedrake.petsc import PETSc
import firedrake as fire

import datetime
import numpy as np
import os
import pandas as pd
import time

from adapt_utils.adapt import recovery
from adapt_utils.io import initialise_fields, export_final_state
from adapt_utils.norms import local_frobenius_norm, local_norm
from adapt_utils.unsteady.solver import AdaptiveProblem
from adapt_utils.unsteady.test_cases.beach_wall.options import BeachOptions


nx = 1
ny = 1

alpha = 1
beta = 1
gamma = 1

kappa = 100 # 12.5

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

    if b is not None:
    	abs_hor_vel_norm = Function(b.function_space()).project(conditional(b > 0.0, Constant(1.0), Constant(0.0)))
    else:
        abs_hor_vel_norm = Function(swp.bathymetry[0].function_space()).project(conditional(swp.bathymetry[0] > 0.0, Constant(1.0), Constant(0.0)))
    comp_new = project(abs_hor_vel_norm, P1)
    mon_init = project(1.0 + alpha * comp_new, P1)
    return mon_init

swp.set_monitor_functions(velocity_monitor)

t1 = time.time()
swp.solve_forward()
t2 = time.time()

print(t2-t1)

# new_mesh = RectangleMesh(880, 20, 220, 10)

# bath = Function(FunctionSpace(new_mesh, "CG", 1)).project(swp.fwd_solutions_bathymetry[0])

export_final_state("hydrodynamics_beach_bath_new_"+str(int(nx*220))+"_basic", swp.fwd_solutions_bathymetry[0])


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
