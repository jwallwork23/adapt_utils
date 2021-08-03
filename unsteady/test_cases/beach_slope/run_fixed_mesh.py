"""
Beach Profile Test case
=======================

Solves the hydro-morphodynamic simulation of a beach profile on a fixed uniform mesh

"""

from thetis import *
import firedrake as fire
from firedrake.petsc import PETSc

import pylab as plt
import pandas as pd
import numpy as np
import time
import datetime

from adapt_utils.unsteady.test_cases.beach_slope.options import BeachOptions
from adapt_utils.unsteady.solver import AdaptiveProblem

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

# number of mesh elements
fac_x = 0.2
fac_y = 0.5

# set output directory name
ts = time.time()
st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
di = os.path.dirname(__file__)
outputdir = os.path.join(di, 'outputs' + st)

kwargs = {
    'approach': 'fixed_mesh',
    'nx': fac_x,
    'ny': fac_y,
    'plot_pvd': True,
    'output_dir': outputdir,
    # Spatial discretisation
    'family': 'dg-dg',
    'stabilisation': None,
    'stabilisation_sediment': None,
    'use_automatic_sipg_parameter': True,
    'friction': 'manning'
}

op = BeachOptions(**kwargs)
if os.getenv('REGRESSION_TEST') is not None:
    op.dt_per_export = 18
    op.end_time = op.dt*op.dt_per_export
swp = AdaptiveProblem(op)

# run model
t1 = time.time()
swp.solve_forward()
t2 = time.time()
if os.getenv('REGRESSION_TEST') is not None:
    sys.exit(0)

print(t2-t1)

print(fac_x)
print(fac_y)

# export full bathymetry
new_mesh = RectangleMesh(1400, 20, 350, 10)

bath = Function(FunctionSpace(new_mesh, "CG", 1)).project(swp.fwd_solutions_bathymetry[0])

export_final_state("fixed_output/hydrodynamics_beach_bath_fixed_"+str(int(fac_x*350)) + '_' + str(fac_y), bath)

bath_real = initialise_fields(new_mesh, 'fixed_output/hydrodynamics_beach_bath_fixed_700_2')

print(fire.errornorm(bath, bath_real))

# export bathymetry along central y-axis
xaxisthetis1 = []
baththetis1 = []

for i in np.linspace(0, 349, 350):
    xaxisthetis1.append(i)
    baththetis1.append(-bath.at([i, 5]))
df = pd.concat([pd.DataFrame(xaxisthetis1, columns = ['x']), pd.DataFrame(baththetis1, columns = ['bath'])], axis = 1)
df.to_csv("fixed_output/final_result_check_nx" + str(fac_x) + "_ny" + str(fac_y) + ".csv", index = False)

df_real = pd.read_csv('fixed_output/final_result_check_nx2_ny2.csv')

error = sum([(df['bath'][i] - df_real['bath'][i])**2 for i in range(len(df))])

print(error)
