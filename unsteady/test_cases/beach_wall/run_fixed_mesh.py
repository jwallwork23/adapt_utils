from thetis import *
import firedrake as fire
<<<<<<< HEAD
from firedrake.petsc import PETSc

import pylab as plt
import pandas as pd
import numpy as np
import time
import datetime

from adapt_utils.unsteady.test_cases.beach_wall.options import BeachOptions
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

t1 = time.time()

nx = 0.5
ny = 0.5

ts = time.time()
st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
outputdir = 'outputs' + st

inputdir = 'hydrodynamics_beach_l_sep_nx_' + str(int(nx*220))
=======

import datetime
import os
import time

from adapt_utils.io import initialise_bathymetry
from adapt_utils.unsteady.solver import AdaptiveProblem
from adapt_utils.unsteady.test_cases.beach_wall.options import BeachOptions


t1 = time.time()

nx = 1
ny = 1

ts = time.time()
st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
di = os.path.dirname(__file__)
outputdir = os.path.join(di, 'outputs' + st)

# to create the input hydrodynamics directiory please run beach_tidal_hydro.py
# setting nx and ny to be the same values as above

# we have included the hydrodynamics input dir for nx = 1 and ny = 1 as an example

inputdir = os.path.join(di, 'hydrodynamics_beach_l_sep_nx_' + str(int(nx*220)))
>>>>>>> dfe1c0b3a34dfef1765835b64b574a69fe60dd9a
print(inputdir)
kwargs = {
    'approach': 'fixed_mesh',
    'nx': nx,
    'ny': ny,

    # Spatial discretisation
    'family': 'dg-dg',
    'stabilisation': None,
<<<<<<< HEAD
=======
    'stabilisation_sediment': None,
>>>>>>> dfe1c0b3a34dfef1765835b64b574a69fe60dd9a
    'use_automatic_sipg_parameter': True,
    'friction': 'manning',

    # I/O
    'plot_bathymetry': True,
    'plot_pvd': True,
    'input_dir': inputdir,
    'output_dir': outputdir,
<<<<<<< HEAD

    # Debugging
    'debug': True,
}

op = BeachOptions(**kwargs)
=======
}

op = BeachOptions(**kwargs)
if os.getenv('REGRESSION_TEST') is not None:
    op.end_time = op.dt*op.dt_per_export
>>>>>>> dfe1c0b3a34dfef1765835b64b574a69fe60dd9a
swp = AdaptiveProblem(op)

t1 = time.time()
swp.solve_forward()
t2 = time.time()
<<<<<<< HEAD
=======
if os.getenv('REGRESSION_TEST') is not None:
    sys.exit(0)
>>>>>>> dfe1c0b3a34dfef1765835b64b574a69fe60dd9a

print(t2-t1)

new_mesh = RectangleMesh(880, 20, 220, 10)

bath = Function(FunctionSpace(new_mesh, "CG", 1)).project(swp.fwd_solutions_bathymetry[0])
<<<<<<< HEAD

#export_final_state("hydrodynamics_beach_bath_new_"+str(int(nx*220)), bath)

xaxisthetis1 = []
baththetis1 = []

for i in np.linspace(0, 219, 220):
    xaxisthetis1.append(i)
    baththetis1.append(-bath.at([i, 5]))
df = pd.concat([pd.DataFrame(xaxisthetis1, columns = ['x']), pd.DataFrame(baththetis1, columns = ['bath'])], axis = 1)
df.to_csv("final_result_check_nx" + str(nx) + "_ny" + str(ny) + ".csv", index = False)

bath_real = initialise_fields(new_mesh, 'hydrodynamics_beach_bath_new_880')
=======
bath_real = initialise_bathymetry(new_mesh, 'hydrodynamics_beach_bath_new_880')
>>>>>>> dfe1c0b3a34dfef1765835b64b574a69fe60dd9a

print('L2')
print(fire.errornorm(bath, bath_real))
