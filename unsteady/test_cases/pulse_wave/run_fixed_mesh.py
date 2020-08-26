from thetis import *
import firedrake as fire
from firedrake.petsc import PETSc

import datetime
import pandas as pd
import numpy as np
import sys
import time

from adapt_utils.unsteady.solver import AdaptiveProblem
from adapt_utils.unsteady.test_cases.pulse_wave.options import BeachOptions


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
        plex = bathymetry_2d.function_space().mesh()._topology_dm
    except AttributeError:
        plex = bathymetry_2d.function_space().mesh()._plex  # Backwards compatability
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
st = datetime.datetime.fromtimestamp(t1).strftime('%Y-%m-%d %H:%M:%S')
outputdir = 'outputs' + st

inputdir = 'hydrodynamics_beach_l_sep_nx_' + str(int(nx*220))

kwargs = {
    'approach': 'fixed_mesh',

    # Spatial discretisation
    'nx': nx,
    'ny': ny,
    'family': 'dg-dg',
    'stabilisation': None,
    'use_automatic_sipg_parameter': True,
    'friction': 'manning',

    # I/O
    'plot_pvd': True,
    'input_dir': inputdir,
    'output_dir': outputdir,
}
if os.getenv('REGRESSION_TEST') is not None:
    kwargs['num_hours'] = 1/120
    kwargs['dt_per_export'] = 5
    kwargs['plot_pvd'] = False

op = BeachOptions(**kwargs)
swp = AdaptiveProblem(op)

t1 = time.time()
swp.solve_forward()
t2 = time.time()
if os.getenv('REGRESSION_TEST') is not None:
    sys.exit(0)

print(t2-t1)

new_mesh = RectangleMesh(880, 20, 220, 10)

bath = Function(FunctionSpace(new_mesh, "CG", 1)).project(swp.fwd_solutions_bathymetry[0])

export_final_state("hydrodynamics_beach_bath_new_"+str(int(nx*220)), bath)

xaxisthetis1 = []
baththetis1 = []

for i in np.linspace(0, 219, 220):
    xaxisthetis1.append(i)
    baththetis1.append(-bath.at([i, 5]))
df = pd.concat([pd.DataFrame(xaxisthetis1, columns=['x']), pd.DataFrame(baththetis1, columns=['bath'])], axis=1)
df.to_csv("final_result_check_nx" + str(nx) + "_ny" + str(ny) + ".csv", index=False)

bath_real = initialise_fields(new_mesh, 'hydrodynamics_beach_bath_new_880')

print('L2')
print(fire.errornorm(bath, bath_real))
