from thetis import *
import firedrake as fire

import datetime
import numpy as np
import os
import pandas as pd
import time

from adapt_utils.io import initialise_bathymetry, export_bathymetry
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
print(inputdir)
kwargs = {
    'approach': 'fixed_mesh',
    'nx': nx,
    'ny': ny,

    # Spatial discretisation
    'family': 'dg-dg',
    'stabilisation': None,
    'use_automatic_sipg_parameter': True,
    'friction': 'manning',

    # I/O
    'plot_bathymetry': True,
    'plot_pvd': True,
    'input_dir': inputdir,
    'output_dir': outputdir,

    # Debugging
    'debug': True,
}

op = BeachOptions(**kwargs)
if os.getenv('REGRESSION_TEST') is not None:
    op.end_time = op.dt*op.dt_per_export
swp = AdaptiveProblem(op)

t1 = time.time()
swp.solve_forward()
t2 = time.time()

print(t2-t1)

new_mesh = RectangleMesh(880, 20, 220, 10)

bath = Function(FunctionSpace(new_mesh, "CG", 1)).project(swp.fwd_solutions_bathymetry[0])

# fpath = "hydrodynamics_beach_bath_new_{:d}".format(int(nx*220))
# export_bathymetry(bath, fpath, op=op)

xaxisthetis1 = []
baththetis1 = []

for i in np.linspace(0, 219, 220):
    xaxisthetis1.append(i)
    baththetis1.append(-bath.at([i, 5]))
df = pd.concat([pd.DataFrame(xaxisthetis1, columns = ['x']), pd.DataFrame(baththetis1, columns = ['bath'])], axis = 1)
df.to_csv("final_result_check_nx" + str(nx) + "_ny" + str(ny) + ".csv", index = False)

bath_real = initialise_bathymetry(new_mesh, 'hydrodynamics_beach_bath_new_880')

print('L2')
print(fire.errornorm(bath, bath_real))
