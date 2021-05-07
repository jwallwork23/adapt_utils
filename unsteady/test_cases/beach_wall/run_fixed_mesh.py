from thetis import *
import firedrake as fire

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
print(inputdir)
kwargs = {
    'approach': 'fixed_mesh',
    'nx': nx,
    'ny': ny,

    # Spatial discretisation
    'family': 'dg-dg',
    'stabilisation': None,
    'stabilisation_sediment': None,
    'friction': 'manning',

    # I/O
    'plot_bathymetry': True,
    'plot_pvd': True,
    'input_dir': inputdir,
    'output_dir': outputdir,
}

op = BeachOptions(**kwargs)
if os.getenv('REGRESSION_TEST') is not None:
    op.end_time = op.dt*op.dt_per_export
swp = AdaptiveProblem(op)

t1 = time.time()
swp.solve_forward()
t2 = time.time()
if os.getenv('REGRESSION_TEST') is not None:
    sys.exit(0)

print(t2-t1)

new_mesh = RectangleMesh(880, 20, 220, 10)

bath = Function(FunctionSpace(new_mesh, "CG", 1)).project(swp.fwd_solutions_bathymetry[0])
bath_real = initialise_bathymetry(new_mesh, 'hydrodynamics_beach_bath_new_880')

print('L2')
print(fire.errornorm(bath, bath_real))
