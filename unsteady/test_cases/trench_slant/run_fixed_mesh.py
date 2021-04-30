"""
Migrating Trench 2D Test case
=======================

Solves the hydro-morphodynamic simulation of a 2D migrating trench on a fixed mesh

"""

from thetis import *
import firedrake as fire

import datetime
import os
import sys
import time

from adapt_utils.io import initialise_bathymetry, export_bathymetry
from adapt_utils.unsteady.test_cases.trench_slant.options import TrenchSlantOptions
from adapt_utils.unsteady.solver import AdaptiveProblem


ts = time.time()
st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
di = os.path.dirname(__file__)
outputdir = os.path.join(di, 'outputs' + st)

fac_x = 0.5
fac_y = 0.5

# to create the input hydrodynamics directiory please run hydro_trench_slant.py
# setting fac_x and fac_y to be the same values as above

# We have included the hydrodynamics input dir for fac_x = 0.5 and fac_y = 0.5 as an example.

# Note for fac_x=fac_y=1 and fac_x=fac_y=1.6 self.dt should be changed to 0.125
# and self.dt_per_mesh_movement should be changed to 80 in options.py

inputdir = os.path.join(di, 'hydrodynamics_trench_slant_' + str(fac_x))

kwargs = {
    'approach': 'fixed_mesh',
    'nx': fac_x,
    'ny': fac_y,
    'plot_pvd': True,
    'input_dir': inputdir,
    'output_dir': outputdir,
    # Spatial discretisation
    'family': 'dg-dg',
    'stabilisation': None,
    'stabilisation_sediment': None,
}

op = TrenchSlantOptions(**kwargs)
if os.getenv('REGRESSION_TEST') is not None:
    op.dt_per_export = 20
    op.end_time = op.dt*op.dt_per_export
swp = AdaptiveProblem(op)

t1 = time.time()
swp.solve_forward()
t2 = time.time()
if os.getenv('REGRESSION_TEST') is not None:
    sys.exit(0)

new_mesh = RectangleMesh(16*5*4, 5*4, 16, 1.1)

bath = Function(FunctionSpace(new_mesh, "CG", 1)).project(swp.fwd_solutions_bathymetry[0])

fpath = "hydrodynamics_trench_slant_bath_new_" + str(fac_x)
export_bathymetry(bath, fpath, op=op)

print("total time: ")
print(t2-t1)
print(fac_x)
bath_real = initialise_bathymetry(new_mesh, 'hydrodynamics_trench_slant_bath_new_4.0')


print('L2')
print(fire.errornorm(bath, bath_real))
