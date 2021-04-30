"""
Beach Profile Test case
=======================

Solves the hydro-morphodynamic simulation of a beach profile on a fixed uniform mesh

"""

from thetis import *
import firedrake as fire

import datetime
import os
import sys
import time

from adapt_utils.io import initialise_bathymetry, export_bathymetry
from adapt_utils.unsteady.solver import AdaptiveProblem
from adapt_utils.unsteady.test_cases.beach_slope.options import BeachOptions

t1 = time.time()

fac_x = 0.5
fac_y = 1

ts = time.time()
st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
di = os.path.dirname(__file__)
outputdir = os.path.join(di, 'outputs' + st)

# to create the input hydrodynamics directiory please run beach_tidal_hydro.py
# setting fac_x and fac_y to be the same values as above

# we have included the hydrodynamics input dir for fac_x = 0.5 and fac_y = 1 as an example

inputdir = os.path.join(di, 'hydrodynamics_beach_l_sep_nx_' + str(int(fac_x*220)) + '_' + str(int(fac_y*10)))
print(inputdir)
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
    'friction': 'manning'
}

op = BeachOptions(**kwargs)
if os.getenv('REGRESSION_TEST') is not None:
    op.dt_per_export = 18
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

fpath = "hydrodynamics_beach_bath_fixed_{:d}_{:d}".format(int(fac_x*220), int(fac_y*10))
export_bathymetry(bath, os.path.join("fixed_output", fpath), op=op)

bath_real = initialise_bathymetry(new_mesh, os.path.join(di, 'fixed_output/hydrodynamics_beach_bath_fixed_440_10'))

print('whole domain error')
print(fire.errornorm(bath, bath_real))

V = FunctionSpace(new_mesh, 'CG', 1)

x, y = SpatialCoordinate(new_mesh)

bath_mod = Function(V).interpolate(conditional(x > 70, bath, Constant(0.0)))
bath_real_mod = Function(V).interpolate(conditional(x > 70, bath_real, Constant(0.0)))

print('subdomain error')

print(fire.errornorm(bath_mod, bath_real_mod))
