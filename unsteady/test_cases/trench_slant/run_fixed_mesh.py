from adapt_utils.unsteady.test_cases.trench_slant.options import TrenchSlantOptions
from adapt_utils.unsteady.solver import AdaptiveProblem
from thetis import *

import numpy as np

from firedrake.petsc import PETSc
import pandas as pd
import time
import datetime
from adapt_utils.io import initialise_bathymetry, export_bathymetry

import firedrake as fire

ts = time.time()
st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
outputdir = 'outputs' + st

nx = 1.6
ny = 1.6

inputdir = 'hydrodynamics_trench_slant_' + str(nx)

kwargs = {
    'approach': 'fixed_mesh',
    'nx': nx,
    'ny': ny,
    'plot_pvd': True,
    'input_dir': inputdir,
    'output_dir': outputdir,
    # Spatial discretisation
    'family': 'dg-dg',
    'stabilisation': None,
    'use_automatic_sipg_parameter': True,
}

op = TrenchSlantOptions(**kwargs)
swp = AdaptiveProblem(op)

t1 = time.time()
swp.solve_forward()
t2 = time.time()

new_mesh = RectangleMesh(16*5*4, 5*4, 16, 1.1)

bath = Function(FunctionSpace(new_mesh, "CG", 1)).project(swp.fwd_solutions_bathymetry[0])

export_bathymetry(bath, "hydrodynamics_trench_slant_bath_new_"+str(nx))

print("total time: ")
print(t2-t1)
print(nx)
bath_real = initialise_bathymetry(new_mesh, 'hydrodynamics_trench_slant_bath_new_4.0')


print('L2')
print(fire.errornorm(bath, bath_real))
