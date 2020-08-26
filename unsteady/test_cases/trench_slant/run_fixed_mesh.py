from thetis import *
import firedrake as fire
from firedrake.petsc import PETSc

import datetime
import numpy as np
import pandas as pd
import time

from adapt_utils.io import initialise_fields, export_final_state
from adapt_utils.unsteady.test_cases.trench_slant.options import TrenchSlantOptions
from adapt_utils.unsteady.solver import AdaptiveProblem


ts = time.time()
st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
outputdir = 'outputs' + st

nx = 2.0
ny = 2.0

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

export_final_state("hydrodynamics_trench_slant_bath_new_"+str(nx), bath)

print("total time: ")
print(t2-t1)
print(nx)
bath_real = initialise_fields(new_mesh, 'hydrodynamics_trench_slant_bath_new_4.0')


print('L2')
print(fire.errornorm(bath, bath_real))
