"""
Migrating Trench Test case
=======================

Testing the adjoint method for the hydro-morphodynamic simulation of a migrating trench on a fixed mesh
"""
from firedrake_adjoint import *
from thetis import *

import argparse
import datetime
import os
import pandas as pd
import sys
import time

from adapt_utils.unsteady.solver import AdaptiveProblem
from adapt_utils.unsteady.test_cases.trench_1d_der.options import TrenchSedimentOptions


# To create the input hydrodynamics directiory please run trench_hydro.py
# setting res to be the same values as above.
# We have included the hydrodynamics input dir for res = 0.5 as an example.

# --- Parse arguments

parser = argparse.ArgumentParser()
parser.add_argument("-res", help="Mesh resolution factor (default 0.5).")
args = parser.parse_args()

res = float(args.res or 0.5)
fric_coeff = Constant(0.025)
diff_coeff = Constant(0.15)
# --- Set parameters

ts = time.time()
st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
di = os.path.dirname(__file__)
outputdir = os.path.join(di, 'outputs' + st)
inputdir = os.path.join(di, 'hydrodynamics_trench_' + str(res))
print(inputdir)
kwargs = {
    'approach': 'fixed_mesh',
    'nx': res,
    'ny': 1 if res < 4 else 2,
    'plot_pvd': True,
    'input_dir': inputdir,
    'output_dir': outputdir,
    'fric_coeff': fric_coeff,
    'diff_coeff': diff_coeff,
    # Spatial discretisation
    'family': 'dg-dg',
    'stabilisation': None,
    'stabilisation_sediment': None,
    'use_automatic_sipg_parameter': True,
}
op = TrenchSedimentOptions(**kwargs)
if os.getenv('REGRESSION_TEST') is not None:
    op.end_time = op.dt*op.dt_per_export
swp = AdaptiveProblem(op)


# --- Simulation and analysis

# Solve forward problem
t1 = time.time()
swp.solve_forward()
t2 = time.time()
J = assemble(swp.fwd_solutions_bathymetry[0]*dx)
rf = ReducedFunctional(J, Control(diff_coeff))

print(J)
print(rf(Constant(0.15)))

h2 = Constant(5e-3)
conv_rate = taylor_test(rf, diff_coeff, h2)
