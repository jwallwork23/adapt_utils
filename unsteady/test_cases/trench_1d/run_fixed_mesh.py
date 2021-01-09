"""
Migrating Trench Test case
=======================

Solves the hydro-morphodynamic simulation of a migrating trench on a fixed mesh
"""
from thetis import *

import argparse
import datetime
import os
import pandas as pd
import sys
import time

from adapt_utils.unsteady.solver import AdaptiveProblem
from adapt_utils.unsteady.test_cases.trench_1d.options import TrenchSedimentOptions


# To create the input hydrodynamics directiory please run trench_hydro.py
# setting res to be the same values as above.
# We have included the hydrodynamics input dir for res = 0.5 as an example.

# --- Parse arguments

parser = argparse.ArgumentParser()
parser.add_argument("-res", help="Mesh resolution factor (default 0.5).")
args = parser.parse_args()

res = float(args.res or 0.5)


# --- Set parameters

ts = time.time()
st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
di = os.path.dirname(__file__)
outputdir = os.path.join(di, 'outputs' + st)
inputdir = os.path.join(di, 'hydrodynamics_trench_{:.1f}'.format(res))
kwargs = {
    'approach': 'fixed_mesh',
    'nx': res,
    'ny': 1,
    'plot_pvd': True,
    'input_dir': inputdir,
    'output_dir': outputdir,

    # Spatial discretisation
    'family': 'dg-dg',
    'stabilisation': None,
    'stabilisation_sediment': None,
    'use_automatic_sipg_parameter': True,
}
op = TrenchSedimentOptions(**kwargs)
if os.getenv('REGRESSION_TEST') is not None:
    op.end_time = op.dt*op.dt_per_export
if res >= 4.0:
    op.dt *= 0.5
swp = AdaptiveProblem(op)


# --- Simulation and analysis

# Solve forward problem
t1 = time.time()
swp.solve_forward()
t2 = time.time()
if os.getenv('REGRESSION_TEST') is not None:
    sys.exit(0)

# Save solution data
new_mesh = RectangleMesh(16*5*5, 5*1, 16, 1.1)
bath = Function(FunctionSpace(new_mesh, "CG", 1)).project(swp.fwd_solutions_bathymetry[0])
datathetis = []
bathymetrythetis1 = []
diff_thetis = []
datathetis = np.linspace(0, 15.9, 160)
bathymetrythetis1 = [-bath.at([i, 0.55]) for i in datathetis]
df = pd.concat([pd.DataFrame(datathetis, columns=['x']), pd.DataFrame(bathymetrythetis1, columns=['bath'])], axis=1)
df.to_csv(os.path.join(di, 'fixed_output/bed_trench_output_uni_c_{:.1f}.csv'.format(res)))

# Compute l2 error against experimental data
datathetis = []
bathymetrythetis1 = []
diff_thetis = []
data = pd.read_csv(os.path.join(di, 'experimental_data.csv'), header=None)
for i in range(len(data[0].dropna())):
    datathetis.append(data[0].dropna()[i])
    bathymetrythetis1.append(-bath.at([np.round(data[0].dropna()[i], 3), 0.55]))
    diff_thetis.append((data[1].dropna()[i] - bathymetrythetis1[-1])**2)
df = pd.concat([pd.DataFrame(datathetis, columns=['x']), pd.DataFrame(bathymetrythetis1, columns=['bath'])], axis=1)
df.to_csv(os.path.join(di, 'fixed_output/bed_trench_output_c_{:.1f}.csv'.format(res)))

# Print to screen
print("res = {:.1f}".format(res))
print("Time: {:.1f}s".format(t2 - t1))
print("Total error: {:.4e}".format(np.sqrt(sum(diff_thetis))))
