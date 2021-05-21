"""
Migrating Trench Test case
=======================

Solves the hydro-morphodynamic simulation of a migrating trench using mesh movement methods
"""
from thetis import *

import argparse
import datetime
import os
import pandas as pd
import time

from adapt_utils.adapt import recovery
from adapt_utils.norms import local_frobenius_norm
from adapt_utils.unsteady.solver import AdaptiveProblem
from adapt_utils.unsteady.test_cases.trench_1d.options import TrenchSedimentOptions


# To create the input hydrodynamics directiory please run trench_hydro.py
# setting res to be the same values as above.
# We have included the hydrodynamics input dir for res = 0.5 as an example.

# --- Parse arguments

parser = argparse.ArgumentParser()
parser.add_argument("-alpha", help="Scaling factor for Hessian.")
parser.add_argument("-res", help="Mesh resolution factor (default 0.5).")
parser.add_argument("-rtol", help="Relative tolerance for relaxation method (default 1.0e-03).")
parser.add_argument("-dt_per_mesh_movement", help="Number of timesteps per mesh movement.")
args = parser.parse_args()

alpha = float(args.alpha or 2.0)
alpha_const = Constant(alpha)
res = float(args.res or 0.5)
rtol = float(args.rtol or 1.0e-03)
freq = int(args.dt_per_mesh_movement or 40)

# --- Set parameters

ts = time.time()
st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
di = os.path.dirname(__file__)
outputdir = os.path.join(di, 'outputs' + st)
inputdir = os.path.join(di, 'hydrodynamics_trench_' + str(res))
kwargs = {
    'approach': 'monge_ampere',
    'nx': res,
    'ny': 1 if res < 4 else 2,
    'plot_pvd': True,
    'input_dir': inputdir,
    'output_dir': outputdir,
    'nonlinear_method': 'relaxation',
    'r_adapt_rtol': rtol,

    # Spatial discretisation
    'family': 'dg-dg',
    'stabilisation': 'lax_friedrichs',
    'stabilisation_sediment': 'lax_friedrichs',
}
op = TrenchSedimentOptions(**kwargs)
op.dt_per_mesh_movement = freq
assert op.num_meshes == 1
swp = AdaptiveProblem(op)


def frobenius_monitor(mesh):
    """
    Frobenius norm taken component-wise.
    """
    P1 = FunctionSpace(mesh, "CG", 1)
    b = project(swp.fwd_solutions_bathymetry[0], P1)
    H = recovery.recover_hessian(b, op=op)
    frob = sqrt(H[0, 0]**2 + H[0, 1]**2 + H[1, 0]**2 + H[1, 1]**2)
    return 1 + alpha_const*frob/interpolate(frob, P1).vector().gather().max()


def frobenius_monitor(mesh, x=None):  # NOQA: Version above not smooth enough
    """
    Frobenius norm taken element-wise.
    """
    P1 = FunctionSpace(mesh, "CG", 1)
    b = swp.fwd_solutions_bathymetry[0]
    P0 = FunctionSpace(b.function_space().mesh(), "DG", 0)
    H = recovery.recover_hessian(b, op=op)
    frob = interpolate(local_frobenius_norm(H), P0)
    return 1 + alpha_const*project(frob, P1)/frob.vector().gather().max()


# --- Simulation and analysis

# Solve forward problem
swp.set_monitor_functions(frobenius_monitor)
t1 = time.time()
swp.solve_forward()
t2 = time.time()

# Save solution data
new_mesh = RectangleMesh(16*5*5, 5*1, 16, 1.1)
bath = Function(FunctionSpace(new_mesh, "CG", 1)).project(swp.fwd_solutions_bathymetry[0])
bathymetrythetis1 = []
diff_thetis = []
datathetis = np.linspace(0, 15.9, 160)
bathymetrythetis1 = [-bath.at([i, 0.55]) for i in datathetis]
df = pd.concat([pd.DataFrame(datathetis, columns=['x']), pd.DataFrame(bathymetrythetis1, columns=['bath'])], axis=1)
df.to_csv('adapt_output/bed_trench_output_uni_s_{:.4f}_{:.1f}_{:.1e}_{:d}.csv'.format(res, alpha, rtol, freq))

# Compute l2 error against experimental data
datathetis = []
bathymetrythetis1 = []
diff_thetis = []
data = pd.read_csv('experimental_data.csv', header=None)
for i in range(len(data[0].dropna())):
    datathetis.append(data[0].dropna()[i])
    bathymetrythetis1.append(-bath.at([np.round(data[0].dropna()[i], 3), 0.55]))
    diff_thetis.append((data[1].dropna()[i] - bathymetrythetis1[-1])**2)
df_exp = pd.concat([pd.DataFrame(datathetis, columns=['x']), pd.DataFrame(bathymetrythetis1, columns=['bath'])], axis=1)
df_exp.to_csv('adapt_output/bed_trench_output_s_{:.4f}_{:.1f}_{:.1e}_{:d}.csv'.format(res, alpha, rtol, freq))

# Print to screen
print("res = {:.4f}".format(res))
print("alpha = {:.1f}".format(alpha))
print("rtol = {:.1e}".format(rtol))
print("Time: {:.1f}s".format(t2 - t1))
print("Total error: {:.4e}".format(np.sqrt(sum(diff_thetis))))
df_real = pd.read_csv('fixed_output/bed_trench_output_uni_c_4.0000.csv')
print("Discretisation error: {:.4f}".format(np.sqrt(sum([(df['bath'][i] - df_real['bath'][i])**2 for i in range(len(df_real))]))))
