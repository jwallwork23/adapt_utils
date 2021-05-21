"""
Migrating Trench Test case
=======================

Solves the hydro-morphodynamic simulation of a migrating trench using mesh movement methods
"""

from thetis import *
import firedrake as fire

import pandas as pd
import datetime
import time

from adapt_utils.adapt import recovery
from adapt_utils.io import initialise_bathymetry, export_bathymetry
from adapt_utils.norms import local_frobenius_norm, local_norm
from adapt_utils.unsteady.test_cases.trench_1d.options import TrenchSedimentOptions
from adapt_utils.unsteady.solver import AdaptiveProblem

ts = time.time()
st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
outputdir = 'outputs' + st

res = 0.5
alpha = 1
beta = 1
gamma = 1

# to create the input hydrodynamics directiory please run hydro_trench_slant.py
# setting res to be the same values as above

# --- Set parameters

ts = time.time()
st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
di = os.path.dirname(__file__)
outputdir = os.path.join(di, 'outputs' + st)
inputdir = os.path.join(di, 'hydrodynamics_trench_'+ str(res))
print(inputdir)
kwargs = {
    'approach': 'monge_ampere',
    'nx': res,
    'ny': 1 if res < 4 else 2,
    'plot_pvd': True,
    'input_dir': inputdir,
    'output_dir': outputdir,
    'nonlinear_method': 'relaxation',
    'r_adapt_rtol': 1e-3,
    # Spatial discretisation
    'family': 'dg-dg',
    'stabilisation': 'lax_friedrichs',
    'use_automatic_sipg_parameter': True,
}

op = TrenchSedimentOptions(**kwargs)
assert op.num_meshes == 1
swp = AdaptiveProblem(op)

def gradient_interface_monitor(mesh, alpha=alpha, beta=beta, gamma=gamma):

    """
    Monitor function focused around the steep_gradient (budd acta numerica)

    NOTE: Defined on the *computational* mesh.

    """
    P1 = FunctionSpace(mesh, "CG", 1)

    b = swp.fwd_solutions_bathymetry[0]
    bath_gradient = recovery.construct_gradient(b)
    bath_hess = recovery.construct_hessian(b, op=op)
    frob_bath_hess = Function(b.function_space()).project(local_frobenius_norm(bath_hess))
    frob_bath_norm = Function(b.function_space()).project(frob_bath_hess/max(frob_bath_hess.dat.data[:]))
    current_mesh = b.function_space().mesh()
    l2_bath_grad = Function(b.function_space()).project(local_norm(bath_gradient))
    bath_dx_l2_norm = Function(b.function_space()).interpolate(l2_bath_grad/max(l2_bath_grad.dat.data[:]))

    comp = interpolate(conditional(alpha*beta*bath_dx_l2_norm > alpha*gamma*frob_bath_norm, alpha*beta*bath_dx_l2_norm, alpha*gamma*frob_bath_norm), b.function_space())
    comp_new = project(comp, P1)
    comp_new2 = interpolate(conditional(comp_new > Constant(0.0), comp_new, Constant(0.0)), P1)
    mon_init = project(Constant(1.0) + comp_new2, P1)

    return mon_init

swp.set_monitor_functions(gradient_interface_monitor)

# --- Simulation and analysis

# Solve forward problem

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
df.to_csv('adapt_output/bed_trench_output_uni_s_{:.4f}_{:.1f}_{:.1f}_{:.1f}.csv'.format(res, alpha, beta, gamma))

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
df_exp.to_csv('adapt_output/bed_trench_output_s_{:.4f}_{:.1f}_{:.1f}_{:1f}.csv'.format(res, alpha, beta, gamma))

# Print to screen
print("res = {:.4f}".format(res))
print("alpha = {:.1f}".format(alpha))
print("beta = {:.1f}".format(beta))
print("gamma = {:.1f}".format(gamma))
print("Time: {:.1f}s".format(t2 - t1))
print("Total error: {:.4e}".format(np.sqrt(sum(diff_thetis))))
df_real = pd.read_csv('fixed_output/bed_trench_output_uni_c_4.0000.csv')
print("Discretisation error: {:.4e}".format(np.sqrt(sum([(df['bath'][i] - df_real['bath'][i])**2 for i in range(len(df_real))]))))
