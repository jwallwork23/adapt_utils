from firedrake_adjoint import *
from thetis import *

import numpy as np

from adapt_utils.unsteady.test_cases.trench_adjoint.options import TrenchSedimentOptions
from adapt_utils.unsteady.solver import AdaptiveProblem
from adapt_utils.adapt import recovery
from adapt_utils.norms import local_frobenius_norm, local_norm

import pandas as pd
import time
import datetime

ts = time.time()
st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
outputdir = 'outputs' + st

res = 0.5
alpha = Constant(1)
beta = Constant(0.5)
gamma = Constant(1)

diff_coeff = Constant(0.18011042551606954)
fric_coeff = Constant(0.025)

tol = 1e-3

inputdir = 'hydrodynamics_trench_' + str(res)

kwargs = {
    'approach': 'monge_ampere',
    'diff_coeff': diff_coeff,
    'fric_coeff': fric_coeff,
    'nx': res,
    'ny': 1,
    'plot_pvd': True,
    'input_dir': inputdir,
    'output_dir': outputdir,
    'nonlinear_method': 'relaxation',
    'r_adapt_rtol': tol,
    # Spatial discretisation
    'family': 'dg-dg',
    'stabilisation': 'lax_friedrichs',
    'use_automatic_sipg_parameter': True,
}


op = TrenchSedimentOptions(**kwargs)
assert op.num_meshes == 1
swp = AdaptiveProblem(op)
# swp.shallow_water_options[0]['mesh_velocity'] = swp.mesh_velocities[0]
swp.shallow_water_options[0]['mesh_velocity'] = None

def gradient_interface_monitor(mesh, alpha=alpha, beta=beta, gamma=gamma, x = None):

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

t1 = time.time()
swp.solve_forward()
t2 = time.time()

#J = assemble(swp.fwd_solutions_bathymetry[0]*dx)
#rf = ReducedFunctional(J, Control(fric_coeff))

#print(J)
#print(rf(fric_coeff))

new_mesh = RectangleMesh(16*5*5, 5*1, 16, 1.1)

bath = Function(FunctionSpace(new_mesh, "CG", 1)).project(swp.fwd_solutions_bathymetry[0])

old_mesh = swp.fwd_solutions_bathymetry[0].function_space().mesh()

mesh_element_size = Function(FunctionSpace(old_mesh, "DG", 0)).interpolate(CellSize(old_mesh))

print(max(mesh_element_size.dat.data[:]))
print(min(mesh_element_size.dat.data[:]))

data = pd.read_csv('experimental_data.csv', header=None)

datathetis = []
bathymetrythetis1 = []
diff_thetis = []
for i in np.linspace(0, 15.9, 160):
    datathetis.append(i)
    bathymetrythetis1.append(-bath.at([i, 0.55]))

df = pd.concat([pd.DataFrame(datathetis, columns=['x']), pd.DataFrame(bathymetrythetis1, columns=['bath'])], axis=1)

#df.to_csv('adapt_output/bed_trench_output_uni_s' + str(nx) + '_' + str(alpha) + '_' + str(tol) + '.csv')

datathetis = []
bathymetrythetis1 = []
diff_thetis = []
for i in range(len(data[0].dropna())):
    datathetis.append(data[0].dropna()[i])
    bathymetrythetis1.append(-bath.at([np.round(data[0].dropna()[i], 3), 0.55]))
    diff_thetis.append((data[1].dropna()[i] - bathymetrythetis1[-1])**2)

df_exp = pd.concat([pd.DataFrame(datathetis, columns=['x']), pd.DataFrame(bathymetrythetis1, columns=['bath'])], axis=1)

#df_exp.to_csv('adapt_output/bed_trench_output_s' + str(nx) + '_' + str(alpha) + '_' + str(tol) + '.csv')

print(res)
print(alpha)
print(beta)
print(gamma)
print("Total error: ")
print(np.sqrt(sum(diff_thetis)))

print("total time: ")
print(t2-t1)


df_real = pd.read_csv('fixed_output/bed_trench_output_uni_c_4.0000.csv')
print("Mesh error: ")
print(sum([(df['bath'][i] - df_real['bath'][i])**2 for i in range(len(df_real))]))

print('tolerance')
print(tol)


