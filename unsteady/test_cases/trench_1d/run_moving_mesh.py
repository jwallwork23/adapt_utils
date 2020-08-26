from thetis import *

import datetime
import numpy as np
import pandas as pd
import time

from adapt_utils.adapt import recovery
from adapt_utils.norms import local_frobenius_norm
from adapt_utils.unsteady.solver import AdaptiveProblem
from adapt_utils.unsteady.test_cases.trench_1d.options import TrenchSedimentOptions


ts = time.time()
st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
outputdir = 'outputs' + st

nx = 0.8
alpha = 2

inputdir = 'hydrodynamics_trench_' + str(nx)

kwargs = {
    'approach': 'monge_ampere',
    'nx': nx,
    'ny': 1,
    'plot_pvd': True,
    'input_dir': inputdir,
    'output_dir': outputdir,
    'nonlinear_method': 'relaxation',
    'r_adapt_rtol': 1.0e-3,
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

def gradient_interface_monitor(mesh, alpha=alpha, gamma=0.0):
    """
    Monitor function focused around the steep_gradient (budd acta numerica)

    NOTE: Defined on the *computational* mesh.
    """
    P1 = FunctionSpace(mesh, "CG", 1)

    # eta = swp.solution.split()[1]
    b = swp.fwd_solutions_bathymetry[0]
    # bath_gradient = recovery.construct_gradient(b)
    bath_hess = recovery.construct_hessian(b, op=op)
    frob_bath_hess = Function(b.function_space()).project(local_frobenius_norm(bath_hess))
    frob_bath_norm = Function(b.function_space()).project(frob_bath_hess/max(frob_bath_hess.dat.data[:]))
    norm_two_proj = project(frob_bath_norm, P1)

    H = Function(P1)
    tau = TestFunction(P1)
    n = FacetNormal(mesh)

    mon_init = project(Constant(1.0) + alpha * norm_two_proj, P1)
    #K = 10*(0.2**2)/4
    #a = (inner(tau, H)*dx)+(K*inner(grad(tau), grad(H))*dx) - (K*(tau*inner(grad(H), n)))*ds
    #a -= inner(tau, mon_init)*dx
    #solve(a == 0, H)

    return mon_init

swp.set_monitor_functions(gradient_interface_monitor)

t1 = time.time()
swp.solve_forward()
t2 = time.time()

new_mesh = RectangleMesh(16*5*5, 5*1, 16, 1.1)

bath = Function(FunctionSpace(new_mesh, "CG", 1)).project(swp.fwd_solutions_bathymetry[0])

data = pd.read_csv('experimental_data.csv', header=None)

datathetis = []
bathymetrythetis1 = []
diff_thetis = []
for i in np.linspace(0, 15.9, 160):
    datathetis.append(i)
    bathymetrythetis1.append(-bath.at([i, 0.55]))

df = pd.concat([pd.DataFrame(datathetis, columns=['x']), pd.DataFrame(bathymetrythetis1, columns=['bath'])], axis=1)

df.to_csv('adapt_output/bed_trench_output_uni_s' + str(nx) + '_' + str(alpha) + '.csv')

datathetis = []
bathymetrythetis1 = []
diff_thetis = []
for i in range(len(data[0].dropna())):
    print(i)
    datathetis.append(data[0].dropna()[i])
    bathymetrythetis1.append(-bath.at([np.round(data[0].dropna()[i], 3), 0.55]))
    diff_thetis.append((data[1].dropna()[i] - bathymetrythetis1[-1])**2)

df_exp = pd.concat([pd.DataFrame(datathetis, columns=['x']), pd.DataFrame(bathymetrythetis1, columns=['bath'])], axis=1)

df_exp.to_csv('adapt_output/bed_trench_output_s' + str(nx) + '_' + str(alpha) + '.csv')

print(nx)
print(alpha)
print("Total error: ")
print(np.sqrt(sum(diff_thetis)))

print("total time: ")
print(t2-t1)


df_real = pd.read_csv('fixed_output/bed_trench_output_uni_4.csv')
print("Mesh error: ")
print(sum([(df['bath'][i] - df_real['bath'][i])**2 for i in range(len(df_real))]))
#f = open("adapt_output/output_frob_norm_" + str(nx) + '_' + str(alpha) + '.txt', "w+")
#f.write(str(np.sqrt(sum(diff_thetis))))
#f.write("\n")
#f.write(str(t2-t1))
#f.close()
