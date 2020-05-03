from thetis import *

import pylab as plt
import pandas as pd
import numpy as np
import time

from adapt_utils.test_cases.trench_test.options import TrenchOptions
from adapt_utils.swe.solver import UnsteadyShallowWaterProblem
from adapt_utils.adapt import recovery
from adapt_utils.norms import local_frobenius_norm

t1 = time.time()

nx = 1.0
alpha = 100.0

op = TrenchOptions(approach='monge_ampere',
                   plot_timeseries=False,
                   plot_pvd=True,
                   debug=False,
                   nonlinear_method='relaxation',
                   num_adapt=1,
                   friction='nikuradse',
                   nx=nx,
                   ny=1,
                   r_adapt_rtol=1.0e-3)

swp = UnsteadyShallowWaterProblem(op, levels=0)
swp.setup_solver()


def gradient_interface_monitor(mesh, alpha=alpha, gamma=0.0):

    """
    Monitor function focused around the steep_gradient (budd acta numerica)

    NOTE: Defined on the *computational* mesh.

    """
    P1 = FunctionSpace(mesh, "CG", 1)

    # eta = swp.solution.split()[1]
    b = swp.solver_obj.fields.bathymetry_2d
    # bath_gradient = recovery.construct_gradient(b)
    bath_hess = recovery.construct_hessian(b, op=op)
    frob_bath_hess = Function(b.function_space()).project(local_frobenius_norm(bath_hess))

    # current_mesh = eta.function_space().mesh()
    # P1_current = FunctionSpace(current_mesh, "CG", 1)
    # bath_dx_sq = interpolate(pow(bath_gradient[0], 2), P1_current)
    # bath_dy_sq = interpolate(pow(bath_gradient[1], 2), P1_current)
    # bath_dx_dx_sq = interpolate(pow(bath_dx_sq.dx(0), 2), P1_current)
    # bath_dy_dy_sq = interpolate(pow(bath_dy_sq.dx(1), 2), P1_current)
    # norm = interpolate(conditional(bath_dx_dx_sq + bath_dy_dy_sq > 10**(-7), bath_dx_dx_sq + bath_dy_dy_sq, Constant(10**(-7))), P1_current)
    # norm_two = interpolate(bath_dx_dx_sq + bath_dy_dy_sq, P1_current)
    # norm_one = interpolate(bath_dx_sq + bath_dy_sq, P1_current)
    # norm_tmp = interpolate(bath_dx_sq/norm, P1_current)
    # norm_one_proj = project(norm_one, P1)
    norm_two_proj = project(frob_bath_hess, P1)

    H = Function(P1)
    tau = TestFunction(P1)
    n = FacetNormal(mesh)

    mon_init = project(sqrt(Constant(1.0) + alpha * norm_two_proj), P1)

    K = 10*(0.4**2)/4
    a = (inner(tau, H)*dx)+(K*inner(grad(tau), grad(H))*dx) - (K*(tau*inner(grad(H), n)))*ds
    a -= inner(tau, mon_init)*dx
    solve(a == 0, H)

    return H


swp.monitor_function = gradient_interface_monitor
swp.solve(uses_adjoint=False)

t2 = time.time()

new_mesh = RectangleMesh(16*5*5, 5*1, 16, 1.1)

bath = Function(FunctionSpace(new_mesh, "CG", 1)).project(swp.solver_obj.fields.bathymetry_2d)

data = pd.read_csv('experimental_data.csv', header=None)

datathetis = []
bathymetrythetis1 = []
diff_thetis = []
for i in range(len(data[0].dropna())):
    print(i)
    datathetis.append(data[0].dropna()[i])
    bathymetrythetis1.append(-bath.at([np.round(data[0].dropna()[i], 3), 0.55]))
    diff_thetis.append((data[1].dropna()[i] - bathymetrythetis1[-1])**2)

df = pd.concat([pd.DataFrame(datathetis, columns=['x']), pd.DataFrame(bathymetrythetis1, columns=['bath'])], axis=1)

df.to_csv('adapt_output2/bed_trench_output' + str(nx) + '_' + str(alpha) + '.csv')

plt.plot(datathetis, bathymetrythetis1, '.', linewidth=2, label='adapted mesh')
plt.legend()
plt.show()

print("L2 norm: ")
print(np.sqrt(sum(diff_thetis)))

print("total time: ")
print(t2-t1)

f = open("adapt_output2/output_frob_norm" + str(nx) + '_' + str(100.0) + '.txt', "w+")
f.write(str(np.sqrt(sum(diff_thetis))))
f.write("\n")
f.write(str(t2-t1))
f.close()
