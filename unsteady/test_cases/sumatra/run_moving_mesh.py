import thetis as th
import pylab as plt

from adapt_utils.unsteady.test_cases.sumatra.options import SumatraOptions
from adapt_utils.unsteady.solver import AdaptiveProblem
from adapt_utils.adapt import recovery
from adapt_utils.norms import local_frobenius_norm, local_norm

import time
import datetime

alpha_std = 30
beta = 1
gamma = 1

ts = time.time()
st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
outputdir = 'outputs' + st

kwargs = {
    'approach': 'monge_ampere',
    'plot_pvd': True,
    'output_dir': outputdir,
    'nonlinear_method': 'relaxation',
    'r_adapt_rtol': 1.0e-3,
    # Spatial discretisation
    'family': 'dg-dg',
    'stabilisation': None,
    'use_automatic_sipg_parameter': True,
}

op = SumatraOptions(**kwargs)
assert op.num_meshes == 1
swp = AdaptiveProblem(op)
swp.shallow_water_options[0]['mesh_velocity'] = None

def gradient_interface_monitor(mesh, alpha=alpha_std, beta=beta, gamma=gamma):

    """
    Monitor function focused around the steep_gradient (budd acta numerica)

    NOTE: Defined on the *computational* mesh.

    """

    P1 = th.FunctionSpace(mesh, "CG", 1)
    b = swp.bathymetry[0]
    #x_1, y_1 = th.SpatialCoordinate(b_copy.function_space().mesh())

    #b = th.Function(b_copy.function_space()).project(th.conditional(x_1 < 3600, b_copy, th.Constant(b_copy.at([3600, 60]))))
    bath_gradient = b.dx(0) #recovery.construct_gradient(b)
    # second order derivative
    bath_hess = recovery.recover_hessian(b, op=op)
    frob_bath_hess = th.Function(swp.bathymetry[0].function_space()).project(local_frobenius_norm(bath_hess))

    if max(abs(frob_bath_hess.dat.data[:]))<1e-10:
        frob_bath_norm = th.Function(swp.bathymetry[0].function_space()).project(frob_bath_hess)
    else:
        frob_bath_norm = th.Function(swp.bathymetry[0].function_space()).project(frob_bath_hess/max(frob_bath_hess.dat.data[:]))

    l2_bath_grad = th.Function(swp.bathymetry[0].function_space()).project(abs(bath_gradient))

    bath_dx_l2_norm = th.Function(swp.bathymetry[0].function_space()).interpolate(l2_bath_grad/max(l2_bath_grad.dat.data[:]))
    
    x1, y1 = th.SpatialCoordinate(b.function_space().mesh())
    alpha = th.conditional(x1 < 3600, th.Constant(alpha_std), (-alpha_std)/(4044 - 3600) *(x1-3600) + alpha_std)
    comp = th.interpolate(th.conditional(alpha*beta*bath_dx_l2_norm > alpha*gamma*frob_bath_norm, alpha*beta*bath_dx_l2_norm, alpha*gamma*frob_bath_norm), b.function_space())
    comp_new = th.project(comp, P1)
    comp_new2 = th.interpolate(th.conditional(comp_new > th.Constant(0.0), comp_new, th.Constant(0.0)), P1)
    mon_init = th.project(th.Constant(1) + comp_new2, P1)

    H = th.Function(P1)
    tau = th.TestFunction(P1)
    n = th.FacetNormal(mesh)

    K = 40*(12**2)/4
    
    a = (th.inner(tau, H)*th.dx)+(K*th.inner(th.grad(tau), th.grad(H))*th.dx) - (K*(tau*th.inner(th.grad(H), n)))*th.ds
    a -= th.inner(tau, mon_init)*th.dx
    th.solve(a == 0, H)
    
    
    return th.Constant(1) #H

swp.set_monitor_functions(gradient_interface_monitor)

t1 = time.time()
swp.solve_forward()
t2 = time.time()
