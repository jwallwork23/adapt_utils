from thetis import *

from adapt_utils.unsteady.test_cases.balzano.options import BalzanoOptions
from adapt_utils.unsteady.solver import AdaptiveProblem


kwargs = {
    'approach': 'monge_ampere',
    'plot_pvd': True,
    'debug': True,
    'nonlinear_method': 'relaxation',
    # 'nonlinear_method': 'quasi_newton',  # FIXME
    'n': 2,
    'r_adapt_rtol': 1.0e-3,
    # 'r_adapt_maxit': 5,
}

op = BalzanoOptions(**kwargs)
swp = AdaptiveProblem(op)

alpha = 2.0  # size of the dense region surrounding the coast
beta = 1.0   # level of refinement at coast

def wet_dry_interface_monitor(mesh):  # FIXME: all this projection is expensive!
    """
    Monitor function focused around the wet-dry interface.

    NOTE: Defined on the *computational* mesh.
    """
    P1 = FunctionSpace(mesh, "CG", 1)
    eta = swp.fwd_solutions[0].split()[1]
    b = swp.bathymetry[0]
    current_mesh = eta.function_space().mesh()
    P1_current = FunctionSpace(current_mesh, "CG", 1)
    diff = interpolate(eta + b, P1_current)
    diff_proj = project(diff, P1)
    return 1.0 + alpha*pow(cosh(beta*diff_proj), -2)


swp.set_monitor_functions(wet_dry_interface_monitor)
swp.solve_forward()
