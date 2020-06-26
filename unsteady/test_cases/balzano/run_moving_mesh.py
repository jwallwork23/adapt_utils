from thetis import *

from adapt_utils.unsteady.test_cases.balzano.options import BalzanoOptions
from adapt_utils.unsteady.solver import AdaptiveProblem


op = BalzanoOptions(approach='monge_ampere',
                    plot_pvd=True,
                    debug=True,
                    nonlinear_method='relaxation',
                    # nonlinear_method='quasi_newton',  # FIXME
                    num_adapt=1,
                    n=2,
                    r_adapt_rtol=1.0e-3)
swp = AdaptiveProblem(op, levels=0)
swp.setup_solver_forward()


def wet_dry_interface_monitor(mesh, alpha=1.0, beta=1.0):  # FIXME: all this projection is expensive!
    """
    Monitor function focused around the wet-dry interface.

    NOTE: Defined on the *computational* mesh.

    :kwarg alpha: controls the size of the dense region surrounding the coast.
    :kwarg beta: controls the level of refinement in this region.
    """
    P1 = FunctionSpace(mesh, "CG", 1)
    eta = swp.solution.split()[1]
    b = swp.fields['bathymetry']
    current_mesh = eta.function_space().mesh()
    P1_current = FunctionSpace(current_mesh, "CG", 1)
    diff = interpolate(eta + b, P1_current)
    diff_proj = project(diff, P1)
    return 1.0 + alpha*pow(cosh(beta*diff_proj), -2)


swp.monitor = wet_dry_interface_monitor
swp.run_moving_mesh()  # FIXME
