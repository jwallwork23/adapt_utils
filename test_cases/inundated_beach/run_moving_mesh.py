from thetis import *

from adapt_utils.test_cases.inundated_beach.options import BalzanoOptions
from adapt_utils.swe.solver import UnsteadyShallowWaterProblem

op = BalzanoOptions(approach='monge_ampere', plot_timeseries=False)
op.qoi_mode = 'inundation_volume'
# op.debug = True
swp = UnsteadyShallowWaterProblem(op, levels=0)
swp.setup_solver()

# FIXME: Doesn't there need to be some interpolation?
def wet_dry_interface_monitor(mesh, alpha=1.0, beta=1.0):
    """
    Monitor function focused around the wet-dry interface.

    NOTE: Defined on the *computational* mesh.

    :kwarg alpha: controls the size of the dense region surrounding the coast.
    :kwarg beta: controls the level of refinement in this region.
    """
    P1DG = FunctionSpace(mesh, "DG", 1)
    eta = swp.solution.split()[1]
    b = swp.bathymetry
    diff = Function(P1DG)
    diff.dat.data[:] += eta.dat.data - (-b.dat.data)
    return 1.0 + alpha*pow(cosh(beta*diff), -2)

swp.monitor_function = wet_dry_interface_monitor
swp.solve(uses_adjoint=False)
