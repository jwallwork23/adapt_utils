from thetis import *

from adapt_utils.test_cases.inundated_beach.options import BalzanoOptions
from adapt_utils.swe.solver import UnsteadyShallowWaterProblem

# op = BalzanoOptions(approach='fixed_mesh_plot', plot_timeseries=False)
op = BalzanoOptions(approach='monge_ampere', plot_timeseries=False)
op.qoi_mode = 'inundation_volume'
op.debug = True
op.num_adapt = 1
swp = UnsteadyShallowWaterProblem(op, levels=0)
swp.setup_solver()

# FIXME: There needs to be some interpolation!!
def monitor_function(mesh, alpha=1.0, beta=1.0):  # NOTE: Need be defined on COMPUTATIONAL mesh
    P1DG = FunctionSpace(mesh, "DG", 1)
    #diff = project(swp.solution.split()[1], P1DG)
    eta = swp.solution.split()[1]
    b = swp.bathymetry
    diff = Function(P1DG)
    diff.dat.data[:] += eta.dat.data - (-b.dat.data)
    # diff = project(eta, FunctionSpace(mesh, "DG", 1))  # FIXME: Projection not working
    # x, y = SpatialCoordinate(mesh)
    # return 1.0 + alpha*pow(cosh(beta*abs(x-13800)), -2)
    return 1.0 + alpha*pow(cosh(beta*abs(diff)), -2)

swp.monitor_function = monitor_function
swp.solve(uses_adjoint=False)
