from thetis import *

from adapt_utils.test_cases.rossby_wave.options import BoydOptions
from adapt_utils.test_cases.rossby_wave.monitors import equator_monitor
from adapt_utils.swe.solver import UnsteadyShallowWaterProblem
from adapt_utils.adapt.recovery import construct_hessian
from adapt_utils.norms import local_frobenius_norm


# NOTE: It seems as though [Huang et al 2008] considers n = 4, 8, 20
n_coarse = 1
n_fine = 30  # TODO: 50
refine_equator = False

op = BoydOptions(n=n_coarse, order=1, approach='monge_ampere')
op.debug = True
op.dt = 0.04/n_coarse
# op.end_time = 10*op.dt
op.plot_pvd = n_coarse < 5
op.dt_per_export = 10*n_coarse
op.dt_per_remesh = 10*n_coarse
swp = UnsteadyShallowWaterProblem(op, levels=0)
swp.setup_solver()

# FIXME: Doesn't there need to be some interpolation?
def elevation_hessian_monitor(mesh, alpha=20.0):
    """
    Monitor function derived from the Frobenius norm of the elevation Hessian.

    NOTE: Defined on the *computational* mesh.

    :kwarg alpha: controls the amplitude of the monitor function.
    """
    P1DG = FunctionSpace(mesh, "DG", 1)
    eta = swp.solution.split()[1]
    eta_copy = Function(P1DG)
    eta_copy.dat.data[:] += eta.dat.data
    H = construct_hessian(eta_copy, op=op)
    return 1.0 + alpha*local_frobenius_norm(H)

if refine_equator:
    swp.monitor_function = equator_monitor
    swp.adapt_mesh()
    swp.__init__(op, mesh=swp.mesh, levels=swp.levels)

swp.monitor_function = elevation_hessian_monitor
swp.solve(uses_adjoint=False)

print_output("\nCalculating error metrics...")
op.get_peaks(swp.solution.split()[1], reference_mesh_resolution=n_fine)
print_output("h+       : {:.8e}".format(op.h_upper))
print_output("h-       : {:.8e}".format(op.h_lower))
print_output("C+       : {:.8e}".format(op.c_upper))
print_output("C-       : {:.8e}".format(op.c_lower))
print_output("RMS error: {:.8e}".format(op.rms))
