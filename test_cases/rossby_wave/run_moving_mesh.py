from thetis import *

from adapt_utils.test_cases.rossby_wave.options import BoydOptions
from adapt_utils.test_cases.rossby_wave.monitors import *
from adapt_utils.swe.solver import UnsteadyShallowWaterProblem
from adapt_utils.adapt.recovery import construct_hessian
from adapt_utils.adapt.metric import metric_intersection
from adapt_utils.norms import *


# NOTE: It seems as though [Huang et al 2008] considers n = 4, 8, 20
n_coarse = 1
n_fine = 30  # TODO: 50
# initial_monitor = None
# initial_monitor = equator_monitor
initial_monitor = soliton_monitor

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
def elevation_norm_monitor(mesh, alpha=40.0, norm_type='H1'):
    """
    Monitor function derived from the elevation `norm_type` norm.

    NOTE: Defined on the *computational* mesh.

    :kwarg alpha: controls the amplitude of the monitor function.
    """
    P1DG = FunctionSpace(mesh, "DG", 1)
    eta = swp.solution.split()[1]
    eta_copy = Function(P1DG)
    eta_copy.dat.data[:] += eta.dat.data
    if norm_type == 'hessian_frobenius':
        H = construct_hessian(eta_copy, op=op)
        return 1.0 + alpha*local_frobenius_norm(H)
    else:
        return 1.0 + alpha*local_norm(eta_copy, norm_type=norm_type)

# FIXME: Doesn't there need to be some interpolation?
def velocity_norm_monitor(mesh, alpha=40.0, norm_type='HDiv'):
    """
    Monitor function derived from the velocity `norm_type` norm.

    NOTE: Defined on the *computational* mesh.

    :kwarg alpha: controls the amplitude of the monitor function.
    """
    P1DG_vec = VectorFunctionSpace(mesh, "DG", 1)
    u = swp.solution.split()[0]
    u_copy = Function(P1DG_vec)
    u_copy.dat.data[:] += u.dat.data
    if norm_type == 'hessian_frobenius':
        H1 = construct_hessian(u_copy[0], op=op)
        H2 = construct_hessian(u_copy[1], op=op)
        return 1.0 + alpha*local_frobenius_norm(metric_intersection(H1, H2))
    else:
        return 1.0 + alpha*local_norm(u_copy, norm_type=norm_type)

def mixed_monitor(mesh):
    return 0.5*(velocity_norm_monitor(mesh, norm_type='HDiv') +
                elevation_norm_monitor(mesh, norm_type='H1'))

if initial_monitor is not None:
    swp.monitor_function = initial_monitor
    swp.adapt_mesh()
    swp.__init__(op, mesh=swp.mesh, levels=swp.levels)

# swp.monitor_function = elevation_norm_monitor
# swp.monitor_function = velocity_norm_monitor
swp.monitor_function = mixed_monitor
swp.solve(uses_adjoint=False)

print_output("\nCalculating error metrics...")
op.get_peaks(swp.solution.split()[1], reference_mesh_resolution=n_fine)
print_output("h+       : {:.8e}".format(op.h_upper))
print_output("h-       : {:.8e}".format(op.h_lower))
print_output("C+       : {:.8e}".format(op.c_upper))
print_output("C-       : {:.8e}".format(op.c_lower))
print_output("RMS error: {:.8e}".format(op.rms))
