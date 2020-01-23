from thetis import *

from adapt_utils.test_cases.rossby_wave.options import BoydOptions
from adapt_utils.test_cases.rossby_wave.monitors import *
from adapt_utils.swe.solver import UnsteadyShallowWaterProblem


# NOTE: It seems as though [Huang et al 2008] considers n = 4, 8, 20
n_coarse = 1
n_fine = 30  # TODO: 50
# initial_monitor = None
initial_monitor = equator_monitor

op = BoydOptions(n=n_coarse, order=1)
op.debug = True
op.dt = 0.04/n_coarse
# op.end_time = 10*op.dt
op.plot_pvd = n_coarse < 5
op.dt_per_export = 10*n_coarse
op.dt_per_remesh = 10*n_coarse
swp = UnsteadyShallowWaterProblem(op, levels=0)
swp.setup_solver()

if initial_monitor is not None:
    swp.approach = 'monge_ampere'
    swp.monitor_function = initial_monitor
    swp.adapt_mesh()
    # op.approach = 'fixed_mesh'  # TODO: check if needed
    swp.__init__(op, mesh=swp.mesh, levels=swp.levels)

swp.solve(uses_adjoint=False)

print_output("\nCalculating error metrics...")
op.get_peaks(swp.solution.split()[1], reference_mesh_resolution=n_fine)
print_output("h+       : {:.8e}".format(op.h_upper))
print_output("h-       : {:.8e}".format(op.h_lower))
print_output("C+       : {:.8e}".format(op.c_upper))
print_output("C-       : {:.8e}".format(op.c_lower))
print_output("RMS error: {:.8e}".format(op.rms))
