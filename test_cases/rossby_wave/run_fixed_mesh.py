from thetis import *
from firedrake.petsc import PETSc

from adapt_utils.test_cases.rossby_wave.options import BoydOptions
from adapt_utils.swe.solver import UnsteadyShallowWaterProblem


# NOTE: It seems as though [Huang et al 2008] considers n = 4, 8, 20
n = 1
op = BoydOptions(n=n, order=1)
op.debug = True
op.dt = 0.04/n
# op.end_time = 10*op.dt
op.plot_pvd = n == 1
op.dt_per_export = 10*n
op.dt_per_remesh = 10*n
sw = UnsteadyShallowWaterProblem(op=op, levels=0)
print_output("Number of mesh elements: {:d}".format(sw.mesh.num_cells()))
sw.solve()

print_output("\nError metrics:")
op.get_peaks(sw.solution.split()[1], reference_mesh_resolution=30)  # TODO: n=50
print_output("h+       : {:.8e}".format(op.h_upper))
print_output("h-       : {:.8e}".format(op.h_lower))
print_output("C+       : {:.8e}".format(op.c_upper))
print_output("C-       : {:.8e}".format(op.c_lower))
print_output("RMS error: {:.8e}".format(op.rms))
