from thetis import *
from firedrake.petsc import PETSc

from adapt_utils.test_cases.rossby_wave.options import BoydOptions
from adapt_utils.swe.solver import UnsteadyShallowWaterProblem


# NOTE: It seems as though [Huang et al 2008] considers n = 4, 8, 20
n = 1
op = BoydOptions(n=n, order=1)
op.dt = 0.04/n
#op.end_time = 10*op.dt
op.plot_pvd = False
op.dt_per_export = 10*n
op.dt_per_remesh = 10*n
sw = UnsteadyShallowWaterProblem(op=op)
PETSc.Sys.Print("Number of mesh elements: %d" % sw.mesh.num_cells())
sw.solve()

PETSc.Sys.Print("\nError metrics:")
op.get_peaks(sw.solution.split()[1], reference_mesh_resolution=30)  # TODO: n=50
PETSc.Sys.Print("h+       : %.8e" % op.h_upper)
PETSc.Sys.Print("h-       : %.8e" % op.h_lower)
PETSc.Sys.Print("C+       : %.8e" % op.c_upper)
PETSc.Sys.Print("C-       : %.8e" % op.c_lower)
PETSc.Sys.Print("RMS error: %.8e" % op.rms)
