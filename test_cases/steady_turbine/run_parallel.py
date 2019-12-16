from thetis import *
from firedrake.petsc import PETSc
from adapt_utils.test_cases.steady_turbine.options import *
from adapt_utils.turbine.solver import *

offset = False
level = 5

op = Steady2TurbineOffsetOptions() if offset else Steady2TurbineOptions()
mh = MeshHierarchy(op.default_mesh, level)
op.family = 'dg-cg'

tp = SteadyTurbineProblem(mesh=mh[level], op=op)
tp.solve()
J = tp.quantity_of_interest()/1000.0
PETSc.Sys.Print("Mesh %d in the hierarchy" % level)
PETSc.Sys.Print("    Number of elements  : %d" % tp.mesh.num_cells())
PETSc.Sys.Print("    Number of dofs      : %d" % sum(tp.V.dof_count))
PETSc.Sys.Print("    Power output        : %.4f kW" % J)
