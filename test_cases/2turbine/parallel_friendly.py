from thetis import *
from firedrake.petsc import PETSc
from adapt_utils.turbine.options import *
from adapt_utils.turbine.solver import *

offset = False
init_mesh = 'xcoarse'
level = 5
num_turbines = 2

if offset:
    filename = '_'.join([init_mesh, str(num_turbines), 'offset', 'turbine.msh'])
else:
    filename = '_'.join([init_mesh, str(num_turbines), 'turbine.msh'])
mesh = Mesh(filename)
mh = MeshHierarchy(mesh, level)

op = Steady2TurbineOffsetOptions() if offset else Steady2TurbineOptions()
op.family = 'dg-cg'

tp = SteadyTurbineProblem(mesh=mh[level], op=op)
tp.solve()
J = tp.quantity_of_interest()/1000.0
PETSc.Sys.Print("Mesh %d in the hierarchy" % level)
PETSc.Sys.Print("    Number of elements  : %d" % tp.mesh.num_cells())
PETSc.Sys.Print("    Number of dofs      : %d" % sum(tp.V.dof_count))
PETSc.Sys.Print("    Power output        : %.4f kW" % J)
