from thetis import *
from adapt_utils.test_cases.steady_turbine.options import *
from adapt_utils.turbine.solver import *

offset = False
level = 0

op = Steady2TurbineOffsetOptions() if offset else Steady2TurbineOptions()
mh = MeshHierarchy(op.default_mesh, level)
op.family = 'dg-cg'

tp = SteadyTurbineProblem(mesh=mh[level], op=op)
tp.solve()
J = tp.quantity_of_interest()/1000.0
print_output("Mesh {:d} in the hierarchy".format(level))
print_output("    Number of elements  : {:d}".format(tp.mesh.num_cells()))
print_output("    Number of dofs      : {:d}".format(sum(tp.V.dof_count)))
print_output("    Power output        : {:.4f} kW".format(J))
