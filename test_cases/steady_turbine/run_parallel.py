from thetis import *
from adapt_utils.test_cases.steady_turbine.options import *
from adapt_utils.turbine.solver import *

offset = False
level = 0

op = Steady2TurbineOffsetOptions() if offset else Steady2TurbineOptions()
op.family = 'dg-cg'
op.plot_pvd = True  # TODO: temp

tp = SteadyTurbineProblem(op, levels=level)
for i in range(level):
    tp = tp.tp_enriched
tp.solve()
J = tp.quantity_of_interest()/1000.0
print_output("\nMesh {:d} in the hierarchy".format(level))
print_output("    Number of elements  : {:d}".format(tp.num_cells[0]))
print_output("    Number of dofs      : {:d}".format(sum(tp.V.dof_count)))
print_output("    Power output        : {:.4f} kW".format(J))
