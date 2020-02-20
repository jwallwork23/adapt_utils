from thetis import *
from adapt_utils.test_cases.steady_turbine.options import *
from adapt_utils.swe.turbine.solver import *

num_levels = 5

kwargs = {'plot_pvd': True, 'debug': True}

data = {}
for offset in (False, True):
    op = Steady2TurbineOffsetOptions(**kwargs) if offset else Steady2TurbineOptions(**kwargs)
    tp = SteadyTurbineProblem(op, levels=num_levels-1)
    data[offset] = {'qois': [], 'num_cells': [], 'dofs': []}
    for level in range(num_levels):
        tp.solve()
        data[offset]['num_cells'].append(tp.num_cells[0])
        data[offset]['dofs'].append(sum(tp.V.dof_count))
        data[offset]['qois'].append(tp.quantity_of_interest()/1000.0)
        op.print_debug("\nMesh {:d} in the hierarchy".format(level+1))
        op.print_debug("    Number of elements  : {:d}".format(tp.num_cells[0]))
        op.print_debug("    Number of dofs      : {:d}".format(data[offset]['dofs'][-1]))
        op.print_debug("    Power output        : {:.4f} kW".format(data[offset]['qois'][-1]))
        if level < num_levels-1:
            tp = tp.tp_enriched

# TODO: Write to logfile
for offset in (False, True):
    print_output("="*80 + "\nLevel  Elements     DOFs        J{:d}".format(2 if offset else 1))
    for level in range(num_levels):
        print_output("{:5d}  {:8d}  {:7d}  {:6.4f}kW".format(level+1, data[offset]['num_cells'][level], data[offset]['dofs'][level], data[offset]['qois'][level]))
