from firedrake import *
from thetis import print_output
from adapt_utils.test_cases.point_discharge2d.options import *
from adapt_utils.tracer.solver2d import *

num_levels = 5

data = {}
for centred in (True, False):
    op = TelemacOptions(centred=centred, offset=1)
    op.degree_increase = 0
    tp = SteadyTracerProblem2d(op, levels=num_levels-1)  # FIXME: Parallel not working
    data[centred] = {'qois': [], 'num_cells': [], 'qois_exact': []}
    for level in range(num_levels):
        tp.solve()
        data[centred]['num_cells'].append(tp.num_cells[0])
        data[centred]['qois'].append(tp.quantity_of_interest())
        print_output("\nMesh {:d} in the hierarchy".format(level+1))
        print_output("    Number of elements  : {:d}".format(tp.num_cells[0]))
        print_output("    Quantity of interest: {:.5f}".format(data[centred]['qois'][-1]))
        exact = op.exact_qoi(tp.P1, tp.P0)
        data[centred]['qois_exact'].append(exact)
        if level < num_levels-1:
            tp = tp.tp_enriched

# TODO: Write to logfile
for centred in (True, False):
    index = 1 if centred else 2
    print_output("="*80 + "\nLevel  Elements       J{:d}  J{:d}exact".format(index, index))
    for level in range(num_levels):
        print_output("{:5d}  {:8d}  {:7.5f}  {:7.5f}".format(level+1, data[centred]['num_cells'][level], data[centred]['qois'][level], data[centred]['qois_exact'][level]))
