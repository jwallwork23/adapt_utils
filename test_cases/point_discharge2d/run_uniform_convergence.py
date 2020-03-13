from thetis import print_output

from adapt_utils.test_cases.point_discharge2d.options import *
from adapt_utils.tracer.solver2d import *


num_levels = 5

data = {}
for centred in (True, False):
    data[centred] = {'qois': [], 'num_cells': [], 'qois_exact': []}
    for n in range(num_levels):
        op = TelemacOptions(n=n, centred=centred, offset=1)
        op.degree_increase = 0
        tp = SteadyTracerProblem2d(op, levels=0)
        tp.solve()

        # TODO: Update
        data[centred]['num_cells'].append(tp.num_cells[0])
        data[centred]['qois'].append(tp.quantity_of_interest())
        print_output("\nMesh {:d} in the hierarchy".format(n+1))
        print_output("    Number of elements  : {:d}".format(tp.num_cells[0]))
        print_output("    Quantity of interest: {:.5f}".format(data[centred]['qois'][-1]))
        exact = op.exact_qoi(tp.P1, tp.P0)
        data[centred]['qois_exact'].append(exact)

# TODO: Write to HDF5
for centred in (True, False):
    index = 1 if centred else 2
    print_output("="*80 + "\nLevel  Elements       J{:d}  J{:d}exact".format(index, index))
    for n in range(num_levels):
        print_output("{:5d}  {:8d}  {:7.5f}  {:7.5f}".format(n+1, data[centred]['num_cells'][n], data[centred]['qois'][n], data[centred]['qois_exact'][n]))
