from thetis import print_output

import h5py

from adapt_utils.test_cases.point_discharge2d.options import *
from adapt_utils.tracer.solver2d import *


num_levels = 5

for centred in (1, 0):
    index = 2 - centred  # (Centred: J_1, Offset: J_2)
    qois, num_cells, qois_exact = [], [], []

    # Loop over mesh hierarchy
    for n in range(num_levels):
        op = TelemacOptions(n=n, centred=bool(centred))
        op.degree_increase = 0
        tp = SteadyTracerProblem2d(op, levels=0)
        tp.solve()

        # TODO: Update
        num_cells.append(tp.num_cells[0])
        qois.append(tp.quantity_of_interest())
        op.print_debug("\nMesh {:d} in the hierarchy".format(n+1))
        op.print_debug("    Number of elements  : {:d}".format(tp.num_cells[0]))
        op.print_debug("    Quantity of interest: {:.5f}".format(qois[-1]))
        qois_exact.append(op.exact_qoi(tp.P1, tp.P0))

    # Store element count and QoI to HDF5
    outfile = h5py.File('outputs/fixed_mesh/hdf5/qoi_{:d}.h5'.format(index), 'w')
    outfile.create_dataset('elements', data=num_cells)
    outfile.create_dataset('qoi', data=qois)
    outfile.create_dataset('qoi_exact', data=qois_exact)
    outfile.close()

    # Print to screen
    print_output("="*80 + "\nLevel  Elements       J{:d}  J{:d}exact".format(index, index))
    msg = "{:5d}  {:8d}  {:7.5f}  {:7.5f}"
    for n in range(num_levels):
        print_output(msg.format(n+1, num_cells[n], qois[n], qois_exact[n]))
