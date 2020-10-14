from thetis import create_directory, print_output

import h5py
import os

from adapt_utils.steady.solver import AdaptiveSteadyProblem
from adapt_utils.steady.test_cases.point_discharge2d.options import PointDischarge2dOptions


# FIXME: Values don't quite agree with what we had previously

num_levels = 5
di = create_directory(os.path.join(os.path.dirname(__file__), 'outputs/fixed_mesh/hdf5'))
for centred in (1, 0):
    index = 2 - centred  # (Centred: J_1, Offset: J_2)
    qois, num_cells, qois_exact = [], [], []

    # Loop over mesh hierarchy
    for level in range(num_levels):
        op = PointDischarge2dOptions(level=level, centred=bool(centred))
        tp = AdaptiveSteadyProblem(op)
        tp.solve_forward()
        num_cells.append(tp.mesh.num_cells())
        qois.append(tp.quantity_of_interest())
        print_output("\nMesh {:d} in the hierarchy".format(level+1))
        print_output("    Number of elements  : {:d}".format(num_cells[-1]))
        print_output("    Quantity of interest: {:.5f}".format(qois[-1]))
        qois_exact.append(op.exact_qoi(tp.P1[0]))

    # Store element count and QoI to HDF5
    outfile = h5py.File(os.path.join(di, 'qoi_{:d}.h5'.format(index)), 'w')
    outfile.create_dataset('elements', data=num_cells)
    outfile.create_dataset('qoi', data=qois)
    outfile.create_dataset('qoi_exact', data=qois_exact)
    outfile.close()

    # Print to screen
    print_output("="*80 + "\nLevel  Elements       J{:d}  J{:d}exact".format(index, index))
    msg = "{:5d}  {:8d}  {:7.5f}  {:7.5f}"
    for level in range(num_levels):
        print_output(msg.format(level+1, num_cells[level], qois[level], qois_exact[level]))
