from thetis import create_directory, print_output

import h5py
import os

from adapt_utils.steady.solver import AdaptiveSteadyProblem
from adapt_utils.steady.test_cases.point_discharge2d.options import PointDischarge2dOptions


# FIXME: Values don't quite agree with what we had previously
# TODO: Consider other discretisations and stabilisation

# Arrays etc.
num_levels = 5
di = create_directory(os.path.join(os.path.dirname(__file__), 'outputs/fixed_mesh/hdf5'))
qois = {'aligned': [], 'offset': []}
num_cells = []
qois_exact = {'aligned': [], 'offset': []}

# Loop over mesh hierarchy
for level in range(num_levels):
    op = PointDischarge2dOptions(level=level, aligned=True)
    tp = AdaptiveSteadyProblem(op)
    tp.solve_forward()

    num_cells.append(tp.mesh.num_cells())
    print_output("\nMesh {:d} in the hierarchy".format(level+1))
    print_output("    Number of elements  : {:d}".format(num_cells[-1]))
    qois['aligned'].append(tp.quantity_of_interest())
    print_output("    Aligned QoI: {:.5f}".format(qois['aligned'][-1]))
    qois_exact['aligned'].append(op.exact_qoi(tp.P1[0]))
    print_output("    (Exact     : {:.5f})".format(qois_exact['aligned'][-1]))

    op.__init__(level=level, aligned=False)
    qois['offset'].append(tp.quantity_of_interest())
    print_output("    Offset QoI : {:.5f}".format(qois['offset'][-1]))
    qois_exact['offset'].append(op.exact_qoi(tp.P1[0]))
    print_output("    (Exact     : {:.5f})".format(qois_exact['offset'][-1]))

# Print to screen
msg = "{:5d}  {:8d}  {:7.5f}  {:7.5f}"
for alignment in ('aligned', 'offset'):
    print_output("="*80 + alignment.capitalize())
    print_output("\nLevel  Elements       J{:d}  J{:d}exact".format(index, index))
    for level in range(num_levels):
        print_output(msg.format(level+1, num_cells[level], qois[alignment][level], qois_exact[alignment][level]))

# Store element count and QoI to HDF5
with h5py.File(os.path.join(di, 'qoi_dg_lf_sipg_aligned.h5'), 'w') as outfile:
    outfile.create_dataset('elements', data=num_cells)
    outfile.create_dataset('qoi', data=qois['aligned'])
    outfile.create_dataset('qoi_exact', data=qois_exact['aligned'])
with h5py.File(os.path.join(di, 'qoi_dg_lf_sipg_offset.h5'), 'w') as outfile:
    outfile.create_dataset('elements', data=num_cells)
    outfile.create_dataset('qoi', data=qois['offset'])
    outfile.create_dataset('qoi_exact', data=qois_exact['offset'])
