<<<<<<< HEAD
=======
from thetis import create_directory, print_output

>>>>>>> dfe1c0b3a34dfef1765835b64b574a69fe60dd9a
import argparse
import h5py
import os

<<<<<<< HEAD
from adapt_utils.io import create_directory, print_output
=======
>>>>>>> dfe1c0b3a34dfef1765835b64b574a69fe60dd9a
from adapt_utils.steady.solver import AdaptiveSteadyProblem
from adapt_utils.steady.test_cases.point_discharge2d.options import PointDischarge2dOptions


# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('family', help="Finite element family.")
parser.add_argument('-stabilisation', help="Stabilisation method to use.")
parser.add_argument('-anisotropic_stabilisation', help="Use anisotropic cell size measure?")
args = parser.parse_args()

# Get filenames
anisotropic_stabilisation = bool(args.anisotropic_stabilisation or False)
ext = args.family
assert ext in ('cg', 'dg')
if ext == 'dg':
    if args.stabilisation in ('lf', 'LF', 'lax_friedrichs'):
        ext += '_lf'
else:
    if args.stabilisation in ('su', 'SU'):
        ext += '_su'
    if args.stabilisation in ('supg', 'SUPG'):
        ext += '_supg'
    if anisotropic_stabilisation:
        ext += '_anisotropic'
fname = 'qoi_{:s}'.format(ext)

# Arrays etc.
num_levels = 5
di = create_directory(os.path.join(os.path.dirname(__file__), 'outputs', 'fixed_mesh', 'hdf5'))
qois = {'aligned': [], 'offset': []}
num_cells = []
<<<<<<< HEAD
dofs = []
=======
>>>>>>> dfe1c0b3a34dfef1765835b64b574a69fe60dd9a
qois_exact = {'aligned': [], 'offset': []}

# Loop over mesh hierarchy
for level in range(num_levels):

    # Solve PDE
    op = PointDischarge2dOptions(level=level, aligned=True)
    op.tracer_family = args.family
<<<<<<< HEAD
    stabilisation = args.stabilisation or 'supg'
    op.stabilisation_tracer = None if stabilisation == 'none' else stabilisation
    op.anisotropic_stabilisation = False if args.anisotropic_stabilisation == '0' else True
=======
    op.stabilisation = args.stabilisation
    op.anisotropic_stabilisation = anisotropic_stabilisation
>>>>>>> dfe1c0b3a34dfef1765835b64b574a69fe60dd9a
    op.use_automatic_sipg_parameter = args.family == 'dg'
    tp = AdaptiveSteadyProblem(op)
    tp.solve_forward()

    # Print element count
    num_cells.append(tp.mesh.num_cells())
<<<<<<< HEAD
    dofs.append(tp.mesh.num_vertices())
=======
>>>>>>> dfe1c0b3a34dfef1765835b64b574a69fe60dd9a
    print_output("\nMesh {:d} in the hierarchy".format(level+1))
    print_output("    Number of elements  : {:d}".format(num_cells[-1]))

    # Evaluate QoI in aligned case
    qois['aligned'].append(tp.quantity_of_interest())
    print_output("    Aligned QoI: {:.5f}".format(qois['aligned'][-1]))
    qois_exact['aligned'].append(op.analytical_qoi())
    print_output("    (Exact     : {:.5f})".format(qois_exact['aligned'][-1]))

    # Evaluate QoI in offset case
    op.__init__(level=level, aligned=False)
    qois['offset'].append(tp.quantity_of_interest())
    print_output("    Offset QoI : {:.5f}".format(qois['offset'][-1]))
    qois_exact['offset'].append(op.analytical_qoi())
    print_output("    (Exact     : {:.5f})".format(qois_exact['offset'][-1]))

# Print to screen
<<<<<<< HEAD
msg = "{:5d}  {:8d}  {:7.8f}  {:7.8f}"
for index, alignment in enumerate(('aligned', 'offset')):
    print_output("="*80 + "\n" + alignment.capitalize())
    print_output("\nLevel  Elements          J{:d}     J{:d}exact".format(index+1, index+1))
=======
msg = "{:5d}  {:8d}  {:7.5f}  {:7.5f}"
for index, alignment in enumerate(('aligned', 'offset')):
    print_output("="*80 + "\n" + alignment.capitalize())
    print_output("\nLevel  Elements       J{:d}  J{:d}exact".format(index+1, index+1))
>>>>>>> dfe1c0b3a34dfef1765835b64b574a69fe60dd9a
    for level in range(num_levels):
        print_output(msg.format(level+1, num_cells[level], qois[alignment][level], qois_exact[alignment][level]))

# Store element count and QoI to HDF5
for alignment in qois:
    with h5py.File(os.path.join(di, '{:s}_{:s}.h5'.format(fname, alignment)), 'w') as outfile:
        outfile.create_dataset('elements', data=num_cells)
<<<<<<< HEAD
        outfile.create_dataset('dofs', data=dofs)
=======
>>>>>>> dfe1c0b3a34dfef1765835b64b574a69fe60dd9a
        outfile.create_dataset('qoi', data=qois[alignment])
        outfile.create_dataset('qoi_exact', data=qois_exact[alignment])
