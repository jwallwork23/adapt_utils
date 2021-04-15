<<<<<<< HEAD
from firedrake import MeshHierarchy
=======
from thetis import create_directory
>>>>>>> dfe1c0b3a34dfef1765835b64b574a69fe60dd9a

import argparse
import h5py
import os

<<<<<<< HEAD
from adapt_utils.io import create_directory
from adapt_utils.swe.turbine.solver import AdaptiveSteadyTurbineProblem
=======
from adapt_utils.steady.swe.turbine.solver import AdaptiveSteadyTurbineProblem
>>>>>>> dfe1c0b3a34dfef1765835b64b574a69fe60dd9a
from adapt_utils.steady.test_cases.turbine_array.options import TurbineArrayOptions


# --- Parse arguments

parser = argparse.ArgumentParser()
parser.add_argument('-offset', help="""
    Number of turbine diameters by which to offset turbines in y-direction.
    'Aligned' configuration given by offset=0, 'Offset' configuration given by offset=1.
    (Default 0)""")
<<<<<<< HEAD
=======
parser.add_argument('-save_plex', help="Save DMPlex to HDF5 (default False)")
>>>>>>> dfe1c0b3a34dfef1765835b64b574a69fe60dd9a
parser.add_argument('-debug', help="Toggle debugging mode (default False).")
parser.add_argument('-debug_mode', help="""
    Choose debugging mode from 'basic' and 'full' (default 'basic').""")
args = parser.parse_args()


# --- Set parameters

<<<<<<< HEAD
levels = 5  # NOTE: PC_FAILED due to FACTOR_OUTMEMORY if levels = 6
=======
levels = 3
# levels = 5
>>>>>>> dfe1c0b3a34dfef1765835b64b574a69fe60dd9a
offset = int(args.offset or 0)
kwargs = {
    'plot_pvd': False,
    'offset': offset,
    'debug': bool(args.debug or False),
    'debug_mode': args.debug_mode or 'basic',
}

qois, num_cells, dofs = [], [], []
discrete_turbines = True
# discrete_turbines = False
<<<<<<< HEAD
op = TurbineArrayOptions(**kwargs)
hierarchy = MeshHierarchy(op.default_mesh, levels)
=======
op = TurbineArrayOptions(level=levels, **kwargs)
>>>>>>> dfe1c0b3a34dfef1765835b64b574a69fe60dd9a


# --- Loop over mesh hierarchy

for level in range(levels):
<<<<<<< HEAD
    op.default_mesh = hierarchy[level]
    di = 'uniform_level{:d}_offset{:d}'.format(level, offset)
    di = create_directory(os.path.join(op.di, di))
    tp = AdaptiveSteadyTurbineProblem(op, discrete_turbines=discrete_turbines, callback_dir=di)
=======
    op.default_mesh = op.hierarchy[level]
    callback_dir = 'uniform_level{:d}_offset{:d}'.format(level, offset)
    callback_dir = create_directory(os.path.join(op.di, callback_dir))
    tp = AdaptiveSteadyTurbineProblem(op, discrete_turbines=discrete_turbines, callback_dir=callback_dir)
>>>>>>> dfe1c0b3a34dfef1765835b64b574a69fe60dd9a

    # Solve forward problem
    tp.solve_forward()

    # Store diagnostics
<<<<<<< HEAD
    num_cells.append(tp.num_cells[-1][0])
    dofs.append(sum(tp.V[0].dof_count))
    J = tp.quantity_of_interest()*1.030e-03
    qois.append(J)
    op.print_debug("\nMesh {:d} in the hierarchy, offset = {:d}".format(level+1, op.offset))
    op.print_debug("    Number of elements  : {:d}".format(num_cells[-1]))
    op.print_debug("    Number of dofs      : {:d}".format(dofs[-1]))
    op.print_debug("    Power output        : {:.4f} MW".format(qois[-1]))

# Print to screen
op.print_debug("="*80 + "\nLevel  Elements     DOFs        J{:d}".format(op.offset))
msg = "{:5d}  {:8d}  {:7d}  {:6.4f}MW"
for i in range(levels):
    op.print_debug(msg.format(i+1, num_cells[i], dofs[i], qois[i]))
=======
    num_cells.append(tp.meshes[0].num_cells())
    dofs.append(sum(tp.V[0].dof_count))
    qois.append(tp.quantity_of_interest())
    op.print_debug("\nMesh {:d} in the hierarchy, offset = {:d}".format(level+1, op.offset))
    op.print_debug("    Number of elements  : {:d}".format(num_cells[-1]))
    op.print_debug("    Number of dofs      : {:d}".format(dofs[-1]))
    op.print_debug("    Power output        : {:.4f} kW".format(qois[-1]/1000.0))

# Print to screen
op.print_debug("="*80 + "\nLevel  Elements     DOFs        J{:d}".format(op.offset))
msg = "{:5d}  {:8d}  {:7d}  {:6.4f}kW"
for i in range(levels):
    op.print_debug(msg.format(i+1, num_cells[i], dofs[i], qois[i]/1000.0))
>>>>>>> dfe1c0b3a34dfef1765835b64b574a69fe60dd9a

# --- Save to file

# Store QoI and element and DOF counts to HDF5
<<<<<<< HEAD
di = create_directory('outputs/fixed_mesh/hdf5')
with h5py.File(os.path.join(di, 'qoi_offset_{:d}.h5'.format(op.offset)), 'w') as outfile:
    outfile.create_dataset('elements', data=num_cells)
    outfile.create_dataset('dofs', data=dofs)
    outfile.create_dataset('qoi', data=qois)
=======
if bool(args.save_plex or False):
    with h5py.File('data/qoi_offset_{:d}.h5'.format(op.offset), 'w') as outfile:
        outfile.create_dataset('elements', data=num_cells)
        outfile.create_dataset('dofs', data=dofs)
        outfile.create_dataset('qoi', data=qois)
>>>>>>> dfe1c0b3a34dfef1765835b64b574a69fe60dd9a
