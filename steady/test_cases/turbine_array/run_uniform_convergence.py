from thetis import create_directory, MeshHierarchy

import argparse
import h5py
import os

from adapt_utils.swe.turbine.solver import AdaptiveSteadyTurbineProblem
from adapt_utils.steady.test_cases.turbine_array.options import TurbineArrayOptions


# --- Parse arguments

parser = argparse.ArgumentParser()
parser.add_argument('-offset', help="""
    Number of turbine diameters by which to offset turbines in y-direction.
    'Aligned' configuration given by offset=0, 'Offset' configuration given by offset=1.
    (Default 0)""")
parser.add_argument('-debug', help="Toggle debugging mode (default False).")
parser.add_argument('-debug_mode', help="""
    Choose debugging mode from 'basic' and 'full' (default 'basic').""")
args = parser.parse_args()


# --- Set parameters

levels = 5
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
op = TurbineArrayOptions(**kwargs)
hierarchy = MeshHierarchy(op.default_mesh, levels)


# --- Loop over mesh hierarchy

for level in range(levels):
    op.default_mesh = hierarchy[level]
    di = 'uniform_level{:d}_offset{:d}'.format(level, offset)
    di = create_directory(os.path.join(op.di, di))
    tp = AdaptiveSteadyTurbineProblem(op, discrete_turbines=discrete_turbines, callback_dir=di)

    # Solve forward problem
    tp.solve_forward()

    # Store diagnostics
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

# --- Save to file

# Store QoI and element and DOF counts to HDF5
with h5py.File('data/qoi_offset_{:d}.h5'.format(op.offset), 'w') as outfile:
    outfile.create_dataset('elements', data=num_cells)
    outfile.create_dataset('dofs', data=dofs)
    outfile.create_dataset('qoi', data=qois)
