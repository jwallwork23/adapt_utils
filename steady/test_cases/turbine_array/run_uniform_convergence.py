import argparse
import h5py
import sys

from adapt_utils.steady.swe.turbine.solver import AdaptiveSteadyTurbineProblem
from adapt_utils.steady.test_cases.turbine_array.options import TurbineArrayOptions


# --- Parse arguments

parser = argparse.ArgumentParser()
parser.add_argument("-offset")
parser.add_argument('-offset', help="""
    Number of turbine diameters by which to offset turbines in y-direction.
    'Aligned' configuration given by offset=0, 'Offset' configuration given by offset=1.
    (Default 0)""")
parser.add_argument('-save_plex', help="Save DMPlex to HDF5 (default False)")
parser.add_argument('-debug', help="Toggle debugging mode (default False).")
parser.add_argument('-debug_mode', help="""
    Choose debugging mode from 'basic' and 'full' (default 'basic').""")
args = parser.parse_args()


# --- Set parameters

levels = 3
# levels = 5
kwargs = {
    'plot_pvd': False,
    'offset': int(args.offset or 0),
    'debug': bool(args.debug or False),
    'debug_mode': args.debug_mode or 'basic',
}

qois, num_cells, dofs = [], [], []
op = TurbineArrayOptions(level=levels, **kwargs)


# --- Loop over mesh hierarchy

for level in range(levels):
    op.default_mesh = op.hierarchy[level]
    tp = AdaptiveSteadyTurbineProblem(op)

    # Solve forward problem
    tp.solve_forward()

    # Store diagnostics
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

# --- Save to file

# Store QoI and element and DOF counts to HDF5
if bool(args.save_plex or False):
    with h5py.File('data/qoi_offset_{:d}.h5'.format(op.offset), 'w') as outfile:
        outfile.create_dataset('elements', data=num_cells)
        outfile.create_dataset('dofs', data=dofs)
        outfile.create_dataset('qoi', data=qois)
