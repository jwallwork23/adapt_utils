import h5py
import argparse

from adapt_utils.test_cases.steady_turbine.options import *
from adapt_utils.swe.turbine.solver import *


parser = argparse.ArgumentParser()
parser.add_argument("-offset")
args = parser.parse_args()

kwargs = {
    'plot_pvd': False,
    'debug': True,
    'outer_iterations': 4,
    # 'outer_iterations': 5,
    'offset' : int(args.offset or 0),
}

# Setup
op = Steady2TurbineOptions(**kwargs)
tp = SteadyTurbineProblem(op, levels=op.outer_iterations-1)
qois, num_cells, dofs = [], [], []

# Loop over mesh hierarchy
for level in range(op.outer_iterations):
    tp.solve()
    num_cells.append(tp.num_cells[0])
    dofs.append(sum(tp.V.dof_count))
    qois.append(tp.quantity_of_interest())
    op.print_debug("\nMesh {:d} in the hierarchy, offset = {:d}".format(level+1, op.offset))
    op.print_debug("    Number of elements  : {:d}".format(num_cells[-1]))
    op.print_debug("    Number of dofs      : {:d}".format(dofs[-1]))
    op.print_debug("    Power output        : {:.4f} kW".format(qois[-1]/1000.0))
    if level < op.outer_iterations-1:
        tp = tp.tp_enriched

# Store element count and QoI to HDF5
outfile = h5py.File('outputs/fixed_mesh/hdf5/qoi_offset_{:d}.h5'.format(op.offset), 'w')
outfile.create_dataset('elements', data=num_cells)
outfile.create_dataset('dofs', data=dofs)
outfile.create_dataset('qoi', data=qois)
outfile.close()

# Print to screen
op.print_debug("="*80 + "\nLevel  Elements     DOFs        J{:d}".format(op.offset))
msg = "{:5d}  {:8d}  {:7d}  {:6.4f}kW"
for i in range(op.outer_iterations):
    op.print_debug(msg.format(i+1, num_cells[i], dofs[i], qois[i]/1000.0))
