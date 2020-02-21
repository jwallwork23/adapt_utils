import h5py

from adapt_utils.test_cases.steady_turbine.options import *
from adapt_utils.swe.turbine.solver import *


kwargs = {
    'plot_pvd': False,
    'debug': True,
    'outer_iterations': 4,  # TODO: 5
}

data = {}
for offset in (0, 1):

    # Setup
    op = Steady2TurbineOptions(offset=offset, **kwargs)
    tp = SteadyTurbineProblem(op, levels=op.outer_iterations-1)
    data[offset] = {'qois': [], 'num_cells': [], 'dofs': []}
    d = data[offset]

    # Loop over mesh hierarchy
    for level in range(op.outer_iterations):
        tp.solve()
        d['num_cells'].append(tp.num_cells[0])
        d['dofs'].append(sum(tp.V.dof_count))
        d['qois'].append(tp.quantity_of_interest())
        op.print_debug("\nMesh {:d} in the hierarchy, offset = {:d}".format(level+1, offset))
        op.print_debug("    Number of elements  : {:d}".format(tp.num_cells[0]))
        op.print_debug("    Number of dofs      : {:d}".format(data[offset]['dofs'][-1]))
        op.print_debug("    Power output        : {:.4f} kW".format(data[offset]['qois'][-1]/1000.0))
        if level < op.outer_iterations-1:
            tp = tp.tp_enriched

    # Store element count and QoI to HDF5
    outfile = h5py.File('outputs/fixed_mesh/hdf5/qoi_offset_{:d}.h5'.format(offset), 'w')
    outfile.create_dataset('elements', data=d['num_cells'])
    outfile.create_dataset('dofs', data=d['dofs'])
    outfile.create_dataset('qoi', data=d['qois'])
    outfile.close()

# Print results to screen
for offset in (0, 1):
    op.print_debug("="*80 + "\nLevel  Elements     DOFs        J{:d}".format(offset))
    msg = "{:5d}  {:8d}  {:7d}  {:6.4f}kW"
    for i in range(op.outer_iterations):
        op.print_debug(msg.format(i+1, d['num_cells'][i], d['dofs'][i], d['qois'][i]/1000.0))
