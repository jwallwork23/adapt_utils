from thetis import *

import h5py

from adapt_utils.test_cases.steady_turbine.options import *
from adapt_utils.swe.turbine.solver import *


kwargs = {'plot_pvd': False, 'debug': True, 'outer_iterations': 5}

data = {}
for offset in (0, 1):

    # Setup
    op = Steady2TurbineOptions(offset=offset, **kwargs)
    tp = SteadyTurbineProblem(op, levels=op.outer_iterations-1)
    data[offset] = {'qois': [], 'num_cells': [], 'dofs': []}

    # Loop over mesh hierarchy
    for level in range(op.outer_iterations):
        tp.solve()
        data[offset]['num_cells'].append(tp.num_cells[0])
        data[offset]['dofs'].append(sum(tp.V.dof_count))
        data[offset]['qois'].append(tp.quantity_of_interest()/1000.0)
        op.print_debug("\nMesh {:d} in the hierarchy".format(level+1))
        op.print_debug("    Number of elements  : {:d}".format(tp.num_cells[0]))
        op.print_debug("    Number of dofs      : {:d}".format(data[offset]['dofs'][-1]))
        op.print_debug("    Power output        : {:.4f} kW".format(data[offset]['qois'][-1]))
        if level < op.outer_iterations-1:
            tp = tp.tp_enriched

    # Store element count and QoI to HDF5
    outfile = h5py.File('outputs/fixed_mesh/hdf5/qoi_offset_{:d}.h5'.format(offset), 'w')
    outfile.create_dataset('elements', data=data['num_cells'])
    outfile.create_dataset('dofs', data=data['dofs'])
    outfile.create_dataset('qoi', data=data['qois'])
    outfile.close()

# Print results to screen
for offset in (0, 1):
    print_output("="*80 + "\nLevel  Elements     DOFs        J{:d}".format(offset))
    for i in range(op.outer_iterations):
        print_output("{:5d}  {:8d}  {:7d}  {:6.4f}kW".format(i+1, data[offset]['num_cells'][i], data[offset]['dofs'][i], data[offset]['qois'][i]))
