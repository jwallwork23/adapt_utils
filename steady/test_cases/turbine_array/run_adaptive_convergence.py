import argparse
import h5py

from adapt_utils.steady.test_cases.turbine_array.options import *
from adapt_utils.steady.swe.turbine.solver import SteadyTurbineProblem

parser = argparse.ArgumentParser()
parser.add_argument('-approach', help="Mesh adaptation strategy")
args = parser.parse_args()
approach = args.approach or 'carpio'

# Parameters
kwargs = {
    'approach': approach,
    'debug': True,

    # Adaptation parameters
    'target': 800.0 if 'isotropic' in approach else 400.0,
    'adapt_field': 'all_int',
    'normalisation': 'complexity',
    'convergence_rate': 1,
    'norm_order': None,
    'h_max': 500.0,

    # Optimisation parameters
    'max_adapt': 35,  # Maximum iterations
    # 'element_rtol': 0.002,
    'element_rtol': 0.001,
    'outer_iterations': 7,
    'target_base': 2,
}

outstrs = {0: [], 1: []}
for offset in (0, 1):

    # Run adaptation loop
    op = TurbineArrayOptions(offset=offset, **kwargs)
    op.set_all_rtols(op.element_rtol)
    tp = SteadyTurbineProblem(op, discrete_adjoint=True, levels=1)
    tp.outer_adaptation_loop()

    # Store element count and QoI to HDF5
    outfile = h5py.File('outputs/{:s}/hdf5/qoi_offset_{:d}.h5'.format(op.approach, op.offset), 'w')
    outfile.create_dataset('elements', data=tp.outer_num_cells)
    outfile.create_dataset('dofs', data=tp.outer_dofs)
    outfile.create_dataset('qoi', data=tp.outer_qois)
    outfile.close()

    for i in range(op.outer_iterations):
        outstrs[offset].append("{:5d}  {:8d}  {:7d}  {:6.4f}kW".format(i+1, tp.outer_num_cells[i], tp.outer_dofs[i], tp.outer_qois[i]/1000.0))

# Print results to screen
for offset in (0, 1):
    op.print_debug("="*80 + "\nLevel  Elements     DOFs        J{:d}".format(offset))
    for i in range(op.outer_iterations):
        op.print_debug(outstrs[offset][i])
