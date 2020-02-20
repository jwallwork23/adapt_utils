from thetis import *
from adapt_utils.test_cases.steady_turbine.options import *
from adapt_utils.swe.turbine.solver import SteadyTurbineProblem

import argparse
import h5py

parser = argparse.ArgumentParser()
parser.add_argument('-approach', help="Mesh adaptation strategy")
args = parser.parse_args()

# Parameters
kwargs = {
    'approach': args.approach or 'carpio',
    'target': 100.0,
    'debug': True,
    'adapt_field': 'all_int',
    'normalisation': 'complexity',
    'convergence_rate': 1,
    'norm_order': None,
    'h_max': 500.0,
    'num_adapt': 35,  # Maximum iterations
    'element_rtol': 0.002,
    'outer_iterations': 3,
}

for offset in (0, 1)

    # Run adaptation loop
    op = Steady2TurbineOptions(offset=offset, **kwargs)
    tp = SteadyTurbineProblem(op, discrete_adjoint=True, levels=1)
    tp.outer_adaptation_loop()

    # Store element count and QoI to HDF5
    outfile = h5py.File('outputs/{:s}/hdf5/qoi_offset_{:d}.h5'.format(op.approach, op.offset), 'w')
    outfile.create_dataset('elements', data=tp.outer_num_cells)
    outfile.create_dataset('dofs', data=tp.outer_dofs)
    outfile.create_dataset('qoi', data=tp.outer_qois)
    outfile.close()

# Print results to screen
for offset in (0, 1):
    print_output("="*80 + "\nLevel  Elements     DOFs        J{:d}".format(offset))
    for i in range(op.outer_iterations):
        print_output("{:5d}  {:8d}  {:7d}  {:6.4f}kW".format(i+1, tp.outer_num_cells[i], tp.outer_dofs[i], tp.outer_qois[i]))

# TODO: Format output as table
# TODO: Plot QoI convergence with nice formatting. (Replacing jupyter notebook.)
