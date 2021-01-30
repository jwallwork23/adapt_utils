import argparse
import h5py
import os

from adapt_utils.io import create_directory
from adapt_utils.steady.test_cases.turbine_array.options import *
from adapt_utils.swe.turbine.solver import AdaptiveSteadyTurbineProblem


# --- Parse arguments

parser = argparse.ArgumentParser()
parser.add_argument('-approach', help="Mesh adaptation strategy")
parser.add_argument('-enrichment_method', help="Defaults to GE_h")
parser.add_argument('-solve_enriched_forward', help="Solve or prolong forward")
args = parser.parse_args()
approach = args.approach or 'isotropic_dwr'


# --- Set parameters

kwargs = {
    'approach': approach,

    # Adaptation parameters
    'enrichment_method': args.enrichment_method or 'GE_h',
    'solve_enriched_forward': bool(args.solve_enriched_forward or False),
    'target': 800.0,
    'adapt_field': 'all_int',
    'normalisation': 'complexity',
    'convergence_rate': 1,
    'norm_order': None,  # i.e. infinity norm
    'h_min': 1.0e-05,
    'h_max': 5.0e+02,

    # Optimisation parameters
    'max_adapt': 35,  # Maximum iterations
    # 'element_rtol': 0.002,
    'element_rtol': 0.001,
    'outer_iterations': 7,
    'target_base': 2,
}
discrete_turbines = True
# discrete_turbines = False


# --- Run to convergence

outstrs = {0: [], 1: []}
for offset in (0, 1):

    # Run adaptation loop
    num_cells, dofs, qois = [], [], []
    for i in range(kwargs['outer_iterations']):
        op = TurbineArrayOptions(offset=offset, **kwargs)
        op.set_all_rtols(op.element_rtol)
        op.target = op.target*op.target_base**i
        method = op.enrichment_method
        if method not in ('DQ', 'PR'):
            method += '_solve' if op.solve_enriched_forward else '_prolong'
        tp = AdaptiveSteadyTurbineProblem(op, discrete_adjoint=True, discrete_turbines=discrete_turbines)
        tp.run()
        num_cells.append(tp.num_cells[-1][0])
        dofs.append(sum(tp.V[0].dof_count))
        qois.append(tp.qoi*1.030e-03)

    # Store element count and QoI to HDF5
    di = create_directory(os.path.join(os.path.dirname(__file__), 'outputs', op.approach, 'hdf5'))
    outfile = h5py.File(os.path.join(di, 'qoi_offset_{:d}_{:s}.h5'.format(op.offset, method)), 'w')
    outfile.create_dataset('elements', data=num_cells)
    outfile.create_dataset('dofs', data=dofs)
    outfile.create_dataset('qoi', data=qois)
    outfile.close()

    for i in range(op.outer_iterations):
        outstrs[offset].append("{:5d}  {:8d}  {:7d}  {:6.4f}MW".format(i+1, num_cells[i], dofs[i], qois[i]))

# Print results to screen
for offset in (0, 1):
    op.print_debug("="*80 + "\nLevel  Elements     DOFs        J{:d}".format(offset))
    for i in range(op.outer_iterations):
        op.print_debug(outstrs[offset][i])
