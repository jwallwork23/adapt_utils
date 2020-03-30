from thetis import print_output, pi

import argparse

from adapt_utils.test_cases.slotted_cylinder.options import LeVequeOptions
from adapt_utils.tracer.solver2d_thetis import UnsteadyTracerProblem2d_Thetis


# Initialise
parser = argparse.ArgumentParser()
parser.add_argument("-approach", help="Choose adaptive approach")
parser.add_argument("-solve_adjoint", help="Solve adjoint problem and save to hdf5")
parser.add_argument("-init_res", help="Initial resolution i, i.e. 2^i elements in each direction")
parser.add_argument("-num_adapt")
parser.add_argument("-desired_error")
args = parser.parse_args()

approach = args.approach or 'fixed_mesh'
adj = bool(args.solve_adjoint or False)
i = int(args.init_res or 0)
n = 2**i
desired_error = float(args.desired_error or 0.1)

# Create parameter class
kwargs = {

    # QoI (Slotted cylinder)
    'shape': 0,

    # Spatial discretisation
    'family': 'dg',
    'stabilisation': 'no',

    # Temporal discretisation
    'dt': pi/(300*n),
    # 'dt': 2*pi/(100*n),
    'dt_per_export': 10*n,
    'dt_per_remesh': 10*n,

    # Adaptation parameters
    'approach': approach,
    'normalisation': 'error',
    'num_adapt': int(args.num_adapt or 3),
    'norm_order': 1,
    'desired_error': desired_error,
    'target': 1.0/desired_error,
}
op = LeVequeOptions(**kwargs)

# Run model
tp = UnsteadyTracerProblem2d_Thetis(op=op)
if approach == 'hessian':
    tp.adapt_mesh()
    tp.set_initial_condition()
    tp.set_fields()
elif adj:
    op.approach = 'fixed_mesh'
    tp_ = UnsteadyTracerProblem2d_Thetis(op=op)
    tp_.solve()
    tp_.solve_adjoint()
    op.approach = args.approach
# TODO: initial solves and adapts to get good initial mesh
tp.solve()

# Print output data
rule = 73*'='
print_output("\n  Shape            Analytic QoI    Quadrature QoI  Calculated QoI  Error")
print_output(rule)
for shape, name in zip(range(3), ('Gaussian', 'Cone', 'Slotted cylinder')):
    op.set_region_of_interest(shape=shape)
    tp.get_qoi_kernel()
    exact = op.exact_qoi()
    qoi = tp.quantity_of_interest()
    qois = (exact, op.quadrature_qoi(tp.P0), qoi, 100.0*abs(1.0 - qoi/exact))
    print_output("{:1d} {:16s} {:14.8e}  {:14.8e}  {:14.8e}  {:5.2f}%".format(shape, name, *qois))
print_output(rule)
approach = "'" + tp.approach.replace('_', ' ').capitalize() + "'"
print_output(19*' ' + "{:14s}  {:14s}  {:14s}  {:6d}".format('Approach', approach,
                                                             'Element count', tp.num_cells[-1]))
