from thetis import print_output

import argparse

from adapt_utils.test_cases.slotted_cylinder.options import LeVequeOptions
from adapt_utils.tracer.solver2d import UnsteadyTracerProblem2d


parser = argparse.ArgumentParser()
parser.add_argument("-initial_resolution", help="Initial mesh resolution")
parser.add_argument("-velocity", help="Type of prescribed velocity. Choose from {'zero', 'fluid'}")
parser.add_argument("-family", help="Choose from {'cg', 'dg'}.")
args = parser.parse_args()


# Create parameter class
i = int(args.initial_resolution or 0)
family = args.family or 'dg'
velocity = args.velocity or 'zero'
kwargs = {
    'approach': 'ale',
    'n': 2**i,
    'family': family,
    'stabilisation': 'SUPG' if family in ('CG', 'cg', 'Lagrange') else 'no',
    'num_adapt': 1,
    'nonlinear_method': 'relaxation',
    'prescribed_velocity': velocity,
}
op = LeVequeOptions(shape=0, **kwargs)

# Run model
tp = UnsteadyTracerProblem2d(op=op)
tp.setup_solver_forward()
tp.solve_ale(check_inverted=velocity != 'zero')  # FIXME: Mesh tangling issue
print_output("\nElement count : {:d}".format(tp.mesh.num_cells()))

# QoIs
for i, shape in zip(range(3), ('Gaussian', 'Cone', 'Slotted cylinder')):
    print_output("\n{:s}\n".format(shape))
    op.__init__(shape=i, **kwargs)
    tp.get_qoi_kernel()
    exact = op.exact_qoi()
    quadrature = op.quadrature_qoi(tp.P0)
    approx = tp.quantity_of_interest()
    print_output("Analytic QoI    : {:.8e}".format(exact))
    print_output("Calculated QoI  : {:.8e}".format(approx))
    print_output("Relative error  : {:.4f}%".format(abs(1.0-approx/exact)))
    print_output("(Quadrature QoI : {:.8e})".format(quadrature))
