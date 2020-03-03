from thetis import *

from adapt_utils.test_cases.slotted_cylinder.options import LeVequeOptions
from adapt_utils.tracer.solver2d import UnsteadyTracerProblem2d


# Create parameter class
i = 0
kwargs = {
    'approach': 'ale',
    'n': 2**i,
    'family': 'CG',
    'stabilisation': 'SUPG',
    'num_adapt': 1,
    'nonlinear_method': 'relaxation',
}
op = LeVequeOptions(shape=0, **kwargs)

# Run model
tp = UnsteadyTracerProblem2d(op=op)
tp.setup_solver_forward()
tp.solve_ale()  # FIXME
print_output("\nElement count : {:d}".format(tp.mesh.num_cells()))

# QoIs
print_output("\nGaussian\n")
print_output("Analytic QoI  : {:.8e}".format(op.exact_qoi()))
print_output("Quadrature QoI: {:.8e}".format(op.quadrature_qoi(tp.P0)))
print_output("Calculated QoI: {:.8e}".format(tp.quantity_of_interest()))
op.__init__(shape=1, **kwargs)
tp.get_qoi_kernel()
print_output("\nCone\n")
print_output("Analytic QoI  : {:.8e}".format(op.exact_qoi()))
print_output("Quadrature QoI: {:.8e}".format(op.quadrature_qoi(tp.P0)))
print_output("Calculated QoI: {:.8e}".format(tp.quantity_of_interest()))
op.__init__(shape=2, **kwargs)
tp.get_qoi_kernel()
print_output("\nSlotted Cylinder\n")
print_output("Analytic QoI  : {:.8e}".format(op.exact_qoi()))
print_output("Quadrature QoI: {:.8e}".format(op.quadrature_qoi(tp.P0)))
print_output("Calculated QoI: {:.8e}".format(tp.quantity_of_interest()))
