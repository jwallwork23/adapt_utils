from thetis import *

import argparse

from adapt_utils.test_cases.slotted_cylinder.options import LeVequeOptions
from adapt_utils.tracer.solver2d_thetis import UnsteadyTracerProblem2d_Thetis


# initialise
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

# Create parameter class
n = 2**i
mesh = UnitSquareMesh(40*n, 40*n)
op = LeVequeOptions(approach=approach, shape=0, family='dg', stabilisation='no')
#op.dt = pi/(300*n)
op.dt = 2*pi/(100*n)
# TODO: adaptive TS?
op.dt_per_export = 10*n
op.dt_per_remesh = 10*n
P1DG = FunctionSpace(mesh, "DG", 1)
op.set_qoi_kernel(P1DG)

# Adaptation parameters
op.normalisation = 'error'
op.num_adapt = int(args.num_adapt or 3)
op.norm_order = 1
desired_error = float(args.desired_error or 0.1)
op.target = 1/desired_error

# Run model
tp = UnsteadyTracerProblem2d_Thetis(op=op, mesh=mesh)
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
print_output("\nElement count : {:d}".format(tp.mesh.num_cells()))

# QoIs
print_output("\nGaussian\n")
print_output("Analytic QoI  : {:.8e}".format(op.exact_qoi()))
print_output("Quadrature QoI: {:.8e}".format(op.quadrature_qoi(tp.P0)))
print_output("Calculated QoI: {:.8e}".format(tp.quantity_of_interest()))
op.__init__(approach=approach, shape=1)
tp.get_qoi_kernel()
print_output("\nCone\n")
print_output("Analytic QoI  : {:.8e}".format(op.exact_qoi()))
print_output("Quadrature QoI: {:.8e}".format(op.quadrature_qoi(tp.P0)))
print_output("Calculated QoI: {:.8e}".format(tp.quantity_of_interest()))
op.__init__(approach=approach, shape=2)
tp.get_qoi_kernel()
print_output("\nSlotted Cylinder\n")
print_output("Analytic QoI  : {:.8e}".format(op.exact_qoi()))
print_output("Quadrature QoI: {:.8e}".format(op.quadrature_qoi(tp.P0)))
print_output("Calculated QoI: {:.8e}".format(tp.quantity_of_interest()))
