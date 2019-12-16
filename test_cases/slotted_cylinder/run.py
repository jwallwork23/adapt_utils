from thetis import *
from firedrake.petsc import PETSc

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

approach = 'fixed_mesh' if args.approach is None else args.approach
adj = False if args.solve_adjoint is None else bool(args.solve_adjoint)
i = 0 if args.init_res is None else int(args.init_res)

# Create parameter class
n = 2**i
mesh = UnitSquareMesh(40*n, 40*n)
op = LeVequeOptions(approach=approach, shape=0)
#op.dt = pi/(300*n)
op.dt = 2*pi/(100*n)
# TODO: adaptive TS?
op.dt_per_export = 10*n
op.dt_per_remesh = 10*n
P1DG = FunctionSpace(mesh, "DG", 1)
op.set_qoi_kernel(P1DG)

# Adaptation parameters
op.normalisation = 'error'
op.num_adapt = 3 if args.num_adapt is None else int(args.num_adapt)
op.norm_order = 1
desired_error = 0.1 if args.desired_error is None else float(args.desired_error)
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
PETSc.Sys.Print("\nElement count : %d" % tp.mesh.num_cells())

# QoIs
PETSc.Sys.Print("\nGaussian\n")
PETSc.Sys.Print("Analytic QoI  : %.8e" % op.exact_qoi())
PETSc.Sys.Print("Quadrature QoI: %.8e" % op.quadrature_qoi(tp.P0))
PETSc.Sys.Print("Calculated QoI: %.8e" % tp.quantity_of_interest())
op.__init__(approach=approach, shape=1)
tp.get_qoi_kernel()
PETSc.Sys.Print("\nCone\n")
PETSc.Sys.Print("Analytic QoI  : %.8e" % op.exact_qoi())
PETSc.Sys.Print("Quadrature QoI: %.8e" % op.quadrature_qoi(tp.P0))
PETSc.Sys.Print("Calculated QoI: %.8e" % tp.quantity_of_interest())
op.__init__(approach=approach, shape=2)
tp.get_qoi_kernel()
PETSc.Sys.Print("\nSlotted Cylinder\n")
PETSc.Sys.Print("Analytic QoI  : %.8e" % op.exact_qoi())
PETSc.Sys.Print("Quadrature QoI: %.8e" % op.quadrature_qoi(tp.P0))
PETSc.Sys.Print("Calculated QoI: %.8e" % tp.quantity_of_interest())

# Relative Lp errors
if approach != 'fixed_mesh':
    op.set_initial_condition(tp.P1DG)
L1_err, L2_err, L_inf_err = op.lp_errors(tp.solution)
PETSc.Sys.Print("\nLp errors")  # FIXME: Are these computed properly? They seem quite large...
PETSc.Sys.Print("Relative L1 error      : %.8e" % L1_err)
PETSc.Sys.Print("Relative L2 error      : %.8e" % L2_err)
PETSc.Sys.Print("Relative L_inf error   : %.8e" % L_inf_err)
