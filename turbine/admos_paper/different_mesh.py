from thetis import *
from adapt_utils.turbine.options import *
from adapt_utils.turbine.solver import SteadyTurbineProblem
from adapt_utils.adapt.metric import *
from adapt_utils.adapt.p0_metric import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-approach', help="Mesh adaptation strategy")
parser.add_argument('-target', help="Scaling parameter for metric")
parser.add_argument('-offset', help="Toggle offset or aligned turbine configurations")
parser.add_argument('-adapt_field', help="Field(s) to compute Hessian w.r.t.")
parser.add_argument('-initial_mesh', help="Resolution of initial mesh")
parser.add_argument('-rtol', help="Relative tolerance for adaptation algorithm termination")
args = parser.parse_args()

op2.init(log_level=INFO)

# Problem setup
level = 'xcoarse' if args.initial_mesh is None else args.initial_mesh
offset = False if args.offset is None else bool(args.offset)
label = level + '_2'
if offset:
    label += '_offset'
sol = None

# Adaptation parameters
approach = 'carpio' if args.approach is None else args.approach
op = Steady2TurbineOffsetOptions(approach) if offset else Steady2TurbineOptions(approach)
op.timestepper = 'SteadyState'
op.target = 1000 if args.target is None else float(args.target)
op.family = 'dg-cg'  # NOTE: dg-cg seems to work better with various adapt_field choices
op.adapt_field = 'all_int' if args.adapt_field is None else args.adapt_field
op.normalisation = 'complexity'
# op.normalisation = 'error'
op.convergence_rate = 1
op.norm_order = None
op.h_max = 500.0

# Termination criteria
op.num_adapt = 35  # Maximum iterations
rtol = 0.002 if args.rtol is None else float(args.rtol)
qoi_rtol = rtol
element_rtol = rtol
estimator_rtol = rtol

# Set initial mesh hierarchy
if level == 'uniform':
    mesh = op.default_mesh
else:
    mesh = Mesh('{:s}_turbine.msh'.format(label))
mh = MeshHierarchy(mesh, 1)
print("Number of elements: {:d}".format(mesh.num_cells()))

# Run adaptation loop
for i in range(op.num_adapt):
    print("Step {:d}".format(i))

    print("Solving in base space")
    tp = SteadyTurbineProblem(mesh=mh[-2], discrete_adjoint=True, op=op, prev_solution=sol)
    tp.solve()

    print("Quantity of interest: {:.4e}".format(tp.qoi))
    if approach == 'fixed_mesh':
        break

    if i > 0 and np.abs(tp.qoi - qoi_old) < qoi_rtol*qoi_old:
        print("Number of elements: ", tp.mesh.num_cells())
        print("Number of dofs: ", sum(tp.V.dof_count))
        print("Converged quantity of interest!")
        break
    if i > 0 and np.abs(tp.mesh.num_cells() - num_cells_old) < element_rtol*num_cells_old:
        print("Number of elements: ", tp.mesh.num_cells())
        print("Number of dofs: ", sum(tp.V.dof_count))
        print("Converged number of mesh elements!")
        break
    if i == op.num_adapt-1:
        print("Did not converge!")
        break

    if approach != 'uniform' or 'hessian' in approach:
        tp.solve_adjoint()

    if approach == 'fixed_mesh_adjoint':
        break

    if approach != 'uniform' or 'hessian' in approach:

        print("Solving in refined space")
        tp_ho = SteadyTurbineProblem(mesh=mh[-1], discrete_adjoint=True, op=op, prev_solution=tp.solution, hierarchy=True)
        tp_ho.setup_solver()
        proj = Function(tp_ho.V)
        prolong(tp.solution, proj)
        tp_ho.lhs = replace(tp_ho.lhs, {tp_ho.solution: proj})
        tp_ho.solution = proj
        tp_ho.solve_adjoint()

        # Approximate adjoint error (in refined space)
        adj_proj = Function(tp_ho.V)
        prolong(tp.adjoint_solution, adj_proj)
        adj_ho_u, adj_ho_eta = tp_ho.adjoint_solution.split()
        adj_proj_u, adj_proj_eta = adj_proj.split()
        adj_ho_u -= adj_proj_u
        adj_ho_eta -= adj_proj_eta

        # Make sure everything is defined on the right mesh
        tp_ho.set_fields()
        tp_ho.boundary_conditions = op.set_bcs(tp.V)

        # Indicate error in enriched space and then project (average) down to base space
        tp_ho.get_strong_residual(proj, tp_ho.adjoint_solution)
        tp_ho.get_flux_terms(proj, tp_ho.adjoint_solution)
        tp_ho.indicator = interpolate(abs(tp_ho.indicators['dwr_cell'] + tp_ho.indicators['dwr_flux']), tp_ho.P0)
        tp.indicator = project(tp_ho.indicator, tp.P0)  # This is equivalent to averaging
        tp.estimators['dwr'] = tp.indicator.vector().sum()

        # Compute metric
        if tp.approach == 'carpio_isotropic':
            amd = AnisotropicMetricDriver(tp.mesh, indicator=tp.indicator, op=tp.op)
            amd.get_isotropic_metric()
        elif tp.approach == 'carpio':
            tp.get_hessian_metric(noscale=True)
            amd = AnisotropicMetricDriver(tp.mesh, hessian=tp.M, indicator=tp.indicator, op=tp.op)
            amd.get_anisotropic_metric()
        else:
            raise NotImplementedError
        tp.M = amd.p1metric
    else:
        tp.indicate_error()
    print("Error estimator: {:.4e}".format(tp.estimators['dwr']))
    if i > 0 and np.abs(tp.estimators['dwr'] - estimator_old) < estimator_rtol*estimator_old:
        print("Number of elements: ", tp.mesh.num_cells())
        print("Number of dofs: ", sum(tp.V.dof_count))
        print("Converged error estimator!")
        break

    # Store QoI, number of cells and error estimator for the next step
    qoi_old = tp.qoi
    num_cells_old = tp.mesh.num_cells()
    estimator_old = tp.estimators['dwr']

    # Adapt mesh and create new mesh hierarchy
    tp.adapt_mesh()
    mh = MeshHierarchy(tp.mesh, 1)
    sol = tp.solution

# Print summary to screen
print('\n'+80*'#')
print('SUMMARY')
print(80*'#' + '\n')
print('Approach:             {:s}'.format(op.approach))
print('Target:               {:.1f}'.format(op.target))
print("Number of elements:   {:d}".format(tp.mesh.num_cells()))
print("Number of dofs:       {:d}".format(sum(tp.V.dof_count)))
print("Quantity of interest: {:.4e}".format(tp.qoi))
print(80*'#')
