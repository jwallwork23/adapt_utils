from firedrake import *

import argparse
import h5py
import os
from time import perf_counter

from adapt_utils.adapt.metric import *
from adapt_utils.params import lu_params
from adapt_utils.norms import *
from adapt_utils.unsteady.solver import AdaptiveProblem
from adapt_utils.unsteady.test_cases.bubble_shear.options import BubbleOptions


# --- Parse arguments

parser = argparse.ArgumentParser()
parser.add_argument("n", help="Mesh resolution level")

parser.add_argument("-limiters", help="Toggle limiters for tracer equation")
parser.add_argument("-stabilisation", help="Stabilisation method")
parser.add_argument("-anisotropic_stabilisation", help="Toggle anisotropic stabilisation")
parser.add_argument("-family", help="Choose finite element from 'cg' and 'dg'")

parser.add_argument("-num_meshes", help="Number of meshes in the sequence")
parser.add_argument("-num_adapt", help="Number of adaptation steps")
parser.add_argument("-metric_advection", help="Apply metric advection")

parser.add_argument("-plot_pvd", help="Write solution field to .pvd")
parser.add_argument("-debug", help="Toggle debugging mode.")
args = parser.parse_args()


# --- Set parameters

n = int(args.n)
metric_advection = bool(args.metric_advection or False)
plot_pvd = bool(args.plot_pvd or False)
kwargs = {

    # Solver
    'tracer_family': args.family or 'cg',
    'stabilisation_tracer': args.stabilisation or 'supg',
    'anisotropic_stabilisation': False if args.anisotropic_stabilisation == "0" else True,
    'use_automatic_sipg_parameter': False,  # We have an inviscid problem
    'use_limiter_for_tracers': bool(args.limiters or False),

    # Mesh adaptation
    'approach': 'hessian',
    'num_meshes': 1,
    'num_adapt': int(args.num_adapt or 3 if metric_advection else 0),
    'target': 1000*2**n,
    'norm_order': 1,
    'normalisation': 'complexity',

    # I/0 and debugging
    'plot_pvd': False,
    'debug': bool(args.debug or False),
}
op = BubbleOptions(approach='hessian', n=1)
op.update(kwargs)
op.dt = 0.01*0.5**n


# --- Solve the tracer transport problem

num_meshes = int(args.num_meshes or 50)
op.end_time /= num_meshes
dt_per_mesh = int(op.end_time/op.dt)
end_time = op.end_time
dtc = Constant(op.dt)
if metric_advection:
    op.di = os.path.join(op.di, 'metric_advection')
else:
    op.di = os.path.join(op.di, 'on_the_fly')
if plot_pvd:
    tracer_file = File(os.path.join(op.di, 'tracer.pvd'))
theta = Constant(0.5)

tic = perf_counter()
tp = AdaptiveProblem(op)
for i in range(op.num_adapt):
    print("INITIAL MESH STEP {:d}".format(i))
    tp.set_initial_condition()
    c = tp.fwd_solution_tracer
    M = steady_metric(c, V=tp.P1_ten[0], normalise=True, enforce_constraints=True, op=op)
    tp = AdaptiveProblem(op, meshes=adapt(tp.mesh, M))
tp.set_initial_condition()
dofs = []
num_cells = []
for i in range(num_meshes):
    V = tp.P1_ten[0]
    print("MESH STEP {:d}".format(i))

    # Construct static metric
    c = tp.fwd_solution_tracer
    M_ = steady_metric(c, V=V, normalise=True, enforce_constraints=True, op=op)
    M = Function(M_, name="Steady metric")

    # Apply metric advection
    if metric_advection:
        M_int = Function(M_, name='Intersected metric')
        # print(norm(M))
        # metric_file = File('outputs/hessian/metric.pvd')
        # metric_file.write(M_int)
        coords = V.mesh().coordinates
        trial, test = TrialFunction(V), TestFunction(V)
        u = Function(tp.P1_vec[0])
        u_ = Function(tp.P1_vec[0])
        a = inner(trial, test)*dx + theta*dtc*inner(dot(u, nabla_grad(trial)), test)*dx
        L = inner(M_, test)*dx - (1-theta)*dtc*inner(dot(u_, nabla_grad(M_)), test)*dx
        bc = DirichletBC(V, [[1/op.h_max**2, 0], [0, 1/op.h_max**2]], 'on_boundary')
        prob = LinearVariationalProblem(a, L, M, bcs=bc)
        solver = LinearVariationalSolver(prob, solver_parameters=lu_params)
        t = 0.0
        while t < op.end_time - 1.0e-05:
            u_.assign(op.get_velocity(coords, t))
            u.assign(op.get_velocity(coords, t+op.dt))
            solver.solve()
            M_int = metric_intersection(M_int, M)
            # metric_file.write(M_int)
            # print(norm(M))
            M_.assign(M)
            t += op.dt
        M.assign(M_int)

    def export_func():
        if plot_pvd:
            tracer_file.write(tp.fwd_solution_tracer)

    # Solve
    tp = AdaptiveProblem(op, meshes=adapt(tp.mesh, M))
    num_cells.append(tp.mesh.num_cells())
    dofs.append(tp.Q[0].dof_count)
    tp.fwd_solution_tracer.project(c)
    tp.setup_solver_forward_step(0)
    tp.iteration = i*dt_per_mesh
    tp.simulation_time = i*op.dt*dt_per_mesh
    if plot_pvd:
        tracer_file._topology = None
    # tp.solve_forward_step(0, export_func=export_func, restarted=True)
    tp.solve_forward_step(0, restarted=True)
    op.end_time += end_time


# --- Assess error

times = [perf_counter() - tic]
print("CPU time: {:.2f}s".format(times[0]))
final_sol = tp.fwd_solution_tracer.copy(deepcopy=True)
final_l1_norm = norm(final_sol, norm_type='L1')
final_l2_norm = norm(final_sol, norm_type='L2')
tp.set_initial_condition()
init_sol = tp.fwd_solution_tracer.copy(deepcopy=True)
init_l1_norm = norm(init_sol, norm_type='L1')
init_l2_norm = norm(init_sol, norm_type='L2')
abs_l2_error = errornorm(init_sol, final_sol, norm_type='L2')
cons_error = [100*abs(init_l1_norm-final_l1_norm)/init_l1_norm]
l2_error = [100*abs_l2_error/init_l2_norm]
print("Conservation error: {:.2f}%".format(l2_error[0]))
print("Relative L2 error:  {:.2f}%".format(cons_error[0]))

# Save to HDF5
with h5py.File(os.path.join(op.di, 'convergence_{:d}.h5'.format(n)), 'w') as outfile:
    outfile.create_dataset('iterations', data=[1])
    outfile.create_dataset('elements', data=num_cells)
    outfile.create_dataset('dofs', data=dofs)
    outfile.create_dataset('time', data=times)
    outfile.create_dataset('l2_error', data=l2_error)
    outfile.create_dataset('cons_error', data=cons_error)
