from firedrake import *

from adapt_utils.test_cases.ale_advection.options import ALEAdvectionOptions
from adapt_utils.tracer.solver2d import UnsteadyTracerProblem2d
from adapt_utils.adapt.r import MeshMover


op = ALEAdvectionOptions(num_adapt=1, nonlinear_method='relaxation', prescribed_velocity='fluid')
tp = UnsteadyTracerProblem2d(op)

# Domain and function space
mesh = op.default_mesh

# Physical parameters
u = op.set_velocity(tp.P1_vec)
nu = op.set_diffusivity(tp.P1)
tp.solution_old = op.set_initial_condition(tp.P1)

# Functions
c = tp.solution
c_ = tp.solution_old
c_trial, c_test = TrialFunction(tp.P1), TestFunction(tp.P1)

# SUPG stabilisation
tau = tp.stabilisation_parameter
c_test = c_test + tau*dot(u, grad(c_test))

# Mesh movement object
mm = MeshMover(mesh, monitor_function=None, method='ale', op=op)

# Lagrangian derivative
Xdot = op.get_mesh_velocity()
Ld = lambda v: dot(grad(v), Xdot(mesh))

# PDE residual (using implicit midpoint rule timestepping)
dtc = Constant(op.dt)
f = lambda v: dot(u, grad(v))*c_test*dx + inner(nu*grad(v), grad(c_test))*dx
a = c_trial*c_test*dx
L = c_*c_test*dx
a += dtc*f(0.5*c_trial)
L -= dtc*f(0.5*c_)
a -= dtc*Ld(0.5*c_trial)*c_test*dx
L += dtc*Ld(0.5*c_)*c_test*dx

tp.setup_solver_forward()  # FIXME
# a = tp.lhs
# L = tp.rhs

# Timestepping loop
t = 0.0
tp.solution.assign(tp.solution_old)
outfile = tp.solution_file
outfile.write(tp.solution)
while t < op.end_time - 0.5*op.dt:
    print("t = {:.1f}s".format(t))

    # Solve mesh movement
    mm.adapt_ale()

    # Solve advection-diffusion PDE
    solve(a == L, tp.solution, solver_parameters=op.params)

    # Update
    tp.solution_old.assign(tp.solution)  # FIXME
    mesh.coordinates.assign(mm.x_new)
    t += op.dt
    outfile.write(tp.solution)
