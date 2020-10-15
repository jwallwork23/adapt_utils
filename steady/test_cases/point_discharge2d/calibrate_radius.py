from thetis import *
from firedrake_adjoint import *

import argparse
import numpy as np

from adapt_utils.steady.tracer.options import bessk0
from adapt_utils.steady.solver import AdaptiveSteadyProblem
from adapt_utils.steady.test_cases.point_discharge2d.options import PointDischarge2dOptions


# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('level')
parser.add_argument('family')
parser.add_argument('-stabilisation')
parser.add_argument('-use_automatic_sipg_parameter')
parser.add_argument('-test_consistency')
parser.add_argument('-test_gradient')
args = parser.parse_args()
test_consistency = bool(args.test_consistency or False)
test_gradient = bool(args.test_gradient or False)

# Parameter object
level = int(args.level)
op = PointDischarge2dOptions(level=level)
assert args.family in ('cg', 'dg')
op.tracer_family = args.family or 'cg'
op.stabilisation = args.stabilisation
op.di = os.path.join(op.di, args.stabilisation or args.family)
auto_sipg = bool(args.use_automatic_sipg_parameter or False)
if op.tracer_family == 'cg':
    op.use_automatic_sipg_parameter = False
else:
    op.use_automatic_sipg_parameter = auto_sipg
    if auto_sipg:
        op.di += '_sipg'
mesh = op.default_mesh
x, y = SpatialCoordinate(mesh)

# Source parametrisation
x0 = 2.0
y0 = 5.0
source_value = 100.0
R = FunctionSpace(mesh, "R", 0)
r_to_calibrate = Function(R).assign(0.1)
control = Control(r_to_calibrate)

# Use high quadrature degree
dx = dx(degree=12)

def exact_solution(r):
    u = Constant(as_vector(op.base_velocity))
    D = Constant(op.base_diffusivity)
    q = 1.0
    rr = max_value(sqrt((x - x0)**2 + (y - y0)**2), r)
    return 0.5*q/(pi*D)*exp(0.5*u[0]*(x - x0)/D)*bessk0(0.5*u[0]*rr/D) 

def scaled_ball(xx, yy, rr, scaling=1.0):
    area = assemble(conditional((x - xx)**2 + (y - yy)**2 <= rr**2, 1.0, 0.0)*dx)
    exact_area = pi*rr*rr
    scaling *= exact_area/area
    return conditional((x - xx)**2 + (y - yy)**2 <= rr**2, scaling, 0.0)

def set_tracer_source(r):
    return scaled_ball(x0, y0, r, scaling=0.5*source_value)

def solve(r):
    tp = AdaptiveSteadyProblem(op)
    tp.set_initial_condition()
    tp.fields[0].tracer_source_2d = set_tracer_source(r)
    tp.setup_solver_forward_step(0)
    tp.solve_forward_step(0)
    return tp.fwd_solution_tracer

def to_test(r):
    c = solve(r)
    sol = exact_solution(r)
    kernel1 = scaled_ball(20.0, 5.0, 0.5)
    J1 = assemble(kernel1*c*dx)
    J1_exact = assemble(kernel1*sol*dx)
    kernel2 = scaled_ball(20.0, 7.5, 0.5)
    J2 = assemble(kernel2*c*dx)
    J2_exact = assemble(kernel2*sol*dx)
    return (J1 - J1_exact)**2 + (J2 - J2_exact)**2

# Progress arrays
control_progress = []
functional_progress = []
gradient_progress = []

def callback(j, dj, m):
    msg = "functional {:15.8e}  gradient {:15.8e}  control {:15.8e}"
    djdm = dj.dat.data[0]
    mm = m.dat.data[0]
    control_progress.append(mm)
    functional_progress.append(j)
    gradient_progress.append(djdm)
    print_output(msg.format(j, djdm, mm))

# Reduced functional
J = to_test(r_to_calibrate)
Jhat = ReducedFunctional(J, control, derivative_cb_post=callback)

# Test consistency
if test_consistency:
    JJ = Jhat(r_to_calibrate)
    assert np.isclose(J, JJ), "{:.4e} vs. {:.4e}".format(J, JJ)

# Taylor test
if test_gradient:
    m = Function(R).assign(r_to_calibrate)
    dm = Function(R).assign(0.1)
    minconv = taylor_test(Jhat, m, dm)
    assert minconv > 1.90

# Optimisation
callback = lambda m: print_output("LINE SEARCH COMPLETE")
r_calibrated = minimize(Jhat, method='L-BFGS-B', bounds=(0, 1), callback=callback)

# Logging
logstr = "level: {:d}\n".format(level)
logstr += "family: {:s}\n".format(op.tracer_family.upper())
if op.stabilisation is not None:
    logstr += "stabilisation: {:}\n".format(op.stabilisation.upper())
if op.tracer_family == 'dg':
    logstr += "automatic SIPG: {:}\n".format(op.use_automatic_sipg_parameter)
logstr += "calibrated radius: {:.8f}\n".format(r_calibrated.dat.data[0])
print_output(logstr)
with open(os.path.join(op.di, "log"), "w") as log:
    log.write(logstr)

# Plot calibrated exact and approx solutions
approx = solve(r_calibrated)
exact = Function(approx.function_space(), name="Exact solution")
exact.interpolate(exact_solution(r_calibrated))
File(os.path.join(op.di, "exact.pvd")).write(exact)
error = Function(approx.function_space(), name="Absolute error")
error.interpolate(abs(exact - approx))
File(os.path.join(op.di, "error.pvd")).write(error)

# Save optimisation progress
ext = args.family
if ext == 'dg':
    if args.stabilisation in ('lf', 'LF', 'lax_friedrichs'):
        ext += '_lf'
    if auto_sipg:
        ext += '_sipg'
else:
    if args.stabilisation in ('su', 'SU'):
        ext += '_su'
    if args.stabilisation in ('supg', 'SUPG'):
        ext += '_supg'
fname = "_".join(["{:s}", ext, str(level)])
fname += ".npy"
np.save(os.path.join(op.di, fname.format("control")), np.array(control_progress))
np.save(os.path.join(op.di, fname.format("functional")), np.array(functional_progress))
np.save(os.path.join(op.di, fname.format("gradient")), np.array(gradient_progress))
