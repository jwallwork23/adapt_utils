from thetis import *
from firedrake_adjoint import *

import argparse
import numpy as np

from adapt_utils.maths import bessk0
from adapt_utils.steady.solver import AdaptiveSteadyProblem
from adapt_utils.steady.test_cases.point_discharge2d.options import PointDischarge2dOptions


# Parse arguments
parser = argparse.ArgumentParser()
<<<<<<< HEAD
parser.add_argument('-level', help="Mesh resolution level.")
parser.add_argument('-family', help="Finite element family, from {'cg', 'dg'}.")
=======
parser.add_argument('level', help="Mesh resolution level.")
parser.add_argument('family', help="Finite element family, from {'cg', 'dg'}.")
>>>>>>> dfe1c0b3a34dfef1765835b64b574a69fe60dd9a
parser.add_argument('-stabilisation', help="""
    Stabilisation method. No stabilisation by default.
    Otherwise, choose from {'su', 'supg', 'lax_friedrichs'}.
    Note that 'su' and 'supg' are ignored unless the finite element family is CG.
    Note that 'lax_friedrichs' is ignored unless the finite element family is DG.
    """)
parser.add_argument('-anisotropic_stabilisation', help="Use anisotropic cell size measure?")
parser.add_argument('-error_to_calibrate', help="Choose from {'l2', 'qoi'}.")
parser.add_argument('-test_consistency', help="Test consistency of taped reduced functional.")
parser.add_argument('-test_gradient', help="Taylor test reduced functional.")
parser.add_argument('-recompute_parameter_space')
args = parser.parse_args()

# Collect parsed arguments
<<<<<<< HEAD
test_consistency = False if args.test_consistency == '0' else True
test_gradient = False if args.test_gradient == '0' else True
=======
test_consistency = bool(args.test_consistency or False)
test_gradient = bool(args.test_gradient or False)
>>>>>>> dfe1c0b3a34dfef1765835b64b574a69fe60dd9a
error_to_calibrate = args.error_to_calibrate or 'l2'
assert error_to_calibrate in ('l2', 'qoi')

# Parameter object
<<<<<<< HEAD
level = int(args.level or 0)
op = PointDischarge2dOptions(level=level)
family = args.family or 'cg'
assert family in ('cg', 'dg')
op.tracer_family = family
stabilisation = args.stabilisation or 'supg'
op.stabilisation_tracer = None if stabilisation == 'none' else stabilisation
op.anisotropic_stabilisation = False if args.anisotropic_stabilisation == '0' else True
op.di = os.path.join(op.di, op.stabilisation_tracer or args.family)
=======
level = int(args.level)
op = PointDischarge2dOptions(level=level)
assert args.family in ('cg', 'dg')
op.tracer_family = args.family or 'cg'
op.stabilisation = args.stabilisation
op.anisotropic_stabilisation = bool(args.anisotropic_stabilisation or False)
op.di = os.path.join(op.di, args.stabilisation or args.family)
>>>>>>> dfe1c0b3a34dfef1765835b64b574a69fe60dd9a
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


def analytical_solution(r):
    """
    Compute analytical solution field for a given source radius.
    """
    u = Constant(as_vector(op.base_velocity))
    D = Constant(op.base_diffusivity)
    Pe = 0.5*u[0]/D
    q = 1.0
    rr = max_value(sqrt((x - x0)**2 + (y - y0)**2), r)
    return 0.5*q/(pi*D)*exp(Pe*(x - x0))*bessk0(Pe*rr)


def scaled_ball(xx, yy, rr, scaling=1.0):
    """
    UFL expression for a disc, scaled by an area adjustment.

    :args xx,yy: centre of disc.
    :arg rr: radius of disc.
    :kwarg scaling: value by which to scale.
    """
    area = assemble(conditional((x - xx)**2 + (y - yy)**2 <= rr**2, 1.0, 0.0)*dx)
    analytical_area = pi*rr*rr
    scaling *= analytical_area/area
    return conditional((x - xx)**2 + (y - yy)**2 <= rr**2, scaling, 0.0)


def gaussian(xx, yy, rr, scaling=1.0):
    """
    UFL expression for a Gaussian bump, scaled as appropriate.

    :args xx,yy: centre of Gaussian.
    :arg rr: radius of Gaussian.
    :kwarg scaling: value by which to scale.
    """
    return scaling*exp(-((x - xx)**2 + (y - yy)**2)/rr**2)


def set_tracer_source(r):
    """
    Generate source field for a given source radius.
    """
    return gaussian(x0, y0, r, scaling=source_value)


def solve(r):
    """
    Solve the tracer transport problem for a given source radius.
    """
    tp = AdaptiveSteadyProblem(op, print_progress=False)
    tp.set_initial_condition()
    tp.fields[0].tracer_source_2d = set_tracer_source(r)
    tp.setup_solver_forward_step(0)
    tp.solve_forward_step(0)
    return tp.fwd_solution_tracer


def l2_error(r):
    """
    Squared L2 error of approximate solution, ignoring the source region.
    """
    c = solve(r)
    sol = analytical_solution(r)
    kernel = conditional((x - x0)**2 + (y - y0)**2 > r**2, 1.0, 0.0)
    return assemble(kernel*(c - sol)**2*dx)


def qoi_error(r):
    """
    Sum of squared errors of the aligned and offset QoIs.
    """
    c = solve(r)
    sol = analytical_solution(r)
    kernel1 = scaled_ball(20.0, 5.0, 0.5)
    J1 = assemble(kernel1*c*dx)
    J1_analytical = assemble(kernel1*sol*dx)
    kernel2 = scaled_ball(20.0, 7.5, 0.5)
    J2 = assemble(kernel2*c*dx)
    J2_analytical = assemble(kernel2*sol*dx)
    return (J1 - J1_analytical)**2 + (J2 - J2_analytical)**2


# Progress arrays
control_progress = []
functional_progress = []
gradient_progress = []


def callback(j, dj, m):
    """
    Print and store progress of optimisation routine.
    """
    djdm = dj.dat.data[0]
    mm = m.dat.data[0]
    control_progress.append(mm)
    functional_progress.append(j)
    gradient_progress.append(djdm)
    msg = "functional {:15.8e}  gradient {:15.8e}  control {:15.8e}"
    print_output(msg.format(j, djdm, mm))


# Reduced functional
print_output("Tracing...")
reduced_functional = l2_error if error_to_calibrate == 'l2' else qoi_error
J = reduced_functional(r_to_calibrate)
Jhat = ReducedFunctional(J, control, derivative_cb_post=callback)
stop_annotating()

# Test consistency
if test_consistency:
    print_output("Testing consistency...")
    JJ = Jhat(r_to_calibrate)
    assert np.isclose(J, JJ), "{:.4e} vs. {:.4e}".format(J, JJ)

# Taylor test
if test_gradient:
    print_output("Testing gradient...")
    m = Function(R).assign(r_to_calibrate)
    dm = Function(R).assign(0.1)
    minconv = taylor_test(Jhat, m, dm)
    assert minconv > 1.90

# Plot parameter space
fname = os.path.join(op.di, "parameter_space_{:d}.npy".format(level))
<<<<<<< HEAD
# if not os.path.isfile(fname) or bool(args.recompute_parameter_space or False):
if bool(args.recompute_parameter_space or False):
=======
if not os.path.isfile(fname) or bool(args.recompute_parameter_space or False):
>>>>>>> dfe1c0b3a34dfef1765835b64b574a69fe60dd9a
    print_output("Exploring parameter space...")
    np.save(fname, np.array([reduced_functional(r) for r in np.linspace(0.01, 0.4, 100)]))

# Optimisation
print_output("Running optimisation...")
callback = lambda _: print_output("LINE SEARCH COMPLETE")
r_calibrated = minimize(Jhat, method='L-BFGS-B', bounds=(0.01, 1), callback=callback)

# Logging
print_output("Logging...")
logstr = "level: {:d}\n".format(level)
logstr += "family: {:s}\n".format(op.tracer_family.upper())
<<<<<<< HEAD
if op.stabilisation_tracer is not None:
    logstr += "stabilisation: {:}\n".format(op.stabilisation_tracer.upper())
=======
if op.stabilisation is not None:
    logstr += "stabilisation: {:}\n".format(op.stabilisation.upper())
>>>>>>> dfe1c0b3a34dfef1765835b64b574a69fe60dd9a
logstr += "calibrated radius: {:.8f}\n".format(r_calibrated.dat.data[0])
print_output(logstr)
with open(os.path.join(op.di, "log"), "a") as log:
    log.write(logstr)

# Plot calibrated analytical and approx solutions
print_output("Plotting...")
approx = solve(r_calibrated)
analytical = Function(approx.function_space(), name="Analytical solution")
analytical.interpolate(analytical_solution(r_calibrated))
File(os.path.join(op.di, "analytical.pvd")).write(analytical)
error = Function(approx.function_space(), name="Absolute error")
error.interpolate(abs(analytical - approx))
File(os.path.join(op.di, "error.pvd")).write(error)

# Save optimisation progress
print_output("Saving optimisation progress...")
<<<<<<< HEAD
ext = op.tracer_family
if ext == 'dg':
    if op.stabilisation_tracer in ('lf', 'LF', 'lax_friedrichs'):
        ext += '_lf'
else:
    if op.stabilisation_tracer in ('su', 'SU'):
        ext += '_su'
    if op.stabilisation_tracer in ('supg', 'SUPG'):
        ext += '_supg'
    if op.anisotropic_stabilisation and op.stabilisation_tracer is not None:
=======
ext = args.family
if ext == 'dg':
    if args.stabilisation in ('lf', 'LF', 'lax_friedrichs'):
        ext += '_lf'
else:
    if args.stabilisation in ('su', 'SU'):
        ext += '_su'
    if args.stabilisation in ('supg', 'SUPG'):
        ext += '_supg'
    if op.anisotropic_stabilisation:
>>>>>>> dfe1c0b3a34dfef1765835b64b574a69fe60dd9a
        ext += '_anisotropic'
fname = "_".join(["{:s}", ext, str(level)])
fname += ".npy"
np.save(os.path.join(op.di, fname.format("control")), np.array(control_progress))
np.save(os.path.join(op.di, fname.format("functional")), np.array(functional_progress))
np.save(os.path.join(op.di, fname.format("gradient")), np.array(gradient_progress))
