from thetis import *
from firedrake_adjoint import *

import numpy as np
import os
import scipy

from adapt_utils.case_studies.tohoku.options.box_options import TohokuBoxBasisOptions
from adapt_utils.case_studies.tohoku.options.radial_options import TohokuRadialBasisOptions
from adapt_utils.norms import vecnorm
from adapt_utils.unsteady.solver_adjoint import AdaptiveDiscreteAdjointProblem


# --- Set parameters

# Setup output directories
dirname = os.path.dirname(__file__)
di = create_directory(os.path.join(dirname, 'box', 'outputs', 'realistic', 'discrete'))

# Parameters
kwargs = {
    'level': 0,
    'save_timeseries': True,
    'end_time': 120,  # TODO: TEMP

    # Spatial discretisation
    'family': 'dg-cg',
    'stabilisation': 'lax_friedrichs',
    'use_automatic_sipg_parameter': False,  # the problem is inviscid

    # Inversion
    'synthetic': False,

    # I/O and debugging
    'plot_pvd': False,
    'debug': False,
    'debug_mode': 'basic',
    'di': di,
}
nonlinear = False
zero_init = False
gaussian_scaling = 6.0

# Construct Options parameter class
op = TohokuBoxBasisOptions(**kwargs)
op.dirty_cache = True
gauges = list(op.gauges.keys())

# Tests
test_consistency = False
taylor_test_source = False
taylor_test_tsunami = False
taylor_test_composition = True


# --- Set initial guess

with stop_annotating():
    if zero_init:
        eps = 1.0e-03  # zero gives an error so just choose small
        kwargs['control_parameters'] = eps*np.ones(op.nx*op.ny)
    else:
        print_output("Projecting initial guess...")

        # Create Radial parameter object
        kwargs_src = kwargs.copy()
        kwargs_src['control_parameters'] = [gaussian_scaling]
        kwargs_src['nx'], kwargs_src['ny'] = 1, 1
        op_src = TohokuRadialBasisOptions(mesh=op.default_mesh, **kwargs_src)
        swp = AdaptiveDiscreteAdjointProblem(op_src, nonlinear=nonlinear, print_progress=op.debug)
        swp.set_initial_condition()
        f_src = swp.fwd_solutions[0].split()[1]

        # Project into chosen basis
        swp = AdaptiveDiscreteAdjointProblem(op, nonlinear=nonlinear, print_progress=op.debug)
        op.project(swp, f_src)
        kwargs['control_parameters'] = [m.dat.data[0] for m in op.control_parameters]


# --- Tracing

# Set initial guess
op = TohokuBoxBasisOptions(**kwargs)
mesh = op.default_mesh
P1_vec = VectorFunctionSpace(mesh, "DG", 1)
P1 = FunctionSpace(mesh, "CG", 1)
P2 = FunctionSpace(mesh, "CG", 2)
V = P1_vec*P2
print_output("Setting initial guess...")
op.assign_control_parameters(kwargs['control_parameters'], mesh)
# NOTE: If we are taking this route, then we don't actually need to use the R space
op.get_basis_functions(P2)


def source(control_vector):
    """
    The source model: linear combination.

    S: R^N -> V
    """
    S = Function(P2)
    assert len(control_vector) == len(op.basis_functions)
    for m_i, phi_i in zip(control_vector, op.basis_functions):
        # S.interpolate(S + m_i*phi_i)
        S.project(S + m_i*phi_i)
    return S


def reverse_source(f):  # FIXME: Or just apply pyadjoint here, too
    """
    Adjoint of the source model: testing.

    S*: V -> R^N
    """
    S_star = np.zeros(len(op.basis_functions))
    for i, phi_i in enumerate(op.basis_functions):
        S_star[i] = np.dot(phi_i.dat.data, f.dat.data)
        # S_star[i] = assemble(phi_i*f*dx(degree=12))
    return S_star


# Get initial surface
tape = get_working_tape()
tape.clear_tape()
m = op.control_parameters
box_controls = [Control(c) for c in m]
eta0 = source(m)
pyadjoint_control = Control(eta0)

# Random search directions
np.random.seed(0)
deta0 = Function(eta0)
deta0.dat.data[:] = np.random.rand(*deta0.dat.data.shape)
dm = [Function(c) for c in m]
for dmi in dm:
    dmi.dat.data[0] = np.random.rand(1)


# --- TAYLOR TEST SOURCE

if taylor_test_source:
    J = assemble(inner(eta0, eta0)*dx)
    Jhat = ReducedFunctional(J, box_controls)
    minconv = taylor_test(Jhat, m, dm)
    assert minconv > 1.90


# --- Tracing

# Create Thetis solver object
print_output("Run forward to get initial timeseries...")
solver_obj = solver2d.FlowSolver2d(mesh, op.set_bathymetry(P1))
options = solver_obj.options
options.use_nonlinear_equations = False
options.element_family = op.family
options.simulation_export_time = op.dt*op.dt_per_export
options.simulation_end_time = op.end_time
options.timestepper_type = op.timestepper
options.timestep = op.dt
options.horizontal_viscosity = None
options.coriolis_frequency = op.set_coriolis(P1)
solver_obj.assign_initial_conditions(elev=eta0)
solver_obj.bnd_functions['shallow_water'] = {
    100: {'un': Constant(0.0), 'elev': Constant(0.0)},
    200: {'un': Constant(0.0)},
    300: {'un': Constant(0.0)},
}
solver_obj.iterate()
q = solver_obj.fields.solution_2d
J = assemble(inner(q, q)*dx)

# Define reduced functional and gradient functions
Jhat = ReducedFunctional(J, pyadjoint_control)
Jhat_box = ReducedFunctional(J, box_controls)
stop_annotating()


# --- TEST CONSISTENCY OF REDUCED FUNCTIONAL EVALUATION

# Change the control parameters
for control in m:
    control.assign(-control)

eta0 = source(m)
if test_consistency:

    # Unroll tape
    J = reduced_functional(m)

    # By hand
    u, eta = swp.fwd_solution.split()
    u.assign(0.0)
    eta.assign(eta0)
    swp.setup_solver_forward_step(0)
    swp.solve_forward_step(0)
    JJ = assemble(inner(swp.fwd_solution, swp.fwd_solution)*dx)

    # Check consistency
    msg = "Pyadjoint disagrees with solve_forward: {:.8e} vs {:.8e}"
    assert np.isclose(J, JJ), msg.format(J, JJ)
    print_output("Tape unroll consistency test passed!")


# --- TAYLOR TEST TSUNAMI


def tsunami(eta_init):
    """
    The tsunami propagation model.

    T: V -> R
    """
    return Jhat(eta_init)


def reverse_tsunami():  # TODO: Do we need an arg?
    """
    Adjoint of the tsunami propagation model.

    T*: R -> V
    """
    return Jhat.derivative()


if taylor_test_tsunami:
    minconv = taylor_test(Jhat, eta0, deta0)
    assert minconv > 1.90
    print_output("Taylor test for tsunami propagation passed!")


# --- TAYLOR TEST COMPOSITION


def reduced_functional(control_vector):
    """
    Compose the source model and the tsunami propagation model.
    """
    J = tsunami(source(control_vector))
    print_output("functional {:15.8e}".format(J))
    return J


def gradient(control_vector):
    """
    Compose the adjoint tsunami propagation and the adjoint source using the chain rule.
    """
    dJdm = reverse_source(reverse_tsunami())  # TODO: args?
    print_output(27*" " + "gradient {:15.8e}".format(vecnorm(dJdm, order=np.Inf)))
    return dJdm


if taylor_test_composition:
    minconv = taylor_test(Jhat_box, m, dm)
    # dJdm = np.dot(gradient(m), [dmi.dat.data[0] for dmi in dm])
    # # dJdm = sum(hi._ad_dot(di) for hi, di in zip(dm, gradient(m)))
    # minconv = taylor_test(reduced_functional, m, dm, dJdm=dJdm)
    assert minconv > 1.90
    print_output("Taylor test for composition passed!")


# --- OPTIMISATION

def optimisation_callback(m):
    # TODO: Save progress here
    print_output("LINE SEARCH COMPLETE")


initial_guess = kwargs['control_parameters']
opt_kwargs = {
    'fprime': gradient,
    'callback': optimisation_callback,
    'maxiter': 1000,
    'gtol': 1.0e-04,
}
print_output("Optimisation begin...")
optimised_value = scipy.optimize.fmin_bfgs(reduced_functional, initial_guess, **opt_kwargs)
