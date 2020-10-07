from thetis import *
from firedrake_adjoint import *

import numpy as np
import os
# import scipy

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
    # 'end_time': 120,  # TODO: TEMP

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
swp = AdaptiveDiscreteAdjointProblem(op, nonlinear=nonlinear, print_progress=op.debug)
print_output("Setting initial guess...")
op.assign_control_parameters(kwargs['control_parameters'], swp.meshes[0])
# NOTE: If we are taking this route, then we don't actually need to use the R space
op.get_basis_functions(swp.V[0].sub(1))


def source(m):
    """
    The source model: linear combination.

    S: R^N -> V
    """
    S = Function(swp.V[0].sub(1))
    assert len(m) == len(op.basis_functions)
    for m_i, phi_i in zip(m, op.basis_functions):
        # S.interpolate(S + m_i*phi_i)
        S.project(S + m_i*phi_i)
    return S


def adjoint_source(f):
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
    assert taylor_test(ReducedFunctional(assemble(inner(eta0, eta0)*dx), box_controls), m, dm) > 1.90


# --- Tracing

# Solve the forward problem
u, eta = swp.fwd_solution.split()
u.assign(0.0)
eta.assign(eta0)
print_output("Run forward to get initial timeseries...")
swp.setup_solver_forward_step(0)
swp.solve_forward_step(0)
# J = swp.quantity_of_interest()  # FIXME
J = assemble(inner(swp.fwd_solution, swp.fwd_solution)*dx)

# Define reduced functionals
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
    # JJ = swp.quantity_of_interest()
    JJ = assemble(inner(swp.fwd_solution, swp.fwd_solution)*dx)

    # Check consistency
    msg = "Pyadjoint disagrees with solve_forward: {:.8e} vs {:.8e}"
    assert np.isclose(J, JJ), msg.format(J, JJ)
    print_output("Tape unroll consistency test passed!")


def tsunami(eta_init):
    """
    The tsunami propagation model.

    T: V -> R
    """
    return Jhat(eta_init)


def adjoint_tsunami():  # TODO: Do we need an arg?
    """
    Adjoint of the tsunami propagation model.

    T*: R -> V
    """
    return Jhat.derivative()


# --- TAYLOR TEST TSUNAMI

if taylor_test_tsunami:
    assert taylor_test(Jhat, eta0, deta0) > 1.90
    print_output("Taylor test for tsunami propagation passed!")


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
    dJdm = adjoint_source(adjoint_tsunami())  # TODO: args?
    print_output(27*" " + "gradient {:15.8e}".format(vecnorm(dJdm, order=np.Inf)))
    return dJdm


# --- TAYLOR TEST COMPOSITION

if taylor_test_composition:
    assert taylor_test(Jhat_box, m, dm) > 1.90
    print_output("Taylor test for composition passed!")


# --- OPTIMISATION

def optimisation_callback(m):
    # TODO: Save progress here
    print_output("LINE SEARCH COMPLETE")


print_output("Optimisation begin...")
opt_kwargs = {
    # 'fprime': gradient,
    # 'callback': optimisation_callback,
    'maxiter': 1000,
    'gtol': 1.0e-04,
}
optimised_value = minimize(Jhat, method='BFGS', callback=optimisation_callback, options=opt_kwargs)
# initial_guess = kwargs['control_parameters']
# optimised_value = scipy.optimize.fmin_bfgs(reduced_functional, initial_guess, **opt_kwargs)
