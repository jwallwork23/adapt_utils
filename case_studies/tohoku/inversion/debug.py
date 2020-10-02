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
    fs = swp.V[0].sub(1)
    S = Function(fs)
    assert len(m) == len(op.basis_functions)
    for m_i, phi_i in zip(m, op.basis_functions):
        S += project(m_i*phi_i, fs)
    return S


def adjoint_source(f):
    """
    Adjoint of the source model: testing.

    S*: V -> R^N
    """
    S_star = np.zeros(len(op.basis_functions))
    for i, phi_i in enumerate(op.basis_functions):
        S_star[i] = assemble(phi_i*f*dx)
    return S_star


# Solve the forward problem / load data
print_output("Run forward to get initial timeseries...")
swp.clear_tape()
u, eta = swp.fwd_solution.split()
u.assign(0.0)
eta.assign(source(op.control_parameters))
pyadjoint_control = Control(eta)
swp.setup_solver_forward_step(0)
swp.solve_forward_step(0)
J = swp.quantity_of_interest()

# Define reduced functional and gradient functions
Jhat = ReducedFunctional(J, pyadjoint_control)
stop_annotating()


def tsunami(eta0):
    """
    The tsunami propagation model.

    T: V -> R
    """
    return Jhat(eta0)


def adjoint_tsunami():
    """
    Adjoint of the tsunami propagation model.

    T*: R -> V
    """
    return Jhat.derivative()


def reduced_functional(m):
    """
    Compose the source model and the tsunami propagation model.
    """
    J = tsunami(source(m))
    print_output("functional {:15.8e}".format(J))
    return J


def gradient(m):
    """
    Compose the adjoint tsunami propagation and the adjoint source using the chain rule.
    """
    dJdm = adjoint_source(Jhat.derivative())
    print_output(27*" " + "gradient {:15.8e}".format(vecnorm(dJdm, order=np.Inf)))
    return dJdm


# --- TEST CONSISTENCY OF REDUCED FUNCTIONAL EVALUATION

# Change the control parameters
m = op.control_parameters
for control in m:
    control.assign(-control)

# Unroll tape
J = reduced_functional(m)

# By hand
eta0 = source(m)
u, eta = swp.fwd_solution.split()
u.assign(0.0)
eta.assign(eta0)
swp.setup_solver_forward_step(0)
swp.solve_forward_step(0)
JJ = swp.quantity_of_interest()

# Check consistency
msg = "Pyadjoint disagrees with solve_forward: {:.8e} vs {:.8e}"
assert np.isclose(J, JJ), msg.format(J, JJ)
print_output("Tape unroll consistency test passed!")


# --- TAYLOR TEST SOURCE

np.random.seed(0)
# TODO
# print_output("Taylor test for source passed!")


# --- TAYLOR TEST TSUNAMI

deta0 = Function(eta0)
deta0.dat.data[:] = np.random.rand(*deta0.dat.data.shape)
minconv = taylor_test(Jhat, eta0, deta0)
assert minconv > 1.90
print_output("Taylor test for tsunami propagation passed!")


# --- TAYLOR TEST COMPOSITION

# TODO
# print_output("Taylor test for composition passed!")


# --- OPTIMISATION

initial_guess = kwargs['control_parameters']
opt_kwargs = {
    'fprime': gradient,
    'callback': lambda _: print_output("LINE SEARCH COMPLETE"),
    'maxiter': 1000,
    'gtol': 1.0e-04,
}
print_output("Optimisation begin...")
optimised_value = scipy.optimize.fmin_bfgs(reduced_functional, initial_guess, **opt_kwargs)
