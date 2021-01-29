from thetis import *
from firedrake_adjoint import *

import numpy as np
import os
# import scipy

from adapt_utils.case_studies.tohoku.options.box_options import TohokuBoxBasisOptions
from adapt_utils.case_studies.tohoku.options.radial_options import TohokuRadialBasisOptions
# from adapt_utils.norms import vecnorm
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
# test_consistency = False
taylor_test_source = False
taylor_test_tsunami = True
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

op = TohokuBoxBasisOptions(**kwargs)
mesh = op.default_mesh

# Create P1DG-P2 space
P1_vec = VectorFunctionSpace(mesh, "DG", 1)
P1 = FunctionSpace(mesh, "CG", 1)
P2 = FunctionSpace(mesh, "CG", 2)
V = P1_vec*P2

# Set initial guess
print_output("Setting initial guess...")
op.assign_control_parameters(kwargs['control_parameters'], mesh)
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

# Create Thetis solver object
# ===========================
#  g = 9.81
#  nu = 0
#  f = 2*Omega_E*sin(lat)
#  C_d = 0
#  nonlinear = False
print_output("Run forward to get initial timeseries...")
solver_obj = solver2d.FlowSolver2d(mesh, op.set_bathymetry(P1))
options = solver_obj.options
options.use_nonlinear_equations = nonlinear
options.element_family = op.family
options.simulation_export_time = op.dt*op.dt_per_export
options.simulation_end_time = op.end_time
options.timestepper_type = op.timestepper
options.use_lax_friedrichs_velocity = op.stabilisation == 'lax_friedrichs'
options.timestep = op.dt
options.horizontal_viscosity = None  # problem is inviscid
options.coriolis_frequency = op.set_coriolis(P1)
solver_obj.assign_initial_conditions(elev=eta0)
solver_obj.bnd_functions['shallow_water'] = {
    100: {'un': Constant(0.0), 'elev': Constant(0.0)},
    200: {'un': Constant(0.0)},
    300: {'un': Constant(0.0)},
}
options._isfrozen = False
options.qoi = 0

# Kernel function for elevation
# kernel = Function(solver_obj.function_spaces.V_2d)
# kernel.assign(solver_obj.fields.solution_2d)  # WORKS
# indicator = Function(P0)
# indicator.assign(2.0)
# kernel.project(indicator*solver_obj.fields.solution_2d)  # WORKS

P0 = FunctionSpace(mesh, "DG", 0)
P0_vec = VectorFunctionSpace(mesh, "DG", 0)
W = P0_vec*P0
kernel = Function(W)

test_u, test_eta = TestFunctions(W)
trial_u, trial_eta = TrialFunctions(W)
a = inner(test_u, trial_u)*dx + inner(test_eta, trial_eta)*dx
L = inner(test_u, Constant(as_vector([0.0, 0.0])))*dx + inner(test_eta, Constant(1.0))*dx
solve(a == L, kernel)  # DOESN'T WORK

# kernel.assign(2.0)  # DOESN'T WORK
# k_u, k_eta = kernel.split()
# k_eta.assign(1.0)


def update_forcings(t):
    q = solver_obj.fields.solution_2d
    # options.qoi = options.qoi + assemble(solver_obj.fields.elev_2d*dx)  # DOESN'T WORK
    # options.qoi = options.qoi + assemble(inner(q, q)*dx)  # WORKS
    # options.qoi = options.qoi + assemble(inner(kernel, q)*dx)  # WORKS if project sol
    elev = project(inner(kernel, q), P1)
    options.qoi = options.qoi + assemble(elev*dx)
    # kernel.assign(q)
    # options.qoi = options.qoi + assemble(inner(kernel, kernel)*dx)


solver_obj.iterate(update_forcings=update_forcings)
# q = solver_obj.fields.solution_2d
# J = assemble(inner(q, q)*dx)
J = options.qoi

# Define reduced functionals
Jhat_box = ReducedFunctional(J, box_controls)
Jhat = ReducedFunctional(J, pyadjoint_control)
stop_annotating()


# --- TEST CONSISTENCY OF REDUCED FUNCTIONAL EVALUATION

# Change the control parameters
for control in m:
    control.assign(-control)

eta0 = source(m)
# if test_consistency:
#
#     # Unroll tape
#     J = reduced_functional(m)
#
#     # By hand
#     u, eta = swp.fwd_solution.split()
#     u.assign(0.0)
#     eta.assign(eta0)
#     swp.setup_solver_forward_step(0)
#     swp.solve_forward_step(0)
#     JJ = assemble(inner(swp.fwd_solution, swp.fwd_solution)*dx)
#
#     # Check consistency
#     msg = "Pyadjoint disagrees with solve_forward: {:.8e} vs {:.8e}"
#     assert np.isclose(J, JJ), msg.format(J, JJ)
#     print_output("Tape unroll consistency test passed!")


# --- TAYLOR TEST TSUNAMI

if taylor_test_tsunami:
    assert taylor_test(Jhat, eta0, deta0) > 1.90
    print_output("Taylor test for tsunami propagation passed!")


# --- TAYLOR TEST COMPOSITION

if taylor_test_composition:
    # dJdm = np.dot(gradient(m), [dmi.dat.data[0] for dmi in dm])
    # # dJdm = sum(hi._ad_dot(di) for hi, di in zip(dm, gradient(m)))
    # minconv = taylor_test(reduced_functional, m, dm, dJdm=dJdm)
    # assert minconv > 1.90
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
optimised_value = minimize(Jhat_box, method='BFGS', callback=optimisation_callback, options=opt_kwargs)
# initial_guess = kwargs['control_parameters']
# optimised_value = scipy.optimize.fmin_bfgs(reduced_functional, initial_guess, **opt_kwargs)
