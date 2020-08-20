from thetis import *
from firedrake_adjoint import *

import adolc
import numpy as np
import os
import scipy

from adapt_utils.case_studies.tohoku.options.okada_options import TohokuOkadaBasisOptions
from adapt_utils.unsteady.solver import AdaptiveProblem
from adapt_utils.unsteady.solver_adjoint import AdaptiveDiscreteAdjointProblem
from adapt_utils.misc import taylor_test
from adapt_utils.norms import vecnorm


class DiscreteAdjointTsunamiProblem(AdaptiveDiscreteAdjointProblem):
    """The subclass exists to pass the QoI as required."""
    def quantity_of_interest(self):
        return self.op.J


# TODO: argparse

level = 0
real_data = False
optimise = True
test_gradient = True
N = 51
active_controls = ('slip', 'rake')
tape_tag = 0
kwargs = {

    # Resolution
    "level": level,
    "okada_grid_resolution": N,

    # Model
    "family": "dg-cg",
    "stabilisation": None,

    # Inversion
    "synthetic": not real_data,

    # I/O and debugging
    "save_timeseries": True,
    "plot_pvd": False,
    "debug": False,
}
nonlinear = False
op = TohokuOkadaBasisOptions(**kwargs)
op.active_controls = active_controls
num_subfaults = len(op.subfaults)
num_active_controls = len(active_controls)

# Setup output directories
dirname = os.path.dirname(__file__)
di = create_directory(os.path.join(dirname, 'outputs', 'realistic' if real_data else 'synthetic'))
op.di = create_directory(os.path.join(di, 'discrete'))

# Synthetic run to get timeseries data, with the default control values as the "optimal" case
if not real_data:
    with stop_annotating():
        swp = AdaptiveProblem(op, nonlinear=nonlinear)
        swp.solve_forward()  # NOTE: pyadolc annotation is also off
        for gauge in op.gauges:
            op.gauges[gauge]["data"] = op.gauges[gauge]["timeseries"]

# Get interpolation operator between Okada grid and computational mesh
op.get_interpolation_operators()

# Get seed matrices which allow us to differentiate all active controls simultaneously
op.get_seed_matrices()

# Set initial guess for the optimisation
# ======================================
# Here we just use a random perturbation of the "optimal" controls. In the case where we just use a
# zero initial guess, the gradient with respect to rake is zero.  # TODO: Maybe this is okay?
kwargs['control_parameters'] = op.control_parameters
mu, sigma = 0, 5
m_init = []
for control in op.active_controls:
    size = np.shape(op.control_parameters[control])
#     kwargs['control_parameters'][control] = np.zeros(np.shape(op.control_parameters[control]))
    kwargs['control_parameters'][control] += np.random.normal(loc=mu, scale=sigma, size=size)
    m_init.append(kwargs['control_parameters'][control])
m_init = np.array(m_init)
J_progress = []

# Create discrete adjoint solver object for the optimisation
swp = DiscreteAdjointTsunamiProblem(op, nonlinear=nonlinear)


def reduced_functional(m):
    """
    Given a vector of (active) control parameters, m, run the forward tsunami propagation model and
    evaluate the gauge timeseries misfit quantity of interest (QoI) as a pyadjoint AdjFloat.
    """
    control_parameters = m.reshape((num_subfaults, num_active_controls))
    for i, subfault in enumerate(op.subfaults):
        for j, control in enumerate(active_controls):
            op.control_parameters[control][i] = control_parameters[i, j]

    # Annotate ADOL-C's tape and propagate seed matrices through the forward mode of AD
    swp.set_initial_condition(annotate_source=True, tag=tape_tag)

    # Store control as a pyadjoint Control
    swp.source_control = Control(swp.fwd_solutions[0])

    # Run forward with zero initial guess
    swp.setup_solver_forward(0)
    swp.solve_forward_step(0)
    J = op.J
    J_progress.append(J)

    return J


def gradient(m):
    """
    Compute the gradient of the QoI w.r.t. a vector of (active) control parameters, m.

    The calculation is done in three steps:
      * Propagate m through the forward mode of AD applied to the Okada earthquake source model;
      * Solve the adjoint problem associated with the tsunami propagation;
      * Assemble the gradient by integrating the products of the adjoint solution at time t = 0
        with derivatives of the source model.
    """

    # Propagate controls through forward mode of AD
    F, dFdm = adolc.fov_forward(tape_tag, m, op.seed_matrices)
    dFdm = dFdm.reshape(num_subfaults, N, N, num_active_controls)

    # Solve adjoint problem and extract solution at t = 0
    swp.compute_gradient(swp.source_control)
    swp.get_solve_blocks()
    swp.extract_adjoint_solution(0)
    u_star, eta_star = swp.adj_solutions[0].split()

    # Assemble gradient using both components
    dJdm = np.zeros((num_subfaults, num_active_controls))
    for i in range(num_subfaults):
        for j in range(num_active_controls):
            deta0dm = op.interpolate_okada_array(dFdm[i, :, :, j])
            dJdm[i, j] = assemble(eta_star*deta0dm*dx)
    dJdm = dJdm.reshape(num_subfaults*num_active_controls)

    # Print optimisation progress
    print_output("J = {:.4e}  dJdm = {:.4e}".format(op.J, vecnorm(dJdm, order=np.Inf)))
    return dJdm

# Taylor test the gradient
if test_gradient:
    taylor_test(reduced_functional, gradient, m_init, verbose=True)


# Run optimisation
if optimise:
    opt_kwargs = {
        'maxiter': 100,
        'gtol': 1.0e-08,
        'callback': lambda m: print_output("LINE SEARCH COMPLETE"),
        'fprime': gradient,
    }
    m_opt = scipy.optimize.fmin_bfgs(reduced_functional, m_init, **opt_kwargs)
    np.save(m_opt, fname)
    fname = os.path.join(op.di, "optimised_controls_{:d}".format(level))
    J_progress = np.array(J_progress)
    fname = os.path.join(op.di, "optimisation_progress_{:d}".format(level))
    np.save(fname, J_progress)
else:
    m_opt = np.load(os.path.join(op.di, "optimised_controls_{:d}.npy".format(level)))
    J_progress = np.load(os.path.join(op.di, "optimisation_progress_{:d}.npy".format(level)))

print("m =")
print(m_opt)
print("\nJ progress =")
print(J_progress)

# Rerun, plotting to .pvd
with stop_annotating():
    op.plot_pvd = True
    swp = AdaptiveProblem(op, nonlinear=nonlinear)
    swp.solve_forward()  # NOTE: pyadolc annotation is also off
