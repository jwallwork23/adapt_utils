from thetis import *
from firedrake_adjoint import *

import adolc
import argparse
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


# --- Parse arguments

parser = argparse.ArgumentParser()

# Model
parser.add_argument("-level", help="Mesh resolution level")
parser.add_argument("-okada_grid_resolution", help="Mesh resolution level for the Okada grid")
parser.add_argument("-family", help="Finite element pair")
parser.add_argument("-stabilisation", help="Stabilisation approach")
parser.add_argument("-nonlinear", help="Toggle nonlinear model")

# Inversion
parser.add_argument("-rerun_optimisation", help="Rerun optimisation routine")
parser.add_argument("-real_data", help="Toggle whether to use real data")
parser.add_argument("-noisy_data", help="Toggle whether to sample noisy data")
parser.add_argument("-continuous_timeseries", help="Toggle discrete or continuous timeseries")

# I/O and debugging
parser.add_argument("-plot_pdf", help="Toggle plotting to .pdf")
parser.add_argument("-plot_png", help="Toggle plotting to .png")
parser.add_argument("-plot_pvd", help="Toggle plotting to .pvd")
parser.add_argument("-plot_all", help="Toggle plotting to .pdf, .png and .pvd")
parser.add_argument("-plot_only", help="Just plot using saved data")
parser.add_argument("-debug", help="Toggle debugging")
parser.add_argument("-debug_mode", help="Choose debugging mode from 'basic' and 'full'")

# --- Set parameters

# Parsed arguments
args = parser.parse_args()
level = int(args.level or 0)
optimise = bool(args.rerun_optimisation or False)
plot_pvd = bool(args.plot_pvd or False)
plot_pdf = bool(args.plot_pdf or False)
plot_png = bool(args.plot_png or False)
plot_all = bool(args.plot_all or False)
plot_only = bool(args.plot_only or False)
if plot_only:
    plot_all = True
if plot_all:
    plot_pvd = plot_pdf = plot_png = True
if optimise:
    assert not plot_only
real_data = bool(args.real_data or False)
timeseries_type = "timeseries"
if bool(args.continuous_timeseries or False):
    timeseries_type = "_".join([timeseries_type, "smooth"])

# Do not attempt to plot in parallel
if COMM_WORLD.size > 1 and (plot_pdf or plot_png):
    print_output(120*'*' + "\nWARNING: Plotting turned off when running in parallel.\n" + 120*'*')
    plot_pdf = plot_png = False

test_gradient = True  # TODO: argparse
if test_gradient:
    optimise = False


def savefig(filename):
    """To avoid duplication."""
    plt.tight_layout()
    if plot_pdf:
        plt.savefig(filename + '.pdf')
    if plot_png:
        plt.savefig(filename + '.png')


# Collect initialisation parameters
N = int(args.okada_grid_resolution or 51)
active_controls = ('slip', 'rake')
tape_tag = 0
kwargs = {

    # Resolution
    'level': level,
    'okada_grid_resolution': N,

    # Model
    'family': args.family or 'dg-cg',
    'stabilisation': args.stabilisation,
    'use_automatic_sipg_parameter': False,  # the problem is inviscid

    # Inversion
    'synthetic': not real_data,
    'qoi_scaling': 1.0,
    'noisy_data': bool(args.noisy_data or False),

    # I/O and debugging
    'save_timeseries': True,
    'plot_pvd': False,
    'debug': bool(args.debug or False),
    'debug_mode': args.debug_mode or 'basic',

}
nonlinear = bool(args.nonlinear or False)
op = TohokuOkadaBasisOptions(**kwargs)
op.active_controls = active_controls
num_subfaults = len(op.subfaults)
num_active_controls = len(active_controls)

# Plotting parameters
if plot_pdf or plot_png:
    fontsize = 22
    fontsize_tick = 18
    plotting_kwargs = {'markevery': 5}

# Setup output directories
dirname = os.path.dirname(__file__)
di = create_directory(os.path.join(dirname, 'outputs', 'realistic' if real_data else 'synthetic'))
op.di = create_directory(os.path.join(di, 'discrete'))
plot_dir = create_directory(os.path.join(di, 'plots'))
create_directory(os.path.join(plot_dir, 'discrete'))

# Synthetic run to get timeseries data, with the default control values as the "optimal" case
if not real_data:
    with stop_annotating():
        swp = AdaptiveProblem(op, nonlinear=nonlinear)
        swp.solve_forward()  # NOTE: pyadolc annotation is also off
        for gauge in op.gauges:
            op.gauges[gauge]["data"] = op.gauges[gauge][timeseries_type]

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

    # Annotate ADOL-C's tape and propagate
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
exit(0)  # TODO: TEMPORARY

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
