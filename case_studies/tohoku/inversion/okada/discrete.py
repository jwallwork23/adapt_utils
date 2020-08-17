from thetis import *
from firedrake_adjoint import *

import adolc
import numpy as np
import os
import scipy

from adapt_utils.case_studies.tohoku.options.okada_options import TohokuOkadaOptions
from adapt_utils.unsteady.solver import AdaptiveProblem
from adapt_utils.unsteady.solver_adjoint import AdaptiveDiscreteAdjointProblem
from adapt_utils.norms import vecnorm


class DiscreteAdjointTsunamiProblem(AdaptiveDiscreteAdjointProblem):
    """The subclass exists to pass the QoI as required."""
    def quantity_of_interest(self):
        return self.op.J


# TODO: argparse

level = 0
real_data = False
optimise = True
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
op = TohokuOkadaOptions(**kwargs)
op.active_controls = active_controls
num_subfaults = len(op.subfaults)
num_active_controls = len(active_controls)

# Setup output directories
dirname = os.path.dirname(__file__)
di = create_directory(os.path.join(dirname, 'outputs', 'realistic' if real_data else 'synthetic'))
op.di = create_directory(os.path.join(di, 'discrete'))

# Synthetic run to get timeseries data
if not real_data:
    with stop_annotating():
        swp = AdaptiveProblem(op, nonlinear=nonlinear)
        swp.solve_forward()  # NOTE: pyadolc annotation is also off
        for gauge in op.gauges:
            op.gauges[gauge]["data"] = op.gauges[gauge]["timeseries"]

# Copy data over to new parameter class  # TODO: Can we not just reuse the original one?
op_opt = TohokuOkadaOptions(**kwargs)
op_opt.active_controls = active_controls
for gauge in op_opt.gauges:
    op_opt.gauges[gauge]["data"] = op.gauges[gauge]["data"]

# Get interpolation operator between Okada grid and computational mesh, as well as seed matrices
op_opt.get_interpolation_operators()
op_opt.get_seed_matrices()

# Set a random perturbation initial guess for the optimisation
#   NOTE: Under an initial guess of zero for the optimisation, rake stays at zero
kwargs['control_parameters'] = op.control_parameters
mu, sigma = 0, 5
m_init = []
for control in op.active_controls:
    size = np.shape(op.control_parameters[control])
#     kwargs['control_parameters'][control] = np.zeros(np.shape(op.control_parameters[control]))
    kwargs['control_parameters'][control] += np.random.normal(loc=mu, scale=sigma, size=size)
    m_init.append(kwargs['control_parameters'][control])
m_init = np.array(m_init)

# Create solver object
swp = DiscreteAdjointTsunamiProblem(op_opt, nonlinear=nonlinear)

def reduced_functional(m):

    # Annotate ADOL-C's tape and propagate seed matrices through the forward mode of AD
    swp.set_initial_condition(annotate_source=True, tag=tape_tag)
    swp.source_control = Control(swp.fwd_solutions[0])  # Store control as a pyadjoint Control

    # Run forward with zero initial guess
    swp.setup_solver_forward(0)
    swp.solve_forward_step(0)

    return op_opt.J

# # Test reduced functional function
# J = reduced_functional(m_init)

def gradient(m):

    # Propagate controls through forward mode of AD
    F, dFdm = adolc.fov_forward(tape_tag, op_opt.input_vector, op_opt.seed_matrices)
    # F = F.reshape(num_subfaults, N, N)
    # F = sum(F[i, :, :] for i in range(num_subfaults))
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
            deta0dm = op_opt.interpolate_okada_array(dFdm[i, :, :, j])
            dJdm[i, j] = assemble(eta_star*deta0dm*dx)
    dJdm = dJdm.reshape(num_subfaults*num_active_controls)

    # Print optimisation progress
    print_output("J = {:.4e}  dJdm = {:.4e}".format(op_opt.J, vecnorm(dJdm, order=np.Inf)))
    return dJdm

# # Test gradient function
# g = gradient(m_init).reshape((num_subfaults, num_active_controls))
# print("dJdm =")
# print(g)

# Run optimisation
fname = os.path.join(op.di, "optimised_controls_{:d}".format(level))
if optimise:
    opt_kwargs = {
        'maxiter': 100,
        'gtol': 1.0e-08,
        'callback': lambda m: print_output("LINE SEARCH COMPLETE"),
        'fprime': gradient,
    }
    m_opt = scipy.optimize.fmin_bfgs(reduced_functional, m_init, **opt_kwargs)
    np.save(m_opt, fname)
else:
    m_opt = np.load(fname + '.npy')

print("m =")
print(m_opt)

# TODO: Rerun and plot to pvd
