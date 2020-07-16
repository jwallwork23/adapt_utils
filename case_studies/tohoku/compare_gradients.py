from thetis import *
from firedrake_adjoint import *

import argparse
import matplotlib.pyplot as plt
import numpy as np
import os

from adapt_utils.unsteady.solver import AdaptiveProblem
from adapt_utils.unsteady.solver_adjoint import AdaptiveDiscreteAdjointProblem
from adapt_utils.case_studies.tohoku.options import *
# from adapt_utils.norms import total_variation


parser = argparse.ArgumentParser()
parser.add_argument("-level", help="Mesh resolution level")
parser.add_argument("-debug", help="Toggle debugging")
args = parser.parse_args()

# --- Setup

# Set parameters
level = int(args.level or 0)
kwargs = {
    'level': level,
    'save_timeseries': True,

    # Spatial discretisation
    # 'family': 'dg-cg',
    'family': 'cg-cg',
    # 'stabilisation': 'lax_friedrichs',
    'stabilisation': None,
    'use_automatic_sipg_parameter': False,  # the problem is inviscid

    # Adjoint
    'control_parameter': 10.0,
    'optimal_value': 5.0,
    'artificial': True,
    'qoi_scaling': 1.0e-12,

    # Misc
    'plot_pvd': False,
    'debug': bool(args.debug or False),
}
nonlinear = False  # TODO
op = TohokuGaussianBasisOptions(fpath='discrete', **kwargs)

# Solve the forward problem to get data with 'optimal' control parameter m = 5
with stop_annotating():
    op.control_parameter.assign(kwargs['optimal_value'])
    swp = AdaptiveProblem(op, nonlinear=nonlinear, checkpointing=False)
    swp.solve_forward()
    for gauge in op.gauges:
        op.gauges[gauge]["data"] = op.gauges[gauge]["timeseries"]
    del swp


# --- Discrete adjoint

class DiscreteAdjointTsunamiProblem(AdaptiveDiscreteAdjointProblem):
    """The subclass exists to pass the QoI as required."""
    def quantity_of_interest(self):
        return self.op.J


# Solve the forward problem with 'suboptimal' control parameter m = 10
op.control_parameter.assign(kwargs['control_parameter'], annotate=False)
swp = DiscreteAdjointTsunamiProblem(op, nonlinear=nonlinear, checkpointing=False)
swp.solve_forward()
J = op.J
print_output("Mean square error QoI = {:.4e}".format(J))

# Plot timeseries
gauges = list(op.gauges.keys())
N = int(np.ceil(np.sqrt(len(gauges))))
fig, axes = plt.subplots(nrows=N, ncols=N, figsize=(14, 12))
plotting_kwargs = {
    'markevery': 5,
}
T = np.array(op.times)/60
for i, gauge in enumerate(gauges):
    ax = axes[i//N, i % N]
    plotting_kwargs['label'] = "{:s} simulated (m = {:.1f})".format(gauge, kwargs['control_parameter'])
    ax.plot(T, op.gauges[gauge]['timeseries'], '--x', **plotting_kwargs)
    plotting_kwargs['label'] = "{:s} data (m = {:.1f})".format(gauge, kwargs['optimal_value'])
    ax.plot(T, op.gauges[gauge]['data'], '--x', **plotting_kwargs)
    ax.legend(loc='upper left')
    ax.set_xlabel('Time (min)')
    ax.set_ylabel('Elevation (m)')
for i in range(len(gauges) % N):
    axes[N-1, N-i-1].axes('off')
di = create_directory(os.path.join(op.di, 'plots'))
plt.tight_layout()
plt.savefig(os.path.join(di, 'single_bf_timeseries_level{:d}.pdf'.format(level)))

# TODO: Compare discrete vs continuous form of error using plot
# TODO: Compare discrete vs continuous form of error using norms / TV

# Compute gradient
g_discrete = swp.compute_gradient(Control(op.control_parameter)).dat.data[0]

# Check consistency of by-hand gradient formula
swp.get_solve_blocks()
swp.save_adjoint_trajectory()
g_by_hand_discrete = assemble(inner(op.basis_function, swp.adj_solutions[0])*dx)
print_output("Gradient computed by hand (discrete): {:.4e}".format(g_by_hand_discrete))
relative_error = abs((g_discrete - g_by_hand_discrete)/g_discrete)
assert np.allclose(relative_error, 0.0)
swp.clear_tape()
del swp
stop_annotating()


# --- Continuous adjoint

# Solve the forward problem with 'suboptimal' control parameter m = 10, checkpointing state
op.di = create_directory(op.di.replace('discrete', 'continuous'))
swp = AdaptiveProblem(op, nonlinear=nonlinear, checkpointing=True)
swp.solve_forward()
J = op.J

# Solve adjoint equation in continuous form
swp.solve_adjoint()
g_continuous = assemble(inner(op.basis_function, swp.adj_solutions[0])*dx)
print_output("Gradient computed by hand (continuous): {:.4e}".format(g_continuous))
relative_error = abs((g_discrete - g_continuous)/g_discrete)
print_output("Relative error (discrete vs. continuous): {:.4f}%".format(100*relative_error))


# --- Finite differences

# Establish gradient using finite differences
epsilon = 1.0
converged = False
rtol = 1.0e-05
g_fd_ = None
op.save_timeseries = False
while not converged:
    swp = AdaptiveProblem(op, nonlinear=nonlinear, checkpointing=False)
    op.control_parameter.assign(kwargs['control_parameter'] + epsilon)
    swp.solve_forward(plot_pvd=False)
    J_step = op.J
    g_fd = (J_step - J)/epsilon
    print_output("J(epsilon=0) = {:.8e}  J(epsilon={:.1e}) = {:.8e}".format(J, epsilon, J_step))
    if g_fd_ is not None:
        print_output("gradient = {:.8e}  difference = {:.8e}".format(g_fd, abs(g_fd - g_fd_)))
        if abs(g_fd - g_fd_) < rtol*J:
            converged = True
        elif epsilon < 1.0e-10:
            raise ConvergenceError
    epsilon *= 0.5
    g_fd_ = g_fd
    del swp

# Logging
logstr = "elements: {:d}\n".format(swp.meshes[0].num_cells())
logstr += "finite difference gradient (rtol={:.1e}): {:.4e}\n".format(rtol, g_fd)
logstr += "discrete gradient: {:.4e}\n".format(g_discrete)
logstr += "continuous gradient: {:.4e}\n".format(g_continuous)
print_output(logstr)
fname = "outputs/fixed_mesh/gradient_at_{:d}_level{:d}.log"
with open(fname.format(int(kwargs['control_parameter']), level), 'w') as logfile:
    logfile.write(logstr)
