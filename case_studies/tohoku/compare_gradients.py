from thetis import *
from firedrake_adjoint import *
from firedrake.adjoint.blocks import GenericSolveBlock

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
args = parser.parse_args()

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
    'artificial': True,

    # Misc
    'debug': True,
}
nonlinear = False  # TODO
# scaling = 1.0e-10
scaling = 1.0
op = TohokuGaussianBasisOptions(fpath='discrete', **kwargs)

# Solve the forward problem to get data with 'optimal' control parameter m = 5
with stop_annotating():
    op.control_parameter.assign(5.0)
    swp = AdaptiveProblem(op, nonlinear=nonlinear, checkpointing=False)
    swp.solve_forward()
    for gauge in op.gauges:
        op.gauges[gauge]["data"] = op.gauges[gauge]["timeseries"]


class DiscreteAdjointTsunamiProblem(AdaptiveDiscreteAdjointProblem):
    """The subclass exists to pass the QoI as required."""
    def quantity_of_interest(self):
        return self.op.J


# Solve the forward problem with 'suboptimal' control parameter m = 10
op.control_parameter.assign(10.0, annotate=False)
swp = DiscreteAdjointTsunamiProblem(op, nonlinear=False, checkpointing=False)
swp.solve_forward()
J = op.J
print_output("Mean square error QoI = {:.4e}".format(J*scaling))

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
    ax.plot(T, op.gauges[gauge]['timeseries'], '--x', label=gauge + ' simulated', **plotting_kwargs)
    ax.plot(T, op.gauges[gauge]['data'], '--x', label=gauge + ' data', **plotting_kwargs)
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
g_discrete = swp.compute_gradient(Control(op.control_parameter), scaling=scaling).dat.data[0]

# Check consistency of by-hand gradient formula
swp.get_solve_blocks()
swp.save_adjoint_trajectory()
g_by_hand_discrete = assemble(inner(op.basis_function, swp.adj_solutions[0])*dx)
print_output("Gradient computed by hand (discrete): {:.4e}".format(g_by_hand_discrete))
relative_error = abs((g_discrete - g_by_hand_discrete)/g_discrete)
print_output("Relative error (discrete vs. discrete by hand): {:.4f}%".format(100*relative_error))
assert np.allclose(relative_error, 0.0)
swp.clear_tape()
stop_annotating()

# Solve the forward problem with 'suboptimal' control parameter m = 10, checkpointing state
op.di = create_directory(op.di.replace('discrete', 'continuous'))
swp = DiscreteAdjointTsunamiProblem(op, nonlinear=False, checkpointing=True)
swp.solve_forward()

# Solve adjoint equation in continuous form
swp.solve_adjoint()
g_continuous = assemble(inner(op.basis_function, swp.adj_solutions[0])*dx)*scaling
print_output("Gradient computed by hand (continuous): {:.4e}".format(g_continuous))
relative_error = abs((g_discrete - g_continuous)/g_discrete)
print_output("Relative error (discrete vs. continuous): {:.4f}%".format(100*relative_error))
with open('outputs/fixed_mesh/gradient_{:d}.log'.format(level), 'w') as logfile:
    logfile.write("elements: {:d}".format(swp.meshes[0].num_cells()))
    # TODO: finite difference gradient
    logfile.write("discrete gradient: {:.4e}".format(g_discrete))
    logfile.write("continuous gradient: {:.4e}".format(g_continuous))
