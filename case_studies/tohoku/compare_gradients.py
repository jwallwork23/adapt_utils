from thetis import *
from firedrake_adjoint import *
from firedrake.adjoint.blocks import GenericSolveBlock

import argparse
import matplotlib.pyplot as plt
import numpy as np
import os

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

    # Misc
    'debug': True,
    'artificial': True,
}
op = TohokuGaussianBasisOptions(fpath='discrete', **kwargs)

# Only consider gauges which lie within the spatial domain
gauges = list(op.gauges.keys())
for gauge in gauges:
    try:
        op.default_mesh.coordinates.at(op.gauges[gauge]['coords'])
    except PointNotInDomainError:
        op.print_debug("NOTE: Gauge {:5s} is not in the domain and so was removed".format(gauge))
        op.gauges.pop(gauge)  # Some gauges aren't within the domain
gauges = list(op.gauges.keys())

# Solve the forward problem to get data
with stop_annotating():
    swp = AdaptiveDiscreteAdjointProblem(op, nonlinear=False)
    swp.solve_forward()
    for gauge in op.gauges:
        op.gauges[gauge]["data"] = op.gauges[gauge]["timeseries"]

# Solve the forward problem
swp = AdaptiveDiscreteAdjointProblem(op, nonlinear=False)
swp.solve_forward()
scaling = 1.0e+10
# J = assemble(op.J/scaling)
J = op.J
print("Mean square error QoI = {:.4e}".format(J))

# Plot timeseries
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
g_discrete = compute_gradient(J, Control(op.control_parameter)).dat.data[0]
# TODO: g_discrete = swp.compute_gradient(Control(op.control_parameter)).dat.data[0]

# TODO: Taylor test discrete gradient

# Check consistency of by-hand gradient formula
swp.get_solve_blocks()
swp.save_adjoint_trajectory()
tape = get_working_tape()
solve_blocks = [block for block in tape.get_blocks() if isinstance(block, GenericSolveBlock)]
# g_by_hand_discrete = assemble(inner(op.basis_function, solve_blocks[0])*dx)
# # g_by_hand_discrete = assemble(inner(op.basis_function, solve_blocks[0])*dx)/scaling
g_by_hand_discrete = assemble(inner(op.basis_function, swp.adj_solutions[0])*dx)
# g_by_hand_discrete = assemble(inner(op.basis_function, swp.adj_solutions[0])*dx)/scaling
print("Gradient computed by hand (discrete): {:.4e}".format(g_by_hand_discrete))
relative_error = abs((g_discrete - g_by_hand_discrete)/g_discrete)
print("Relative error: {:.4f}%".format(100*relative_error))
assert np.allclose(relative_error, 0.0)

# TODO: Continuous adjoint (in nonlinear case with Manning friction)
# TODO: Taylor test continuous gradient
