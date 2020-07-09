from __future__ import absolute_import

from thetis import *
from firedrake_adjoint import *
from firedrake.adjoint.blocks import GenericSolveBlock

import argparse
import matplotlib.pyplot as plt
import os

from adapt_utils.unsteady.solver import AdaptiveProblem
from adapt_utils.case_studies.tohoku.options import TohokuOptions
from adapt_utils.norms import total_variation
from adapt_utils.misc import gaussian, ellipse


parser = argparse.ArgumentParser()
parser.add_argument("-level", help="Mesh resolution level")
args = parser.parse_args()


class TohokuGaussianBasisOptions(TohokuOptions):
    """
    Initialise the free surface with initial condition consisting of a single Gaussian basis function
    scaled by a control parameter.
    """
    def __init__(self, control_parameter=10.0, **kwargs):
        super(TohokuGaussianBasisOptions, self).__init__(**kwargs)
        R = FunctionSpace(self.default_mesh, "R", 0)
        self.control_parameter = Function(R, name="Control parameter")
        self.control_parameter.assign(control_parameter)

    def set_initial_condition(self, prob):
        basis_function = Function(prob.V[0])
        psi, self.phi = basis_function.split()
        loc = (0.7e+06, 4.2e+06)
        radii = (48e+03, 96e+03)
        angle = pi/12
        self.phi.interpolate(gaussian([loc + radii, ], prob.meshes[0], rotation=angle))

        prob.fwd_solutions[0].project(self.control_parameter*basis_function)

# Set parameters
level = int(args.level or 0)
kwargs = {
    'level': level,

    # Spatial discretisation
    'family': 'dg-cg',
    'stabilisation': 'lax_friedrichs',
    'use_automatic_sipg_parameter': False,  # the problem is inviscid

    # Misc
    'debug': True,
}
op = TohokuGaussianBasisOptions(**kwargs)

# Only consider gauges which lie within the spatial domain
gauges = list(op.gauges.keys())
for gauge in gauges:
    try:
        op.default_mesh.coordinates.at(op.gauges[gauge]['coords'])
    except PointNotInDomainError:
        op.print_debug("NOTE: Gauge {:5s} is not in the domain and so was removed".format(gauge))
        op.gauges.pop(gauge)  # Some gauges aren't within the domain
gauges = list(op.gauges.keys())

# Solve the forward problem
swp = AdaptiveProblem(op)
swp.solve_forward()
print("Quantity of interest = {:.4e}".format(op.J))

# Plot timeseries
fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(14, 12))
for i, gauge in enumerate(gauges):
    ax = axes[i//3, i%3]
    ax.plot(np.array(op.times)/60, op.gauges[gauge]['timeseries'], '--x', label=gauge + ' simulated')
    ax.plot(np.array(op.times)/60, op.gauges[gauge]['data'], '--x', label=gauge + ' data')
    ax.legend(loc='upper right')
    ax.set_xlabel('Time (min)')
    ax.set_ylabel('Elevation (m)')
di = create_directory(os.path.join(op.di, 'plots'))
plt.savefig(os.path.join(di, 'single_bf_timeseries_level{:d}.pdf'.format(level)))

# TODO: Compare discrete vs continuous form of error using plot
# TODO: Compare discrete vs continuous form of error using norms / TV

# Compute gradient
g_discrete = compute_gradient(op.J, Control(op.control_parameter)).dat.data[0]

# Check consistency of by-hand gradient formula
tape = get_working_tape()
solve_blocks = [block for block in tape.get_blocks() if isinstance(block, GenericSolveBlock)]
g_by_hand_discrete = assemble(op.phi*solve_blocks[0].adj_sol.split()[1]*dx)
print("Gradient computed by hand (discrete): {:.4e}".format(g_by_hand_discrete))
print("Relative error: {:.4f}%".format(100*abs((g_discrete - g_by_hand_discrete)/g_discrete)))

# TODO: Continuous adjoint (in nonlinear case with Manning friction)
