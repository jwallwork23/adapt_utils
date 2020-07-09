from thetis import *
from firedrake_adjoint import *
from firedrake.adjoint.blocks import GenericSolveBlock

import argparse
import matplotlib.pyplot as plt
import numpy as np
import os

from adapt_utils.unsteady.solver import AdaptiveProblem
from adapt_utils.case_studies.tohoku.options import *
from adapt_utils.norms import total_variation


parser = argparse.ArgumentParser()
parser.add_argument("-level", help="Mesh resolution level")
args = parser.parse_args()

# Set parameters
level = int(args.level or 0)
kwargs = {
    'level': level,
    'save_timeseries': False,

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

# Explore parameter space
n = 9
control_values = np.linspace(2.0, 10.0, n)
fname = os.path.join(op.di, 'parameter_space_level{:d}.npy'.format(level))
with stop_annotating():
    swp = AdaptiveProblem(op)
    if os.path.exists(fname):
        func_values = np.load(fname)
    else:
        func_values = np.zeros(n)
        for i, m in enumerate(control_values):
            op.control_parameter.assign(m)
            swp.set_initial_condition()
            swp.solve_forward()
            func_values[i] = op.J/len(op.times)
    np.save(fname, func_values)
for i, m in enumerate(control_values):
    print_output("{:2d}: control value {:.4e}  functional value {:.4e}".format(i, m, func_values[i]))

# Plot parameter space
fig, axes = plt.subplots(figsize=(8, 8))
axes.plot(control_values, func_values, '--x')
axes.set_xlabel("Control values")
axes.set_ylabel("Mean square error QoI")
plt.tight_layout()
plt.savefig(os.path.join(op.di, 'plots', 'single_bf_parameter_space_level{:d}.pdf'.format(level)))

# --- Optimisation

# Solve the forward problem
op.save_timeseries = True
swp = AdaptiveProblem(op)
swp.solve_forward()
J = op.J/len(op.times)
print_output("Mean square error QoI = {:.4e}".format(J))

# Plot timeseries
N = int(np.ceil(np.sqrt(len(gauges))))
fig, axes = plt.subplots(nrows=N, ncols=N, figsize=(14, 12))
plotting_kwargs = {
    'markevery': 5,
}
T = np.array(op.times)/60
for i, gauge in enumerate(gauges):
    ax = axes[i//N, i%N]
    ax.plot(T, op.gauges[gauge]['timeseries'], '--x', label=gauge + ' simulated', **plotting_kwargs)
    ax.plot(T, op.gauges[gauge]['data'], '--x', label=gauge + ' data', **plotting_kwargs)
    ax.legend(loc='upper left')
    ax.set_xlabel('Time (min)')
    ax.set_ylabel('Elevation (m)')
for i in range(len(gauges)%N):
    axes[N-1, N-i-1].axes('off')
di = create_directory(os.path.join(op.di, 'plots'))
plt.tight_layout()
plt.savefig(os.path.join(di, 'plots', 'single_bf_timeseries_level{:d}.pdf'.format(level)))

# Arrays to log progress
control_values_opt = []
func_values_opt = []
gradient_values_opt = []

def derivative_cb_post(j, dj, m):
    control = m.dat.data[0]
    control_values_opt.append(control)
    func_values_opt.append(j)
    gradient = dj.dat.data[0]
    gradient_values_opt.append(gradient)
    print_output("control {:.8e}  functional  {:.8e}  gradient {:.8e}".format(control, j, gradient))

# Run BFGS optimisation and store trajectory
opt_kwargs = {  # TODO: Tighter tolerances
    'maxiter': 10,
    'gtol': 1.0,
}
Jhat = ReducedFunctional(J, Control(op.control_parameter), derivative_cb_post=derivative_cb_post)
m_opt = minimize(Jhat, method='BFGS', options=opt_kwargs)
fname = os.path.join(op.di, 'optimisation_progress_controls_level{:d}.npy'.format(level))
np.save(fname, np.array(control_values_opt))
fname = os.path.join(op.di, 'optimisation_progress_funcs_level{:d}.npy'.format(level))
np.save(fname, np.array(func_values_opt))

# Plot progress of optimisation routine
fig, axes = plt.subplots(figsize=(8, 8))
axes.plot(control_values, func_values, '--x')
axes.plot(control_values_opt, func_values_opt, 'o', color='r')
delta_m = 0.25
for m, f, g in zip(control_values_opt, func_values_opt, gradient_values_opt):
    x = np.array([m - delta_m, m + delta_m])
    axes.plot(x, g*(x-m) + f, '-', color='g')
axes.set_xlabel("Control values");
axes.set_ylabel("Functional values");
plt.tight_layout()
plt.savefig(os.path.join(di, 'plots', 'single_bf_optimisation_discrete_level{:d}.pdf'.format(level)))

# TODO: Run forward again to compare timeseries
