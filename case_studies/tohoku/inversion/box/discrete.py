from thetis import *
from firedrake_adjoint import *

import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

from adapt_utils.case_studies.tohoku.options.okada_options import TohokuOkadaBasisOptions
from adapt_utils.case_studies.tohoku.options.box_options import TohokuBoxBasisOptions
from adapt_utils.plotting import *
from adapt_utils.norms import total_variation, vecnorm
from adapt_utils.unsteady.solver import AdaptiveProblem
from adapt_utils.unsteady.solver_adjoint import AdaptiveDiscreteAdjointProblem
from adapt_utils.unsteady.swe.tsunami.conversion import lonlat_to_utm


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
extensions = []
if plot_pdf:
    extensions.append('pdf')
if plot_png:
    extensions.append('png')
plot_any = len(extensions) > 0
real_data = bool(args.real_data or False)
timeseries_type = "timeseries"
if bool(args.continuous_timeseries or False):
    timeseries_type = "_".join([timeseries_type, "smooth"])

# Do not attempt to plot in parallel
if COMM_WORLD.size > 1 and plot_any:
    print_output(120*'*' + "\nWARNING: Plotting turned off when running in parallel.\n" + 120*'*')
    plot_pdf = plot_png = False

# Collect initialisation parameters
N = int(args.okada_grid_resolution or 51)
kwargs = {
    'level': level,
    'save_timeseries': True,

    # Spatial discretisation
    'family': args.family or 'dg-cg',
    'stabilisation': args.stabilisation,
    'use_automatic_sipg_parameter': False,  # the problem is inviscid

    # Inversion
    'synthetic': not real_data,
    'qoi_scaling': 1.0,
    'noisy_data': bool(args.noisy_data or False),

    # I/O and debugging
    'plot_pvd': False,
    'debug': bool(args.debug or False),
    'debug_mode': args.debug_mode or 'basic',
}
nonlinear = bool(args.nonlinear or False)
op = TohokuBoxBasisOptions(**kwargs)

# Setup output directories
dirname = os.path.dirname(__file__)
di = create_directory(os.path.join(dirname, 'outputs', 'realistic' if real_data else 'synthetic'))
op.di = create_directory(os.path.join(di, 'discrete'))
plot_dir = create_directory(os.path.join(di, 'plots'))
create_directory(os.path.join(plot_dir, 'discrete'))


# --- Synthetic run to get timeseries data

# Project optimal solution in the 'synthetic' case
if not real_data:
    with stop_annotating():
        print_output("Projecting optimal solution...")

        # Create Okada parameter class and set the default initial conditionm
        kwargs_okada = {"okada_grid_resolution": N}
        kwargs_okada.update(kwargs)
        op_okada = TohokuOkadaBasisOptions(mesh=op.default_mesh, **kwargs_okada)
        swp = AdaptiveProblem(op_okada, nonlinear=nonlinear, print_progress=op.debug)
        f_okada = op_okada.set_initial_condition(swp)

        # Construct 'optimal' control vector by projection
        swp = AdaptiveProblem(op, nonlinear=nonlinear, print_progress=op.debug)
        op.project(swp, f_okada)
        # op.interpolate(swp, f_okada)
        swp.set_initial_condition()

        # Plot optimum solution
        if plot_pdf or plot_png:
            levels = np.linspace(-6, 16, 51)
            ticks = np.linspace(-5, 15, 9)

            # Project into P1 for plotting
            f_box = project(swp.fwd_solutions[0].split()[1], swp.P1[0])

            # Get corners of zoom
            lonlat_corners = [(138, 32), (148, 42), (138, 42)]
            utm_corners = [lonlat_to_utm(*corner, 54) for corner in lonlat_corners]
            xlim = [utm_corners[0][0], utm_corners[1][0]]
            ylim = [utm_corners[0][1], utm_corners[2][1]]

            # Plot optimum in both (original) Okada basis and also in (projected) box basis
            for f, name in zip((f_okada, f_box), ('okada', 'box')):
                fig, axes = plt.subplots(figsize=(4.5, 4))
                cbar = fig.colorbar(tricontourf(f, axes=axes, levels=levels, cmap='coolwarm'), ax=axes)
                cbar.set_ticks(ticks)
                axes.set_xlim(xlim)
                axes.set_ylim(ylim)
                axes.axis(False)
                fname = 'optimum_{:s}_{:d}'.format(name, level)
                savefig(fname, fpath=plot_dir, extensions=extensions)

        # Synthetic run
        if plot_only:
            sys.exit(0)
        print_output("Run forward to get 'data'...")
        swp.setup_solver_forward(0)
        swp.solve_forward_step(0)
        for gauge in op.gauges:
            op.gauges[gauge]["data"] = op.gauges[gauge][timeseries_type]


# --- Optimisation

gauges = list(op.gauges.keys())

# Set zero initial guess for the optimisation  # NOTE: zero gives an error so just choose small
eps = 1.0e-03
for control in op.control_parameters:
    control.assign(eps)

# Solve the forward problem with initial guess
op.save_timeseries = True
print_output("Run forward to get timeseries...")
swp = AdaptiveProblem(op, nonlinear=nonlinear, print_progress=op.debug)
swp.solve_forward()
J = op.J

# Save timeseries
for gauge in gauges:
    fname = os.path.join(di, '_'.join([gauge, 'data', str(level)]))
    np.save(fname, op.gauges[gauge]['data'])
    fname = os.path.join(di, '_'.join([gauge, timeseries_type, str(level)]))
    np.save(fname, op.gauges[gauge][timeseries_type])

# Arrays to log progress
fname = os.path.join(di, 'discrete', 'optimisation_progress_{:s}' + '_{:d}.npy'.format(level))
control_values_opt = []
func_values_opt = []
gradient_values_opt = []


def derivative_cb_post(j, dj, m):
    control = [mi.dat.data[0] for mi in m]
    djdm = [dji.dat.data[0] for dji in dj]
    print_output("functional {:.8e}  gradient {:.8e}".format(j, vecnorm(djdm, order=np.Inf)))

    # Save progress to NumPy arrays on-the-fly
    control_values_opt.append(control)
    func_values_opt.append(j)
    gradient_values_opt.append(djdm)
    np.save(fname.format('ctrl'), np.array(control_values_opt))
    np.save(fname.format('func'), np.array(func_values_opt))
    np.save(fname.format('grad'), np.array(gradient_values_opt))


# Run BFGS optimisation
opt_kwargs = {'maxiter': 1000, 'gtol': 1.0e-08}
print_output("Optimisation begin...")
controls = [Control(c) for c in op.control_parameters]
Jhat = ReducedFunctional(J, controls, derivative_cb_post=derivative_cb_post)
optimised_value = [o.dat.data[0] for o in minimize(Jhat, method='BFGS', options=opt_kwargs)]

# Create a new parameter class
kwargs['control_parameters'] = optimised_value
kwargs['plot_pvd'] = plot_pvd
op_opt = TohokuBoxBasisOptions(**kwargs)
gauges = list(op_opt.gauges.keys())
for gauge in gauges:
    op_opt.gauges[gauge]["data"] = op.gauges[gauge]["data"]

# Clear tape
tape = get_working_tape()
tape.clear_tape()

# Plot optimised source against optimum
swp = DiscreteAdjointTsunamiProblem(op_opt, nonlinear=nonlinear, print_progress=op.debug)
swp.set_initial_condition()
plot_dir = os.path.join(plot_dir, 'discrete')
if not real_data and plot_any:
    fig, axes = plt.subplots(ncols=2, figsize=(9, 4))
    f_opt = project(swp.fwd_solutions[0].split()[1], swp.P1[0])
    levels = np.linspace(-6, 16, 51)
    ticks = np.linspace(-5, 15, 9)
    for f, ax in zip((f_box, f_opt), (axes[0], axes[1])):
        cbar = fig.colorbar(tricontourf(f, axes=ax, levels=levels, cmap='coolwarm'), ax=ax)
        cbar.set_ticks(ticks)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.axis(False)
    axes[0].set_title("Optimum source field")
    axes[1].set_title("Optimised source field")
    savefig('optimised_source_{:d}'.format(level), fpath=plot_dir, extensions=extensions)


# --- Compare timeseries

# Run forward again so that we can compare timeseries
print_output("Run to plot optimised timeseries...")
swp.setup_solver_forward(0)
swp.solve_forward_step(0)
J = swp.quantity_of_interest()

# Save timeseries
for gauge in gauges:
    fname = os.path.join(op.di, '_'.join([gauge, timeseries_type, str(level)]))
    np.save(fname, op_opt.gauges[gauge][timeseries_type])

# Compare total variation
msg = "total variation for gauge {:s}:  before {:.4e}  after {:.4e}  reduction {:.1f}%"
for tt, cd in zip(('diff', 'diff_smooth'), ('Continuous', 'Discrete')):
    print_output("\n{:s} form QoI:".format(cd))
    for gauge in gauges:
        tv = total_variation(op.gauges[gauge][tt])
        tv_opt = total_variation(op_opt.gauges[gauge][tt])
        print_output(msg.format(gauge, tv, tv_opt, 100*(1-tv_opt/tv)))

# Solve adjoint problem and plot solution fields
if plot_pvd:
    swp.compute_gradient(Control(op_opt.control_parameters[0]))  # TODO: Use solve_adjoint
    swp.get_solve_blocks()
    swp.save_adjoint_trajectory()
