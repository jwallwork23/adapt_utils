from thetis import *
from firedrake_adjoint import *

import argparse
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import scipy

from adapt_utils.case_studies.tohoku.options.okada_options import TohokuOkadaOptions
from adapt_utils.case_studies.tohoku.options.gaussian_options import TohokuGaussianBasisOptions
from adapt_utils.unsteady.solver import AdaptiveProblem
from adapt_utils.unsteady.solver_adjoint import AdaptiveDiscreteAdjointProblem
from adapt_utils.unsteady.swe.tsunami.conversion import lonlat_to_utm
from adapt_utils.misc import rotation_matrix
from adapt_utils.norms import vecnorm


# --- Parse arguments

parser = argparse.ArgumentParser()

# Model
parser.add_argument("-level", help="Mesh resolution level")
parser.add_argument("-okada_grid_resolution", help="Mesh resolution level for the Okada grid")
parser.add_argument("-family", help="Finite element pair")
parser.add_argument("-stabilisation", help="Stabilisation approach")
parser.add_argument("-nonlinear", help="Toggle nonlinear model")

# Inversion
parser.add_argument("-initial_guess", help="Initial guess for control parameter")
parser.add_argument("-rerun_optimisation", help="Rerun optimisation routine")
parser.add_argument("-real_data", help="Toggle whether to use real data")
parser.add_argument("-smooth_timeseries", help="Toggle discrete or smoothed timeseries data")

# I/O and debugging
parser.add_argument("-plot_only", help="Just plot parameter space, optimisation progress and timeseries")
parser.add_argument("-plot_pdf", help="Toggle plotting to .pdf")
parser.add_argument("-plot_png", help="Toggle plotting to .png")
parser.add_argument("-plot_pvd", help="Toggle plotting to .pvd")
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
plot_only = bool(args.plot_only or False)
if plot_only:
    plot_pdf = True
    plot_png = True
if optimise:
    assert not plot_only
real_data = bool(args.real_data or False)
use_smoothed_timeseries = bool(args.smooth_timeseries or False)
timeseries_type = "timeseries"
if use_smoothed_timeseries:
    timeseries_type = "_".join([timeseries_type, "smooth"])

N = int(args.okada_grid_resolution or 51)
nx = 13
ny = 10
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

    # I/O and debugging
    'plot_pvd': False,
    'debug': bool(args.debug or False),
    'debug_mode': args.debug_mode or 'basic',
}
nonlinear = bool(args.nonlinear or False)

if plot_pdf or plot_png:

    # Fonts
    matplotlib.rc('text', usetex=True)
    matplotlib.rcParams['mathtext.fontset'] = 'custom'
    matplotlib.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
    matplotlib.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
    matplotlib.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'
    matplotlib.rcParams['mathtext.fontset'] = 'stix'
    matplotlib.rcParams['font.family'] = 'STIXGeneral'

    # Plotting
    fontsize = 22
    fontsize_tick = 18
    plotting_kwargs = {'markevery': 5}

# Setup output directories
dirname = os.path.dirname(__file__)
di = create_directory(os.path.join(dirname, 'outputs', 'realistic' if real_data else 'synthetic'))
plot_dir = create_directory(os.path.join(di, 'plots'))
create_directory(os.path.join(plot_dir, 'discrete'))

# Zero initial guess gives an error so just choose small
eps = 1.0e-03

# --- Synthetic run to get timeseries data

if not real_data:
    with stop_annotating():
        print_output("Projecting optimal solution...")

        # Create Okada parameter class and set the default initial conditionm
        kwargs_okada = {"okada_grid_resolution": N}
        kwargs_okada.update(kwargs)
        op_okada = TohokuOkadaOptions(**kwargs_okada)
        swp = AdaptiveProblem(op_okada, nonlinear=nonlinear, print_progress=False)
        f_okada = op_okada.set_initial_condition(swp)

        # Create GaussianBasis parameter class and an associated AdaptiveProblem
        op = TohokuGaussianBasisOptions(mesh=op_okada.default_mesh, **kwargs)
        op.di = create_directory(os.path.join(di, 'discrete'))
        swp = AdaptiveProblem(op, nonlinear=nonlinear, print_progress=False)

        # Construct 'optimal' control vector by projection
        op.project(swp, f_okada)
        swp.set_initial_condition()

        # Plot optimum solution
        if plot_pdf or plot_png:

            # Get corners of zoom
            lonlat_corners = [(138, 32), (148, 42), (138, 42)]
            utm_corners = [lonlat_to_utm(*corner, 54) for corner in lonlat_corners]
            xlim = [utm_corners[0][0], utm_corners[1][0]]
            ylim = [utm_corners[0][1], utm_corners[2][1]]

            # Plot optimum in both (original) Okada basis and also in (projected) box basis
            fig, axes = plt.subplots(ncols=2, figsize=(9, 4))
            f_box = project(swp.fwd_solutions[0].split()[1], swp.P1[0])
            levels = np.linspace(-6, 16, 51)
            ticks = np.linspace(-5, 15, 9)
            for f, ax in zip((f_okada, f_box), (axes[0], axes[1])):
                cbar = fig.colorbar(tricontourf(f, axes=ax, levels=levels, cmap='coolwarm'), ax=ax)
                cbar.set_ticks(ticks)
                ax.set_xlim(xlim)
                ax.set_ylim(ylim)
                ax.axis(False)
            axes[0].set_title("Okada basis")
            axes[1].set_title("Radial basis")
            fname = os.path.join(plot_dir, 'optimum_{:d}'.format(level))
            if plot_pdf:
                plt.savefig(fname + '.pdf')
            if plot_png:
                plt.savefig(fname + '.png')
            plt.show()  # TODO: temp

        # Synthetic run
        if not plot_only:
            print_output("Run forward to get 'data'...")
            swp.setup_solver_forward(0)
            swp.solve_forward_step(0)
            for gauge in op.gauges:
                op.gauges[gauge]["data"] = op.gauges[gauge][timeseries_type]

# Set zero initial guess for the optimisation  # NOTE: zero gives an error so just choose small
eps = 1.0e-03
for control in op.control_parameters:
    control.assign(eps)

# --- Optimisation

gauges = list(op.gauges.keys())
if plot_only:

    # Load timeseries
    for gauge in gauges:
        fname = os.path.join(di, '_'.join([gauge, 'data', str(level) + '.npy']))
        op.gauges[gauge]['data'] = np.load(fname)
        fname = os.path.join(di, '_'.join([gauge, timeseries_type, str(level) + '.npy']))
        op.gauges[gauge][timeseries_type] = np.load(fname)

else:

    # Solve the forward problem with initial guess
    op.save_timeseries = True
    print_output("Run forward to get timeseries...")
    swp = AdaptiveProblem(op, nonlinear=nonlinear, print_progress=False)
    swp.solve_forward()
    J = op.J

    # Save timeseries
    for gauge in gauges:
        fname = os.path.join(di, '_'.join([gauge, 'data', str(level)]))
        np.save(fname, op.gauges[gauge]['data'])
        fname = os.path.join(di, '_'.join([gauge, timeseries_type, str(level)]))
        np.save(fname, op.gauges[gauge][timeseries_type])

# Plot timeseries
if plot_pdf or plot_png:
    N = int(np.ceil(np.sqrt(len(gauges))))
    fig, axes = plt.subplots(nrows=N, ncols=N, figsize=(14, 12))
    for i, gauge in enumerate(gauges):
        T = np.array(op.gauges[gauge]['times'])/60
        ax = axes[i//N, i % N]
        ax.plot(T, op.gauges[gauge]['data'], '--x', label=gauge + ' data', **plotting_kwargs)
        ax.plot(T, op.gauges[gauge][timeseries_type], '--x', label=gauge + ' simulated', **plotting_kwargs)
        ax.legend(loc='upper left')
        ax.set_xlabel('Time (min)', fontsize=fontsize)
        ax.set_ylabel('Elevation (m)', fontsize=fontsize)
        plt.xticks(fontsize=fontsize_tick)
        plt.yticks(fontsize=fontsize_tick)
        ax.grid()
    for i in range(len(gauges), N*N):
        axes[i//N, i % N].axis(False)
    plt.tight_layout()
    fname = os.path.join(plot_dir, 'timeseries_{:d}'.format(level))
    if plot_pdf:
        plt.savefig(fname + '.pdf')
    if plot_png:
        plt.savefig(fname + '.png')

fname = os.path.join(di, 'discrete', 'optimisation_progress_{:s}' + '_{:d}.npy'.format(level))
if np.all([os.path.exists(fname.format(ext)) for ext in ('ctrl', 'func', 'grad')]) and not optimise:

    # Load trajectory
    control_values_opt = np.load(fname.format('ctrl', level))
    print(control_values_opt.shape)
    func_values_opt = np.load(fname.format('func', level))
    gradient_values_opt = np.load(fname.format('grad', level))
    optimised_value = control_values_opt[-1]

else:

    # Arrays to log progress
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

        # # Stagnation termination condition
        # if len(func_values_opt) > 1:
        #     if abs(func_values_opt[-1] - func_values_opt[-2]) < 1.0e-06*abs(func_values_opt[-2]):
        #         raise StagnationError

    # Run BFGS optimisation
    opt_kwargs = {
        'maxiter': 100,
        'gtol': 1.0e-08,
    }
    print_output("Optimisation begin...")
    controls = [Control(c) for c in op.control_parameters]
    Jhat = ReducedFunctional(J, controls, derivative_cb_post=derivative_cb_post)
    optimised_value = minimize(Jhat, method='BFGS', options=opt_kwargs).dat.data
    # try:
    #     optimised_value = minimize(Jhat, method='BFGS', options=opt_kwargs).dat.data
    # except StagnationError:
    #     optimised_value = control_values_opt[-1]
    #     print_output("StagnationError: Stagnation of objective functional")

# Create a new parameter class
kwargs['control_parameters'] = optimised_value
kwargs['plot_pvd'] = plot_pvd
op_opt = TohokuBoxBasisOptions(**kwargs)

if plot_only:

    # Load timeseries
    for gauge in gauges:
        fname = os.path.join(op.di, '_'.join([gauge, timeseries_type, str(level) + '.npy']))
        op_opt.gauges[gauge][timeseries_type] = np.load(fname)

else:
    tape = get_working_tape()
    tape.clear_tape()

    class DiscreteAdjointTsunamiProblem(AdaptiveDiscreteAdjointProblem):
        """The subclass exists to pass the QoI as required."""
        def quantity_of_interest(self):
            return self.op.J

    # Run forward again so that we can compare timeseries
    gauges = list(op_opt.gauges.keys())
    for gauge in gauges:
        op_opt.gauges[gauge]["data"] = op.gauges[gauge]["data"]
    print_output("Run to plot optimised timeseries...")
    swp = DiscreteAdjointTsunamiProblem(op_opt, nonlinear=nonlinear, print_progress=False)
    swp.solve_forward()
    J = swp.quantity_of_interest()

    # Compare total variation
    msg = "total variation for gauge {:s}: before {:.4e}  after {:.4e} reduction  {:.1f}%"
    print_output("\nContinuous form QoI:")
    for gauge in op.gauges:
        tv = total_variation(op.gauges[gauge]['diff_smooth'])
        tv_opt = total_variation(op_opt.gauges[gauge]['diff_smooth'])
        print_output(msg.format(gauge, tv, tv_opt, 100*(1-tv_opt/tv)))
    print_output("\nDiscrete form QoI:")
    for gauge in op.gauges:
        tv = total_variation(op.gauges[gauge]['diff'])
        tv_opt = total_variation(op_opt.gauges[gauge]['diff'])
        print_output(msg.format(gauge, tv, tv_opt, 100*(1-tv_opt/tv)))

    # Solve adjoint problem and plot solution fields
    if plot_pvd:
        swp.compute_gradient(Control(op_opt.control_parameters[0]))
        swp.get_solve_blocks()
        swp.save_adjoint_trajectory()

# Plot timeseries for both initial guess and optimised control
if plot_pdf or plot_png:
    fig, axes = plt.subplots(nrows=N, ncols=N, figsize=(14, 12))
    for i, gauge in enumerate(gauges):
        T = np.array(op.gauges[gauge]['times'])/60
        ax = axes[i//N, i % N]
        ax.plot(T, op.gauges[gauge]['data'], '--x', label=gauge + ' data', **plotting_kwargs)
        ax.plot(T, op.gauges[gauge][timeseries_type], '--x', label=gauge + ' initial guess', **plotting_kwargs)
        ax.plot(T, op_opt.gauges[gauge][timeseries_type], '--x', label=gauge + ' optimised', **plotting_kwargs)
        ax.legend(loc='upper left')
        ax.set_xlabel('Time (min)', fontsize=fontsize)
        ax.set_ylabel('Elevation (m)', fontsize=fontsize)
        plt.xticks(fontsize=fontsize_tick)
        plt.yticks(fontsize=fontsize_tick)
        ax.grid()
    for i in range(len(gauges), N*N):
        axes[i//N, i % N].axis(False)
    plt.tight_layout()
    fname = os.path.join(plot_dir, 'discrete', 'timeseries_optimised_{:d}'.format(level))
    if plot_pdf:
        plt.savefig(fname + '.pdf')
    if plot_png:
        plt.savefig(fname + '.png')
print_output("Done!")
