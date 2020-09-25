from thetis import *
from firedrake_adjoint import *

import adolc
import matplotlib.pyplot as plt
import numpy as np
import os
# import scipy
import sys

from adapt_utils.argparse import ArgumentParser
from adapt_utils.case_studies.tohoku.options.okada_options import TohokuOkadaBasisOptions
from adapt_utils.case_studies.tohoku.options.radial_options import TohokuRadialBasisOptions
from adapt_utils.norms import vecnorm
from adapt_utils.plotting import *
from adapt_utils.unsteady.solver_adjoint import AdaptiveDiscreteAdjointProblem
from adapt_utils.unsteady.swe.tsunami.conversion import lonlat_to_utm


# --- Parse arguments

parser = ArgumentParser(
    prog="invert",
    description="""
            Invert for an initial condition defined in an Okada basis over an array.

            GPS and pressure gauge data are obtained from post-processed versions of observations.

            The gradient of the square gauge timeseries misfit w.r.t. the basis coefficients is
            computed using either a discrete or continuous adjoint approach.
            """,
    optimisation=True,
    plotting=True,
    shallow_water=True,
)
parser.add_argument("-noisy_data", help="Toggle whether to sample noisy data")
parser.add_argument("-zero_initial_guess", help="""
    Toggle between a zero initial guess and scaled Gaussian.
    """)
parser.add_argument("-gaussian_scaling", help="Scaling for Gaussian initial guess (default 6.0)")


# --- Set parameters

# Parsed arguments
args = parser.args
level = int(args.level or 0)
optimise = bool(args.rerun_optimisation or False)
gtol = float(args.gtol or 1.0e-04)
plot_pvd = bool(args.plot_pvd or False)
plot = parser.plotting_args()
timeseries = 'timeseries'
if bool(args.continuous_timeseries or False):
    timeseries = '_'.join([timeseries, 'smooth'])

# Do not attempt to plot in parallel
if COMM_WORLD.size > 1 and plot['any']:
    print_output(120*'*' + "\nWARNING: Plotting turned off when running in parallel.\n" + 120*'*')
    plot.pdf = plot.png = False

# Setup output directories
dirname = os.path.dirname(__file__)
di = create_directory(os.path.join(dirname, 'outputs', 'realistic'))
if args.extension is not None:
    di = '_'.join([di, args.extension])
di = create_directory(os.path.join(di, 'discrete'))
plot_dir = create_directory(os.path.join(di, 'plots'))
create_directory(os.path.join(plot_dir, 'discrete'))

# Collect initialisation parameters
nonlinear = bool(args.nonlinear or False)
family = args.family or 'dg-cg'
stabilisation = args.stabilisation or 'lax_friedrichs'
if stabilisation == 'none' or family == 'cg-cg' or not nonlinear:
    stabilisation = None
zero_init = bool(args.zero_initial_guess or False)
taylor = bool(args.taylor_test or False)
kwargs = {
    'level': level,
    'save_timeseries': True,

    # Spatial discretisation
    'family': family,
    'stabilisation': stabilisation,
    'use_automatic_sipg_parameter': False,  # the problem is inviscid

    # Inversion
    'synthetic': False,
    'qoi_scaling': 1.0,
    'noisy_data': bool(args.noisy_data or False),

    # I/O and debugging
    'plot_pvd': False,
    'debug': bool(args.debug or False),
    'debug_mode': args.debug_mode or 'basic',
    'di': di,
}
if args.end_time is not None:
    kwargs['end_time'] = float(args.end_time)

# Construct Options parameter class
gaussian_scaling = float(args.gaussian_scaling or 6.0)
op = TohokuOkadaBasisOptions(**kwargs)
op.active_control = ('slip', 'rake')
op.dirty_cache = bool(args.dirty_cache or False)
gauges = list(op.gauges.keys())


# --- Set initial guess

with stop_annotating():
    if zero_init:
        eps = 1.0e-03  # zero gives an error so just choose small
        kwargs['control_parameters'] = eps*np.ones(op.nx*op.ny)
    else:
        # TODO: Cache the field itself
        print_output("Projecting initial guess...")

        # Create Radial parameter object
        kwargs_src = kwargs.copy()
        kwargs_src['control_parameters'] = [gaussian_scaling]
        kwargs_src['nx'], kwargs_src['ny'] = 1, 1
        op_src = TohokuRadialBasisOptions(mesh=op.default_mesh, **kwargs_src)
        swp = AdaptiveDiscreteAdjointProblem(op_src, nonlinear=nonlinear, print_progress=op.debug)
        swp.set_initial_condition()
        f_src = swp.fwd_solutions[0].split()[1]

        # Project into Okada basis
        swp = AdaptiveDiscreteAdjointProblem(op, nonlinear=nonlinear, print_progress=op.debug)
        op.project(swp, f_src)
        kwargs['control_parameters'] = op.control_parameters.copy()

        # Plot
        if plot.any:
            levels = np.linspace(-0.1*gaussian_scaling, 1.1*gaussian_scaling, 51)
            ticks = np.linspace(0, gaussian_scaling, 5)

            # Project into P1 for plotting
            swp.set_initial_condition()
            f = project(swp.fwd_solutions[0].split()[1], swp.P1[0])

            # Print L2 error
            msg = "Relative l2 error in initial guess projection: {:.2f}%"
            print_output(msg.format(100*errornorm(f, f_src)/norm(f)))

            # Get corners of zoom
            lonlat_corners = [(138, 32), (148, 42), (138, 42)]
            utm_corners = [lonlat_to_utm(*corner, 54) for corner in lonlat_corners]

            # Plot initial guess
            fig, axes = plt.subplots(figsize=(4.5, 4))
            cbar = fig.colorbar(tricontourf(f, axes=axes, levels=levels, cmap='coolwarm'), ax=axes)
            cbar.set_ticks(ticks)
            axes.set_xlim([utm_corners[0][0], utm_corners[1][0]])
            axes.set_ylim([utm_corners[0][1], utm_corners[2][1]])
            axes.axis(False)
            fname = 'initial_guess_okada_{:d}'.format(level)
            savefig(fname, plot_dir, extensions=plot.extensions)
if plot.only:
    sys.exit(0)


# --- Tracing

# Set initial guess
op = TohokuOkadaBasisOptions(**kwargs)
op.active_control = ('slip', 'rake')
num_active_controls = len(op.active_controls)
swp = AdaptiveDiscreteAdjointProblem(op, nonlinear=nonlinear, print_progress=op.debug)
swp.clear_tape()
print_output("Setting initial guess...")
op.assign_control_parameters(kwargs['control_parameters'])
num_subfaults = len(op.subfaults)

# Annotate the source model to ADOL-C's tape
tape_tag = 0
swp.set_initial_condition(annotate_source=True, tag=tape_tag, separate_faults=False)
if op.debug:
    print(adolc.tapestats(tape_tag))

# Annotate the tsunami model to pyadjoint's tape
control = Control(swp.fwd_solutions[0])
swp.setup_solver_forward_step(0)
print_output("Run forward to get initial timeseries...")
swp.solve_forward_step(0)
J = swp.quantity_of_interest()

# Save data
fnames = [os.path.join(di, '{:s}_{:s}_{:d}'.format(gauge, timeseries, level)) for gauge in gauges]
fnames_data = [os.path.join(di, '{:s}_data_{:d}'.format(gauge, level)) for gauge in gauges]
for gauge, fname, fname_data in zip(gauges, fnames, fnames_data):
    np.save(fname, op.gauges[gauge][timeseries])
    np.save(fname_data, op.gauges[gauge]['data'])


# ---  Define reduced functional and gradient functions

Jhat = ReducedFunctional(J, control)
stop_annotating()


def reduced_functional(m):
    """
    Compose both unrolled tapes
    """

    # Unroll ADOL-C's tape
    op.set_initial_condition(swp, unroll_tape=True, separate_faults=False)

    # Unroll pyadjoint's tape
    return Jhat(swp.fwd_solutions[0])


def gradient(m):
    """
    Apply the chain rule to both tapes
    """

    # Reverse propagate on pyadjoint's tape
    dJdq0 = Jhat.derivative()

    # Restrict to source region
    dJdeta0 = interpolate(dJdq0.split()[1], swp.P1[0])
    dJdeta0 = dJeta0.dat.data[op.indices]

    # Reverse propagate on ADOL-C's tape
    return adolc.fos_reverse(tape_tag, dJdeta0)


# --- Taylor test

if taylor:
    raise NotImplementedError  # TODO

# --- Optimisation

raise NotImplementedError  # TODO

fname = os.path.join(di, 'optimisation_progress_{:s}' + '_{:d}.npy'.format(level))
if optimise:
    control_values_opt = []
    func_values_opt = []
    gradient_values_opt = []

    opt_kwargs = {'maxiter': 1000, 'gtol': gtol}
    print_output("Optimisation begin...")

    def derivative_cb_post(j, dj, m):
        """
        Callback for saving progress data to file during discrete adjoint inversion.
        """
        control = [mi.dat.data[0] for mi in m]
        djdm = [dji.dat.data[0] for dji in dj]
        msg = "functional {:15.8e}  gradient {:15.8e}"
        print_output(msg.format(j, vecnorm(djdm, order=np.Inf)))

        # Save progress to NumPy arrays on-the-fly
        control_values_opt.append(control)
        func_values_opt.append(j)
        gradient_values_opt.append(djdm)
        np.save(fname.format('ctrl'), np.array(control_values_opt))
        np.save(fname.format('func'), np.array(func_values_opt))
        np.save(fname.format('grad'), np.array(gradient_values_opt))

    # Run BFGS optimisation
    Jhat = ReducedFunctional(J, control, derivative_cb_post=derivative_cb_post)
    optimised_value = minimize(Jhat, method='BFGS', options=opt_kwargs)
    optimised_value = [m.dat.data[0] for m in optimised_value]
else:
    optimised_value = np.load(fname.format('ctrl'))[-1]


# --- Run with optimised control

# Run forward again using the optimised control parameters
op.plot_pvd = plot_pvd
op.di = di
swp = AdaptiveDiscreteAdjointProblem(op, nonlinear=nonlinear, print_progress=op.debug)
swp.clear_tape()
print_output("Assigning optimised control parameters...")
op.assign_control_parameters(optimised_value)
print_output("Run to plot optimised timeseries...")
swp.solve_forward()

# Save timeseries to file
for gauge in gauges:
    fname = os.path.join(di, '{:s}_{:s}_{:d}.npy'.format(gauge, timeseries, level))
    np.save(fname, op.gauges[gauge][timeseries])

# Solve adjoint problem and plot solution fields
if plot_pvd:
    swp.solve_adjoint()
    swp.save_adjoint_trajectory()
