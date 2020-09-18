from thetis import *
from firedrake_adjoint import *

import matplotlib.pyplot as plt
import numpy as np
import os
import sys

from adapt_utils.argparse import ArgumentParser
from adapt_utils.case_studies.tohoku.options.box_options import TohokuBoxBasisOptions
from adapt_utils.case_studies.tohoku.options.okada_options import TohokuOkadaBasisOptions
from adapt_utils.case_studies.tohoku.options.radial_options import TohokuRadialBasisOptions
from adapt_utils.norms import vecnorm
from adapt_utils.plotting import *
from adapt_utils.unsteady.solver_adjoint import AdaptiveDiscreteAdjointProblem
from adapt_utils.unsteady.swe.tsunami.conversion import lonlat_to_utm


class DiscreteAdjointTsunamiProblem(AdaptiveDiscreteAdjointProblem):
    """
    The subclass exists to pass the QoI as required.
    """
    def quantity_of_interest(self):
        return self.op.J


# --- Parse arguments

parser = ArgumentParser(basis=True, plotting=True, shallow_water=True)

# Resolution
parser.add_argument("-level", help="Mesh resolution level")

# Inversion
parser.add_argument("-rerun_optimisation", help="Rerun optimisation routine")
parser.add_argument("-noisy_data", help="Toggle whether to sample noisy data")
parser.add_argument("-continuous_timeseries", help="Toggle discrete or continuous timeseries")
parser.add_argument("-gtol", help="Gradient tolerance (default 1.0e-08)")
parser.add_argument("-zero_initial_guess", help="""
    Toggle between a zero initial guess and scaled Gaussian.
    """)
parser.add_argument("-gaussian_scaling", help="Scaling for Gaussian initial guess (default 6.0)")

# Testing
parser.add_argument("-end_time", help="End time of simulation (to shorten Taylor test)")
parser.add_argument("-taylor_test", help="Toggle Taylor testing")


# --- Set parameters

# Parsed arguments
args = parser.parse_args()
basis = args.basis
level = int(args.level or 0)
optimise = bool(args.rerun_optimisation or False)
gtol = float(args.gtol or 1.0e-08)
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
timeseries_type = "timeseries"
if bool(args.continuous_timeseries or False):
    timeseries_type = "_".join([timeseries_type, "smooth"])

# Do not attempt to plot in parallel
if COMM_WORLD.size > 1 and plot_any:
    print_output(120*'*' + "\nWARNING: Plotting turned off when running in parallel.\n" + 120*'*')
    plot_pdf = plot_png = False

# Setup output directories
dirname = os.path.dirname(__file__)
di = create_directory(os.path.join(dirname, basis, 'outputs', 'realistic'))
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
if basis == 'box':
    options_constructor = TohokuBoxBasisOptions
elif basis == 'radial':
    options_constructor = TohokuRadialBasisOptions
elif basis == 'okada':
    options_constructor = TohokuOkadaBasisOptions
    raise NotImplementedError  # TODO: Hook up Okada reduced functional and gradient
else:
    raise ValueError("Basis type '{:s}' not recognised.".format(basis))
op = options_constructor(**kwargs)
gauges = list(op.gauges.keys())


# --- Set initial guess

if zero_init:
    eps = 1.0e-03  # zero gives an error so just choose small
    kwargs['control_parameters'] = eps*np.ones(len(op.control_parameters))
else:
    print_output("Projecting initial guess...")

    # Create Radial parameter object
    kwargs_src = kwargs.copy()
    gaussian_scaling = float(args.gaussian_scaling or 6.0)
    kwargs_src['control_parameters'] = [gaussian_scaling, ]
    kwargs_src['nx'], kwargs_src['ny'] = 1, 1
    op_src = TohokuRadialBasisOptions(mesh=op.default_mesh, **kwargs_src)
    swp = DiscreteAdjointTsunamiProblem(op_src, nonlinear=nonlinear, print_progress=op.debug)
    swp.set_initial_condition()
    f_src = swp.fwd_solutions[0].split()[1]

    # Project into chosen basis
    swp = DiscreteAdjointTsunamiProblem(op, nonlinear=nonlinear, print_progress=op.debug)
    op.project(swp, f_src)
    kwargs['control_parameters'] = [m.dat.data[0] for m in op.control_parameters]

    # Plot
    if plot_any:
        levels = np.linspace(-0.1*gaussian_scaling, 1.1*gaussian_scaling, 51)
        ticks = np.linspace(0, gaussian_scaling, 5)

        # Project into P1 for plotting
        swp.set_initial_condition()
        f_src = project(f_src, swp.P1[0])
        f = project(swp.fwd_solutions[0].split()[1], swp.P1[0])

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
        fname = 'initial_guess_{:s}_{:d}'.format(basis, level)
        savefig(fname, fpath=plot_dir, extensions=extensions)
if plot_only:
    sys.exit(0)


# --- Optimisation

# Solve the forward problem with initial guess
op.save_timeseries = True
op = options_constructor(**kwargs)
print_output("Run forward to get timeseries...")
swp = DiscreteAdjointTsunamiProblem(op, nonlinear=nonlinear, print_progress=op.debug)
print_output("Clearing tape...")
swp.clear_tape()
print_output("Setting initial guess...")
op.assign_control_parameters(kwargs['control_parameters'])
controls = [Control(m) for m in op.control_parameters]
swp.solve_forward()
J = swp.quantity_of_interest()

# Taylor test
if bool(args.taylor_test or False):
    Jhat = ReducedFunctional(J, controls)

    def get_controls(mode):
        """
        Get control parameter values assuming one of three modes:

          'init' - control parameters are set to the initial guess;
          'random' - control parameters are set randomly with a Normal distribution;
          'optimised' - optimal control parameters from a previous run are used.
        """
        c = [Function(m) for m in op.control_parameters]
        if mode == 'init':
            pass
        elif mode == 'random':
            for ci in c:
                ci.dat.data[0] = 10.0*np.random.random()
        elif mode == 'optimised':
            fname = os.path.join(di, 'optimisation_progress_ctrl_{:d}.npy'.format(level))
            try:
                c_dat = np.load(fname)[-1]
            except IOError:
                print_output("Skipping Taylor test at optimised controls because no data found.")
                sys.exit(0)
            for i, ci in enumerate(c):
                ci.dat.data[0] = c_dat[i]
        else:
            raise ValueError("Taylor test mode '{:s}' not recognised.".format(mode))
        return c

    # Random search direction
    dc = [Function(m) for m in op.control_parameters]
    for dci in dc:
        dci.dat.data[0] = 10.0*np.random.random()

    # Run tests
    for mode in ("init", "random", "optimised"):
        print_output("Taylor test '{:s}' begin...".format(mode))
        minconv = taylor_test(Jhat, get_controls(mode), dc)
        if minconv > 1.90:
            print_output("Taylor test '{:s}' passed!".format(mode))
        else:
            msg = "Taylor test '{:s}' failed! Convergence ratio {:.2f} < 2."
            raise ConvergenceError(msg.format(mode, minconv))
    sys.exit(0)

# Save timeseries
for gauge in gauges:
    fname = os.path.join(di, '_'.join([gauge, 'data', str(level)]))
    np.save(fname, op.gauges[gauge]['data'])
    fname = os.path.join(di, '_'.join([gauge, timeseries_type, str(level)]))
    np.save(fname, op.gauges[gauge][timeseries_type])

# Run optimisation / load optimised controls
fname = os.path.join(di, 'optimisation_progress_{:s}' + '_{:d}.npy'.format(level))
if optimise:
    control_values_opt = [[m.dat.data[0] for m in op.control_parameters], ]
    func_values_opt = [J, ]
    gradient_values_opt = []

    def derivative_cb_post(j, dj, m):
        control = [mi.dat.data[0] for mi in m]
        djdm = [dji.dat.data[0] for dji in dj]
        print_output("functional {:.8e}  gradient {:.8e}".format(j, vecnorm(djdm, order=np.Inf)))
        control_values_opt.append(control)
        func_values_opt.append(j)
        gradient_values_opt.append(djdm)
        np.save(fname.format('ctrl'), np.array(control_values_opt))
        np.save(fname.format('func'), np.array(func_values_opt))
        np.save(fname.format('grad'), np.array(gradient_values_opt))

    # Run BFGS optimisation
    opt_kwargs = {'maxiter': 1000, 'gtol': gtol}
    print_output("Optimisation begin...")
    Jhat = ReducedFunctional(J, controls, derivative_cb_post=derivative_cb_post)
    optimised_value = minimize(Jhat, method='BFGS', options=opt_kwargs)
    optimised_value = [m.dat.data[0] for m in optimised_value]
else:
    optimised_value = np.load(fname.format('ctrl'))[-1]


# --- Compare timeseries

# Run forward again using the optimised control parameters
op.plot_pvd = plot_pvd
swp = DiscreteAdjointTsunamiProblem(op, nonlinear=nonlinear, print_progress=op.debug)
print_output("Clearing tape...")
get_working_tape().clear_tape()
print_output("Assigning optimised control parameters...")
op.assign_control_parameters(optimised_value)
print_output("Run to plot optimised timeseries...")
swp.solve_forward()
J = swp.quantity_of_interest()

# Save timeseries to file
for gauge in gauges:
    fname = os.path.join(di, '_'.join([gauge, timeseries_type, str(level)]))
    np.save(fname, op.gauges[gauge][timeseries_type])

# Solve adjoint problem and plot solution fields
if plot_pvd:
    swp.solve_adjoint()
    swp.save_adjoint_trajectory()
