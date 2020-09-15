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
from adapt_utils.norms import total_variation, vecnorm
from adapt_utils.plotting import *
from adapt_utils.unsteady.solver import AdaptiveProblem
from adapt_utils.unsteady.solver_adjoint import AdaptiveDiscreteAdjointProblem
from adapt_utils.unsteady.swe.tsunami.conversion import lonlat_to_utm


class DiscreteAdjointTsunamiProblem(AdaptiveDiscreteAdjointProblem):
    """The subclass exists to pass the QoI as required."""
    def quantity_of_interest(self):
        return self.op.J


# --- Parse arguments

parser = ArgumentParser(basis=True, plotting=True, shallow_water=True)

# Resolution
parser.add_argument("-level", help="Mesh resolution level")
# parser.add_argument("-okada_grid_resolution", help="Mesh resolution level for the Okada grid")

# Inversion
parser.add_argument("-rerun_optimisation", help="Rerun optimisation routine")
parser.add_argument("-noisy_data", help="Toggle whether to sample noisy data")
parser.add_argument("-continuous_timeseries", help="Toggle discrete or continuous timeseries")
parser.add_argument("-gtol", help="Gradient tolerance (default 1.0e-08)")
parser.add_argument("-zero_initial_guess", help="""
    Toggle between a zero initial guess and a static interpretation of the dynamic source generated
    in [Shao et al. 2012].
    """)
parser.add_argument("-gaussian_scaling", help="Scaling for Gaussian initial guess (default 6.0)")


# --- Set parameters

# Parsed arguments
args = parser.parse_args()
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
}

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

# Setup output directories
dirname = os.path.dirname(__file__)
di = create_directory(os.path.join(dirname, basis, 'outputs', 'realistic'))
op.di = create_directory(os.path.join(di, 'discrete'))
plot_dir = create_directory(os.path.join(di, 'plots'))
create_directory(os.path.join(plot_dir, 'discrete'))


# --- Set initial guess

if zero_init:
    print_output("Setting (near) zero initial guess...")
    eps = 1.0e-03  # zero gives an error so just choose small
    for control in op.control_parameters:
        control.assign(eps)
else:
    with stop_annotating():
        print_output("Projecting initial guess...")

        # Create Okada parameter object
        # kwargs_src = kwargs.copy()
        # kwargs_src['okada_grid_resolution'] = int(args.okada_grid_resolution or 51)
        # op_src = TohokuOkadaBasisOptions(mesh=op.default_mesh, **kwargs_src)
        # swp = AdaptiveProblem(op_src, nonlinear=nonlinear, print_progress=op.debug)
        # f_src = op_src.set_initial_condition(swp)

        # Create Radial parameter object
        kwargs_src = kwargs.copy()
        gaussian_scaling = float(args.gaussian_scaling or 6.0)
        kwargs_src['control_parameters'] = [gaussian_scaling, ]
        kwargs_src['nx'], kwargs_src['ny'] = 1, 1
        op_src = TohokuRadialBasisOptions(mesh=op.default_mesh, **kwargs_src)
        swp = AdaptiveProblem(op_src, nonlinear=nonlinear, print_progress=op.debug)
        swp.set_initial_condition()
        f_src = swp.fwd_solutions[0].split()[1]

        # Project into chosen basis
        swp = AdaptiveProblem(op, nonlinear=nonlinear, print_progress=op.debug)
        op.project(swp, f_src)
        # op.interpolate(swp, f_src)

        # Plot
        if plot_any:
            # levels = np.linspace(-6, 16, 51)
            # ticks = np.linspace(-5, 15, 9)
            levels = np.linspace(-0.1*gaussian_scaling, 1.1*gaussian_scaling, 51)
            ticks = np.linspace(0, gaussian_scaling, 5)

            # Project into P1 for plotting
            swp.set_initial_condition()
            f_src = project(f_src, swp.P1[0])
            f = project(swp.fwd_solutions[0].split()[1], swp.P1[0])

            # Get corners of zoom
            lonlat_corners = [(138, 32), (148, 42), (138, 42)]
            utm_corners = [lonlat_to_utm(*corner, 54) for corner in lonlat_corners]
            xlim = [utm_corners[0][0], utm_corners[1][0]]
            ylim = [utm_corners[0][1], utm_corners[2][1]]

            # Plot initial guess
            fig, axes = plt.subplots(figsize=(4.5, 4))
            cbar = fig.colorbar(tricontourf(f, axes=axes, levels=levels, cmap='coolwarm'), ax=axes)
            cbar.set_ticks(ticks)
            axes.set_xlim(xlim)
            axes.set_ylim(ylim)
            axes.axis(False)
            fname = 'initial_guess_{:s}_{:d}'.format(basis, level)
            savefig(fname, fpath=plot_dir, extensions=extensions)
if plot_only:
    sys.exit(0)


# --- Optimisation

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

# Run optimisation / load optimised controls
fname = os.path.join(op.di, 'optimisation_progress_{:s}' + '_{:d}.npy'.format(level))
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
    controls = [Control(c) for c in op.control_parameters]
    Jhat = ReducedFunctional(J, controls, derivative_cb_post=derivative_cb_post)
    optimised_value = [o.dat.data[0] for o in minimize(Jhat, method='BFGS', options=opt_kwargs)]
else:
    optimised_value = np.load(fname.format('ctrl'))[-1]


# --- Compare timeseries

# Create a new parameter class
kwargs['control_parameters'] = optimised_value
kwargs['plot_pvd'] = plot_pvd
op_opt = options_constructor(**kwargs)
for gauge in gauges:
    op_opt.gauges[gauge]["data"] = op.gauges[gauge]["data"]

# Run forward again and save timeseries
print_output("Run to plot optimised timeseries...")
get_working_tape().clear_tape()
swp = DiscreteAdjointTsunamiProblem(op_opt, nonlinear=nonlinear, print_progress=op.debug)
swp.solve_forward()
J = swp.quantity_of_interest()
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
    swp.solve_adjoint()
    swp.save_adjoint_trajectory()
