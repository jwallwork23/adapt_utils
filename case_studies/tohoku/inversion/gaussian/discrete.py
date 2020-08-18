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

        # Create GaussianBasis parameter class and establish the basis functions
        kwargs['nx'], kwargs['ny'] = nx, ny
        kwargs['control_parameters'] = eps*np.ones(nx*ny)
        op = TohokuGaussianBasisOptions(mesh=op_okada.default_mesh, **kwargs)
        op.di = create_directory(os.path.join(di, 'discrete'))
        swp = AdaptiveProblem(op, nonlinear=nonlinear, print_progress=False)
        op.get_basis_functions(swp.V[0])

        # Get array coordinates before rotation and rescale down so that dx = dy = 1
        x0, y0 = op.centre_x, op.centre_y
        x = np.linspace(-0.5*op.extent_x, 0.5*op.extent_x, op.nx)/op.radius_x
        y = np.linspace(-0.5*op.extent_y, 0.5*op.extent_y, op.ny)/op.radius_y
        X, Y = np.meshgrid(x, y)
        eps = 1.0

        # Rotate mesh coordinates  # TODO: Modify rotation_matrix to avoid for loop
        R = rotation_matrix(op.strike_angle)  # NOTE: Opposite direction
        coords_x, coords_y = [], []
        for xy in op.default_mesh.coordinates.dat.data:
            x_rot, y_rot = tuple(np.dot(R, np.array([xy[0] - x0, xy[1] - y0])))
            coords_x.append(x_rot)
            coords_y.append(y_rot)
        coords = [np.array(coords_x)/op.radius_x, np.array(coords_y)/op.radius_y]

        # Interpolate with radial basis functions
        rbfi = scipy.interpolate.Rbf(*coords, f_okada.dat.data, function='gaussian', epsilon=eps)
        eps = 1.0e-03
        for i, (x, y) in enumerate(zip(X.flatten(), Y.flatten())):
            coeff = float(rbfi(x, y))
            if np.abs(coeff) < eps:
                coeff = eps
            print(i, x, y, coeff)
            op.control_parameters[i].assign(coeff)
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

raise NotImplementedError  # TODO: Optimisation

# TODO: (synthetic case)
#  * Interpolate "optimum" from Okada case using
#    - https://scipy-cookbook.readthedocs.io/items/RadialBasisFunctions.html;
#    - https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.Rbf.html.
#  * May need to transform to a non-rotated space where the basis functions are square rather than rectangular
