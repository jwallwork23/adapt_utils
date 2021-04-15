from thetis import *

import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

from adapt_utils.case_studies.tohoku.options.options import TohokuOptions
from adapt_utils.case_studies.tohoku.options.radial_options import TohokuRadialBasisOptions
from adapt_utils.plotting import *
from adapt_utils.norms import timeseries_error
from adapt_utils.swe.tsunami.conversion import lonlat_to_utm
from adapt_utils.unsteady.solver import AdaptiveProblem


# --- Parse arguments

parser = argparse.ArgumentParser()

# Model
parser.add_argument("-levels", help="Number of mesh resolution levels")
parser.add_argument("-family", help="Finite element pair")
parser.add_argument("-stabilisation", help="Stabilisation approach")
parser.add_argument("-nonlinear", help="Toggle nonlinear model")

# Norms
parser.add_argument("-norm_type", help="Norm type for timeseries error")

# Plotting and debugging
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
levels = int(args.levels or 3)
plot_pvd = bool(args.plot_pvd or False)
plot_pdf = bool(args.plot_pdf or False)
plot_png = bool(args.plot_png or False)
plot_all = bool(args.plot_all or False)
plot_only = bool(args.plot_only or False)
if plot_only:
    plot_all = True
if plot_all:
    plot_pvd = plot_pdf = plot_png = True

# Plotting parameters
fontsize = 22
fontsize_tick = 18
plotting_kwargs = {'markevery': 5}
norm_type = args.norm_type or 'l2'
assert norm_type in ('l2', 'tv')

# Do not attempt to plot in parallel
if COMM_WORLD.size > 1 and (plot_pdf or plot_png):
    print_output(120*'*' + "\nWARNING: Plotting turned off when running in parallel.\n" + 120*'*')
    plot_pdf = plot_png = False


def savefig(filename):
    """To avoid duplication."""
    if plot_pdf:
        plt.savefig(filename + '.pdf')
    if plot_png:
        plt.savefig(filename + '.png')


# Collect initialisation parameters
kwargs = {
    'save_timeseries': True,

    # Spatial discretisation
    'family': args.family or 'dg-cg',
    'stabilisation': args.stabilisation,
    'use_automatic_sipg_parameter': False,  # the problem is inviscid

    # I/O and debugging
    'plot_pvd': False,
    'debug': bool(args.debug or False),
    'debug_mode': args.debug_mode or 'basic',
}
nonlinear = bool(args.nonlinear or False)

# Setup output directories
dirname = os.path.dirname(__file__)
di = create_directory(os.path.join(dirname, 'outputs', 'test_project'))
plot_dir = create_directory(os.path.join(di, 'plots'))


# --- Loop over mesh hierarchy

errors = {'timeseries': {}, 'timeseries_smooth': {}}
num_cells = []
for level in range(levels):
    kwargs['level'] = level

    # Create Options parameter object
    op = TohokuRadialBasisOptions(**kwargs)
    op.di = di
    op.plot_pvd = plot_pvd

    # Bookkeeping for storing total variation
    num_cells.append(op.default_mesh.num_cells())
    gauges = list(op.gauges.keys())
    if errors['timeseries'] == {}:
        errors['timeseries'] = {gauge: [] for gauge in gauges}
        errors['timeseries_smooth'] = {gauge: [] for gauge in gauges}
    N = int(np.ceil(np.sqrt(len(gauges))))

    # Interpolate initial condition from [Saito et al. 2011] into a P1 space on the current mesh
    op_saito = TohokuOptions(mesh=op.default_mesh, **kwargs)
    swp_saito = AdaptiveProblem(op_saito, nonlinear=nonlinear, print_progress=op.debug)
    ic_saito = op_saito.set_initial_condition(swp_saito)

    # Project Saito's initial condition into the radial basis
    swp_radial = AdaptiveProblem(op, nonlinear=nonlinear, print_progress=op.debug)
    op.project(swp_radial, ic_saito)
    op.set_initial_condition(swp_radial)
    ic_radial = project(swp_radial.fwd_solutions[0].split()[1], swp_radial.P1[0])

    # Load or save timeseries, as appropriate
    if plot_only:
        for gauge in gauges:
            for options, name in zip((op_saito, op), ('original', 'projected')):
                for tt in errors:
                    fname = os.path.join(di, '_'.join([gauge, name, str(level) + '.npy']))
                    options.gauges[gauge][tt] = np.load(fname)
    else:
        for swp in (swp_saito, swp_radial):
            print_output("Solving forward on {:s}...".format(swp.__class__.__name__))
            swp.setup_solver_forward_step(0)
            swp.solve_forward_step(0)
        for gauge in gauges:
            for options, name in zip((op_saito, op), ('original', 'projected')):
                for tt in ('timeseries', 'timeseries_smooth'):
                    fname = os.path.join(di, '_'.join([gauge, name, str(level)]))
                    np.save(fname, options.gauges[gauge][tt])

    # Compare timeseries error
    for tt, cd in zip(errors.keys(), ('Continuous', 'Discrete')):
        print_output("\n{:s} form QoI:".format(cd))
        for gauge in op.gauges:
            orig = op_saito.gauges[gauge][tt]
            error = timeseries_error(orig, op.gauges[gauge][tt], relative=True, norm_type=norm_type)
            errors[tt][gauge].append(error)
            print_output("Relative {:s} error for gauge {:s}: {:.4e}".format(norm_type, gauge, error))

    # Skip if not plotting
    if not (plot_pdf or plot_png):
        continue

    # Get corners of zoom
    lonlat_corners = [(138, 32), (148, 42), (138, 42)]
    utm_corners = [lonlat_to_utm(*corner, 54) for corner in lonlat_corners]
    xlim = [utm_corners[0][0], utm_corners[1][0]]
    ylim = [utm_corners[0][1], utm_corners[2][1]]

    # Plot
    fig, axes = plt.subplots(ncols=2, figsize=(9, 4))
    levels = np.linspace(-3, 8, 51)
    ticks = np.linspace(-2.5, 7.5, 9)
    for ic, ax in zip((ic_saito, ic_radial), (axes[0], axes[1])):
        cbar = fig.colorbar(tricontourf(ic, axes=ax, levels=levels, cmap='coolwarm'), ax=ax)
        cbar.set_ticks(ticks)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.axis(False)
    axes[0].set_title("P1 basis")
    axes[1].set_title("Piecewise constant basis")
    savefig(os.path.join(plot_dir, 'ic_{:d}'.format(level)))

    # Plot timeseries
    for tt in errors:
        fig, axes = plt.subplots(nrows=N, ncols=N, figsize=(14, 12))
        for i, gauge in enumerate(gauges):
            T = np.array(op.gauges[gauge]['times'])/60
            ax = axes[i//N, i % N]
            ax.plot(T, op_saito.gauges[gauge][tt], '--x', label=gauge+' original', **plotting_kwargs)
            ax.plot(T, op.gauges[gauge][tt], '--x', label=gauge+' projected', **plotting_kwargs)
            ax.legend(loc='upper right')
            ax.set_xlabel('Time (min)', fontsize=fontsize)
            ax.set_ylabel('Elevation (m)', fontsize=fontsize)
            plt.xticks(fontsize=fontsize_tick)
            plt.yticks(fontsize=fontsize_tick)
            ax.grid()
        for i in range(len(gauges), N*N):
            axes[i//N, i % N].axis(False)
        plt.tight_layout()
        savefig(os.path.join(plot_dir, 'timeseries_{:d}'.format(level)))
if not (plot_pdf or plot_png):
    sys.exit(0)

# Plot timeseries errors over mesh iterations
for tt in errors:
    fig, axes = plt.subplots(nrows=N, ncols=N, figsize=(14, 12))
    label = r"$\ell_2$-error" if norm_type == 'l2' else 'total variation'
    for i, gauge in enumerate(gauges):
        ax = axes[i//N, i % N]
        ax.plot(num_cells, errors[tt][gauge], '--x', label=gauge, **plotting_kwargs)
        ax.legend(loc='upper right')
        ax.set_xlabel('Mesh elements', fontsize=fontsize)
        ax.set_ylabel('Timeseries {:s}'.format(label), fontsize=fontsize)
        plt.xticks(fontsize=fontsize_tick)
        plt.yticks(fontsize=fontsize_tick)
        ax.grid()
    for i in range(len(gauges), N*N):
        axes[i//N, i % N].axis(False)
    plt.tight_layout()
    savefig(os.path.join(plot_dir, '_'.join([tt, norm_type, 'error'])))
