from thetis import *

import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

from adapt_utils.case_studies.tohoku.options.options import TohokuOptions
from adapt_utils.case_studies.tohoku.options.box_options import TohokuBoxBasisOptions
from adapt_utils.plotting import *
from adapt_utils.norms import total_variation
from adapt_utils.unsteady.solver import AdaptiveProblem
from adapt_utils.unsteady.swe.tsunami.conversion import lonlat_to_utm


# --- Parse arguments

parser = argparse.ArgumentParser()

# Model
parser.add_argument("-level", help="Mesh resolution level")
parser.add_argument("-family", help="Finite element pair")
parser.add_argument("-stabilisation", help="Stabilisation approach")
parser.add_argument("-nonlinear", help="Toggle nonlinear model")

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
plot_pvd = bool(args.plot_pvd or False)
plot_pdf = bool(args.plot_pdf or False)
plot_png = bool(args.plot_png or False)
plot_all = bool(args.plot_all or False)
plot_only = bool(args.plot_only or False)
if plot_only:
    plot_all = True
if plot_all:
    plot_pvd = plot_pdf = plot_png = True

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
    'level': level,
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
op = TohokuBoxBasisOptions(**kwargs)
op.plot_pvd = plot_pvd

# Plotting parameters
if plot_pdf or plot_png:
    fontsize = 22
    fontsize_tick = 18
    plotting_kwargs = {'markevery': 5}

# Setup output directories
dirname = os.path.dirname(__file__)
op.di = create_directory(os.path.join(dirname, 'outputs', 'test_project'))
plot_dir = create_directory(os.path.join(op.di, 'plots'))


# --- Consider different initial conditions

# Interpolate initial condition from [Saito et al. 2011] into a P1 space on the current mesh and run
op_saito = TohokuOptions(mesh=op.default_mesh, **kwargs)
swp_saito = AdaptiveProblem(op_saito, nonlinear=nonlinear, print_progress=False)
ic_saito = op_saito.set_initial_condition(swp_saito)

# Project Saito's initial condition into the box basis
swp_box = AdaptiveProblem(op, nonlinear=nonlinear, print_progress=False)
op.project(swp_box, ic_saito)
op.set_initial_condition(swp_box)

# Load or save timeseries, as appropriate
gauges = list(op.gauges.keys())
if plot_only:
    for gauge in gauges:
        for options, name in zip((op_saito, op), ('original', 'projected')):
            for tt in ('timeseries', 'timeseries_smooth'):
                fname = os.path.join(op.di, '_'.join([gauge, name, str(level) + '.npy']))
                options.gauges[gauge][tt] = np.load(fname)
else:
    for swp in (swp_saito, swp_box):
        print_output("Solving forward on {:s}...".format(swp.__class__.__name__))
        swp.setup_solver_forward(0)
        swp.solve_forward_step(0)
        print_output("Done!")
    for gauge in gauges:
        for options, name in zip((op_saito, op), ('original', 'projected')):
            for tt in ('timeseries', 'timeseries_smooth'):
                fname = os.path.join(op.di, '_'.join([gauge, name, str(level)]))
                np.save(fname, options.gauges[gauge][tt])

# Compare total variation
for tt, cd in zip(('timeseries', 'timeseries_smooth'), ('Continuous', 'Discrete')):
    print_output("\n{:s} form QoI:".format(cd))
    for gauge in op.gauges:
        tv = total_variation(np.array(op.gauges[gauge][tt]) - np.array(op_saito.gauges[gauge][tt]))
        print_output("total variation for gauge {:s}: {:.4e}".format(gauge, tv))

# Exit if not plotting
if not (plot_pdf or plot_png):
    sys.exit(0)

# Get corners of zoom
lonlat_corners = [(138, 32), (148, 42), (138, 42)]
utm_corners = [lonlat_to_utm(*corner, 54) for corner in lonlat_corners]
xlim = [utm_corners[0][0], utm_corners[1][0]]
ylim = [utm_corners[0][1], utm_corners[2][1]]

# Plot
fig, axes = plt.subplots(ncols=2, figsize=(9, 4))
ic_box = project(swp_box.fwd_solutions[0].split()[1], swp_box.P1[0])
# levels = np.linspace(-6, 16, 51)
levels = 51
# ticks = np.linspace(-5, 15, 9)
for ic, ax in zip((ic_saito, ic_box), (axes[0], axes[1])):
    cbar = fig.colorbar(tricontourf(ic, axes=ax, levels=levels, cmap='coolwarm'), ax=ax)
    # cbar.set_ticks(ticks)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.axis(False)
axes[0].set_title("P1 basis")
axes[1].set_title("Piecewise constant basis")
savefig(os.path.join(plot_dir, 'ic_{:d}'.format(level)))

# Plot timeseries
N = int(np.ceil(np.sqrt(len(gauges))))
for tt in ('timeseries', 'timeseries_smooth'):
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
