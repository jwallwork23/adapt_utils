from thetis import *

import argparse
import matplotlib.pyplot as plt
import numpy as np
import os

from adapt_utils.case_studies.tohoku.hazard.options import TohokuHazardOptions
from adapt_utils.plotting import *
from adapt_utils.swe.tsunami.solver import AdaptiveTsunamiProblem


# --- Parse arguments

parser = argparse.ArgumentParser(prog="run_fixed_mesh")

# Space-time domain
parser.add_argument("-end_time", help="End time of simulation in seconds (default 1440s, i.e. 24min)")
parser.add_argument("-level", help="(Integer) mesh resolution (default 0)")
parser.add_argument("-num_meshes", help="Number of meshes to consider (for testing, default 1)")

# Solver
parser.add_argument("-family", help="Element family for mixed FE space (default 'cg-cg')")
parser.add_argument("-nonlinear", help="Toggle nonlinear equations (default False)")
parser.add_argument("-stabilisation", help="Stabilisation method to use (default None)")

# QoI
parser.add_argument("-start_time", help="""
    Start time of period of interest in seconds (default 1220s, i.e. 20min).
    """)
parser.add_argument("-locations", help="""
    Locations of interest, separated by commas. Choose from {'Fukushima Daiichi', 'Onagawa',
    'Fukushima Daini', 'Tokai', 'Hamaoka', 'Tohoku', 'Tokyo'}. (Default 'Fukushima Daiichi')
    """)
parser.add_argument("-radius", help="Radius of interest (default 100km)")
parser.add_argument("-kernel_shape", help="""
    Choose kernel shape from {'gaussian', 'circular_bump', 'ball'}.
    """)

# I/O and debugging
parser.add_argument("-plot_pdf", help="Toggle plotting to .pdf")
parser.add_argument("-plot_png", help="Toggle plotting to .png")
parser.add_argument("-plot_pvd", help="Toggle plotting to .pvd")
parser.add_argument("-plot_all", help="Toggle plotting to .pdf, .png and .pvd")
parser.add_argument("-plot_only", help="Just plot using saved data")
parser.add_argument("-debug", help="Print all debugging statements")
parser.add_argument("-debug_mode", help="Choose debugging mode from 'basic' and 'full'")

args = parser.parse_args()


# --- Set parameters

plot_pdf = bool(args.plot_pdf or False)
plot_png = bool(args.plot_png or False)
plot_pvd = False if args.plot_pvd == '0' else True
plot_all = bool(args.plot_all or False)
plot_only = bool(args.plot_only or False)
if plot_only:
    plot_all = True
if plot_all:
    plot_pvd = plot_pdf = plot_png = True
plot_any = plot_pdf or plot_png
extensions = []
if plot_pdf:
    extensions.append('pdf')
if plot_png:
    extensions.append('png')
if plot_only:
    assert len(extensions) > 0
if args.locations is None:  # TODO: Parse as list
    locations = ['Fukushima Daiichi']
else:
    locations = args.locations.split(',')
radius = float(args.radius or 100.0e+03)
family = args.family or 'cg-cg'
nonlinear = bool(args.nonlinear or False)
stabilisation = args.stabilisation or 'lax_friedrichs'
if stabilisation == 'none' or family == 'cg-cg' or not nonlinear:
    stabilisation = None
kwargs = {
    'approach': 'fixed_mesh',

    # Space-time domain
    'level': int(args.level or 0),
    'num_meshes': int(args.num_meshes or 1),
    'end_time': float(args.end_time or 1440.0),

    # Physics
    'bathymetry_cap': 30.0,

    # Solver
    'family': family,
    'stabilisation': stabilisation,
    'use_wetting_and_drying': False,

    # QoI
    'start_time': float(args.start_time or 1200.0),
    'radius': radius,
    'locations': locations,
    'kernel_shape': args.kernel_shape or 'gaussian',

    # I/O and debugging
    'plot_pvd': plot_pvd,
    'debug': bool(args.debug or False),
    'debug_mode': args.debug_mode or 'basic',
}
op = TohokuHazardOptions(**kwargs)
data_dir = create_directory(os.path.join(op.di, 'data'))
plot_dir = create_directory(os.path.join(op.di, 'plots'))
op.di = create_directory(os.path.join(op.di, op.kernel_shape))


# --- Solve

swp = AdaptiveTsunamiProblem(op, nonlinear=nonlinear)
if plot_pvd:
    kernel_file = File(os.path.join(op.di, 'kernel.pvd'))
    for i, P1 in enumerate(swp.P1):
        swp.get_qoi_kernels(i)
        k_u, k_eta = swp.kernels[i].split()
        kernel = Function(P1, name="QoI kernel")
        kernel.project(k_eta)
        kernel_file._topology = None
        kernel_file.write(kernel)
        msg = "level {:d}  mesh {:d}  kernel volume {:.8e}"
        print_output(msg.format(op.level, i, assemble(k_eta*dx)))
    bathymetry_file = File(os.path.join(op.di, 'bathymetry.pvd'))
    for i, P1 in enumerate(swp.P1):
        b = Function(P1, name="Bathymetry [m]")
        b.project(swp.bathymetry[i])
        bathymetry_file._topology = None
        bathymetry_file.write(b)

# Load or generate QoI timeseries
fname = 'qoi_timeseries'
if plot_only:
    qoi_timeseries = np.load(os.path.join(data_dir, fname + '.npy'))
else:
    swp.solve_forward()
    print_output("Quantity of interest: {:.8e}".format(swp.quantity_of_interest()))
    qoi_timeseries = np.array(swp.qoi_timeseries)
    np.save(os.path.join(data_dir, fname), qoi_timeseries)


# --- Plotting

# Timeseries of QoI integrand
if plot_any:
    fig, axes = plt.subplots(figsize=(6, 5))
    time_seconds = np.linspace(op.start_time, op.end_time, len(qoi_timeseries))
    time_minutes = time_seconds/60
    axes.plot(time_minutes, qoi_timeseries, '--x')
    axes.set_xlabel(r"Time [$\mathrm{min}$]")
    axes.set_ylabel(r"Quantity of interest [$m^3$]")
    for ext in extensions:
        plt.savefig(os.path.join(plot_dir, fname + ext))
