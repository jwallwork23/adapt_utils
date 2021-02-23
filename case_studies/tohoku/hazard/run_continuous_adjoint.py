from thetis import *

import argparse
import os

from adapt_utils.case_studies.tohoku.hazard.options import TohokuHazardOptions
from adapt_utils.swe.tsunami.solver import AdaptiveTsunamiProblem


# --- Parse arguments

parser = argparse.ArgumentParser(prog="run_continuous_adjoint")

# Space-time domain
parser.add_argument("-end_time", help="End time of simulation in seconds (default 1440s, i.e. 24min)")
parser.add_argument("-level", help="(Integer) resolution for initial mesh (default 0)")
parser.add_argument("-num_meshes", help="Number of meshes to consider (for testing, default 1)")
parser.add_argument("-levels", help="Number of iso-P2 refinements (for testing, default 0)")

# Solver
parser.add_argument("-family", help="Element family for mixed FE space (default 'cg-cg')")
parser.add_argument("-nonlinear", help="Toggle nonlinear equations (default False)")
parser.add_argument("-stabilisation", help="Stabilisation method to use (default None)")

# QoI
parser.add_argument("-start_time", help="""
    Start time of period of interest in seconds (default 1200s, i.e. 20min)""")
parser.add_argument("-locations", help="""
    Locations of interest, separated by commas. Choose from {'Fukushima Daiichi', 'Onagawa',
    'Fukushima Daini', 'Tokai', 'Hamaoka', 'Tohoku', 'Tokyo'}. (Default 'Fukushima Daiichi')""")
parser.add_argument("-radius", help="Radius of interest (default 100km)")

# I/O and debugging
parser.add_argument("-plot_pvd", help="Toggle saving output to .pvd")
parser.add_argument("-debug", help="Print all debugging statements")
parser.add_argument("-debug_mode", help="Choose debugging mode from 'basic' and 'full'")

args = parser.parse_args()


# --- Set parameters

locations = ['Fukushima Daiichi'] if args.locations is None else args.locations.split(',')
radius = float(args.radius or 100.0e+03)
plot_pvd = False if args.plot_pvd == '0' else True
kwargs = {
    'approach': 'fixed_mesh',

    # Space-time domain
    'level': int(args.level or 0),
    'num_meshes': int(args.num_meshes or 1),
    'end_time': float(args.end_time or 1440.0),

    # Physics
    'bathymetry_cap': 30.0,

    # Solver
    'family': args.family or 'cg-cg',
    'stabilisation': args.stabilisation,
    'use_wetting_and_drying': False,
    'wetting_and_drying_alpha': Constant(10.0),

    # QoI
    'start_time': float(args.start_time or 1200.0),
    'radius': radius,
    'locations': locations,

    # I/O and debugging
    'plot_pvd': plot_pvd,
    'debug': bool(args.debug or False),
    'debug_mode': args.debug_mode or 'basic',
}
nonlinear = bool(args.nonlinear or False)
op = TohokuHazardOptions(**kwargs)


# --- Initialisation

# Wrap mesh(es) in an iso-P2 refined mesh hiearchy and choose the finest level(s)
levels = int(args.levels or 0)
base_meshes = [op.default_mesh for i in range(op.num_meshes)]
hierarchies = [MeshHierarchy(mesh, levels) for mesh in base_meshes]
meshes = [hierarchy[levels] for hierarchy in hierarchies]


# --- Solve continuous adjoint

swp = AdaptiveTsunamiProblem(op, meshes=meshes, nonlinear=nonlinear)
if plot_pvd:
    kernel_file = File(os.path.join(op.di, 'kernel.pvd'))
    for i, P1 in enumerate(swp.P1):
        swp.get_qoi_kernels(i)
        k_u, k_eta = swp.kernels[i].split()
        kernel = Function(P1, name="QoI kernel")
        kernel.project(k_eta)
        kernel_file._topology = None
        kernel_file.write(kernel)
swp.solve_adjoint()
