from thetis import *

import argparse
import matplotlib.pyplot as plt
import numpy as np
import os

from adapt_utils.io import load_mesh
from adapt_utils.mesh import *
from adapt_utils.plotting import *
from adapt_utils.steady.test_cases.point_discharge2d.options import PointDischarge2dOptions
from adapt_utils.swe.utils import speed


# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('approach', help="Mesh adaptation approach")
parser.add_argument('family', help="Finite element family")
parser.add_argument('-stabilisation', help="Stabilisation method to use")
parser.add_argument('-anisotropic_stabilisation', help="Use anisotropic cell size measure?")
parser.add_argument('-debug', help="Toggle debugging mode.")
args = parser.parse_args()

# Set parameters
assert args.family in ('cg', 'dg')
kwargs = {
    'approach': args.approach or 'dwr',
    'plot_pvd': False,
    'debug': bool(args.debug or False),
}
op = PointDischarge2dOptions(**kwargs)
op.tracer_family = args.family
stabilisation = args.stabilisation or 'supg'
op.stabilisation_tracer = None if stabilisation == 'none' else stabilisation
op.anisotropic_stabilisation = False if args.anisotropic_stabilisation == '0' else True
op.di = os.path.join(op.di, op.stabilisation_tracer or args.family)

# Load from HDF5
op.plot_pvd = False
mesh = load_mesh("mesh", fpath=op.di)
h = anisotropic_cell_size(mesh) if op.anisotropic_stabilisation else isotropic_cell_size(mesh)

# Plot
if args.family == 'cg':

    # Plot cell size measure
    fig, axes = plt.subplots(figsize=(8, 3))
    hmin = np.floor(h.vector().gather().min())
    hmax = np.ceil(h.vector().gather().max())
    levels = np.linspace(hmin, hmax, 40)
    tc = tricontourf(h, axes=axes, levels=levels, cmap='coolwarm')
    cbar = fig.colorbar(tc, ax=axes, orientation="horizontal", pad=0.1)
    cbar.set_ticks(np.linspace(hmin, hmax, 5))
    axes.axis(False)
    axes.set_xlim([0, 50])
    axes.set_ylim([0, 10])
    fname = 'anisotropic_cell_size' if op.anisotropic_stabilisation else 'isotropic_cell_size'
    savefig(fname, op.di, extensions=["jpg"])

    # Calculate stabilisation parameter
    uv = Constant(as_vector(op.base_velocity))
    unorm = speed(uv)
    tau = 0.5*h/unorm
    Pe = 0.5*h*unorm/op.base_diffusivity
    tau *= min_value(1, Pe/3)
    tau = interpolate(tau, h.function_space())

    # Plot stabilisation parameter
    fig, axes = plt.subplots(figsize=(8, 3))
    taumin = np.floor(tau.vector().gather().min())
    taumax = np.ceil(tau.vector().gather().max())
    levels = np.linspace(taumin, taumax, 40)
    tc = tricontourf(tau, axes=axes, levels=levels, cmap='coolwarm')
    cbar = fig.colorbar(tc, ax=axes, orientation="horizontal", pad=0.1)
    cbar.set_ticks(np.linspace(taumin, taumax, 5))
    axes.axis(False)
    axes.set_xlim([0, 50])
    axes.set_ylim([0, 10])
    fname = 'anisotropic_stabilisation' if op.anisotropic_stabilisation else 'isotropic_stabilisation'
    savefig(fname, op.di, extensions=["jpg"])
else:

    # Calculate SIPG parameter
    min_angles = get_minimum_angles_2d(mesh)
    cot_theta = 1.0/tan(min_angles)
    p = op.degree_tracer
    alpha = Constant(5.0*p*(p+1) if p != 0 else 1.5)
    alpha = alpha*get_sipg_ratio(op.base_diffusivity)*cot_theta
    # alpha = interpolate(alpha/h, min_angles.function_space())
    alpha = interpolate(h/alpha, min_angles.function_space())

    # Plot stabilisation parameter
    fig, axes = plt.subplots(figsize=(8, 3))
    alphamin = np.floor(alpha.vector().gather().min())
    alphamax = np.ceil(alpha.vector().gather().max())
    levels = np.linspace(alphamin, alphamax, 40)
    tc = tricontourf(alpha, axes=axes, levels=levels, cmap='coolwarm')
    cbar = fig.colorbar(tc, ax=axes, orientation="horizontal", pad=0.1)
    cbar.set_ticks(np.linspace(alphamin, alphamax, 5))
    axes.axis(False)
    axes.set_xlim([0, 50])
    axes.set_ylim([0, 10])
    fname = 'anisotropic_sipg' if op.anisotropic_stabilisation else 'isotropic_sipg'
    savefig(fname, op.di, extensions=["jpg"])
