from thetis import *

import argparse
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import os

from adapt_utils.plotting import *
from adapt_utils.unsteady.test_cases.turbine_array.options import TurbineArrayOptions
from adapt_utils.unsteady.swe.turbine.solver import AdaptiveTurbineProblem


# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("Re", help="Target Reynolds number.")
parser.add_argument("-min_viscosity", help="Minimum tolerated viscosity (default 0).")
args = parser.parse_args()

Re = float(args.Re)
kwargs = {
    'target_mesh_reynolds_number': Re,
    'min_viscosity': float(args.min_viscosity or 0.0),
    'spun': True,
    'debug': True,
}

# Setup problem
op = TurbineArrayOptions(**kwargs)
swp = AdaptiveTurbineProblem(op, ramp_dir='data/ramp')
mesh = swp.meshes[0]
swp.set_initial_condition()
u, eta = swp.fwd_solutions[0].split()

# Enforce maximum mesh Reynolds number and plot
nu = swp.fields[0].horizontal_viscosity
nu_min = nu.vector().gather().min()
nu_max = nu.vector().gather().max()
fig, axes = plt.subplots(figsize=(12, 6))
levels = np.linspace(-0.01, 1.01, 50)
levels = nu_min**np.flip(levels)*nu_max**levels
tc = tricontourf(
    nu, axes=axes, levels=levels, cmap='coolwarm',
    norm=colors.LogNorm(vmin=nu_min, vmax=nu_max),
)
cbar = fig.colorbar(tc, ax=axes, orientation="horizontal", pad=0.2)
cbar.set_label(r"(Kinematic) viscosity [$\mathrm m^2\,\mathrm s^{-1}$]")
cbar.set_ticks([1.0e-03, 1.0e-02, 1.0e-01, 1])
plot_dir = create_directory(os.path.join(os.path.dirname(__file__), 'plots'))
fname = "viscosity_Re{:.1f}_min_viscosity{:.1e}".format(Re, op.min_viscosity)
savefig(fname, plot_dir, extensions=['png'])

# Plot mesh Reynolds number
fig, axes = plt.subplots(figsize=(12, 6))
tc = tricontourf(swp.shallow_water_options[0].sipg_parameter, axes=axes, levels=50, cmap='coolwarm')
cbar = fig.colorbar(tc, ax=axes)
cbar.set_label("SIPG parameter")
plt.tight_layout()

# Plot mesh Reynolds number
fig, axes = plt.subplots(figsize=(12, 6))
tc = swp.plot_mesh_reynolds_number(0, axes=axes)
cbar = fig.colorbar(tc, ax=axes)
cbar.set_label("Mesh Reynolds number")
plt.tight_layout()
plt.show()