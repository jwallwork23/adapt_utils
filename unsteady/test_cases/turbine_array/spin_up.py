from thetis import *

import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
from time import perf_counter

from adapt_utils.plotting import *  # NOQA
from adapt_utils.unsteady.swe.turbine.solver import AdaptiveTurbineProblem
from adapt_utils.unsteady.test_cases.turbine_array.options import TurbineArrayOptions


# --- Parse arguments

parser = argparse.ArgumentParser()
parser.add_argument("-plot_only", help="If True, the QoI is plotted and no simulations are run")
args = parser.parse_args()


# --- Set parameters

plot_pvd = True
# plot_pvd = False
op = TurbineArrayOptions(approach='fixed_mesh', plot_pvd=plot_pvd, num_meshes=1)
L = op.domain_length
W = op.domain_width
op.end_time = op.T_ramp
plot_only = bool(args.plot_only or False)
ramp_dir = create_directory(os.path.join(os.path.dirname(__file__), "data", "ramp"))
op.di = ramp_dir
swp = AdaptiveTurbineProblem(op, callback_dir=ramp_dir, ramp_dir=ramp_dir)


# --- Run forward model; export solution tuple and QoI timeseries

if plot_only:
    swp.load_state(0, ramp_dir)
else:
    cpu_timestamp = perf_counter()
    swp.solve_forward()
    cpu_time = perf_counter() - cpu_timestamp
    msg = "Total CPU time: {:.1f} seconds / {:.1f} minutes / {:.3f} hours"
    msg = msg.format(cpu_time, cpu_time/60, cpu_time/3600)
    msg += "\nAverage power output of array: {:.1f}W".format(swp.average_power_output())
    print_output(msg)
    with open(os.path.join(ramp_dir, "log"), "w+") as logfile:
        logfile.write(msg + "\n")
    op.plot_pvd = True
    swp.export_state(0, ramp_dir)


# --- Plot

plot_dir = create_directory(os.path.join(os.path.dirname(__file__), 'plots'))

# Get fluid speed and elevation in P1 space
u, eta = swp.fwd_solutions[0].split()
speed = interpolate(sqrt(dot(u, u)), swp.P1[0])
eta_proj = project(eta, swp.P1[0])

# Plot fluid speed
fig, axes = plt.subplots(figsize=(14, 6))
cbar = fig.colorbar(tricontourf(speed, axes=axes, levels=50, cmap='coolwarm'), ax=axes)
cbar.set_label(r"Fluid speed [$\mathrm{m\,s}^{-1}$]")
axes.set_xlim([-L/2, L/2])
axes.set_ylim([-W/2, W/2])
axes.set_xlabel(r"$x$-coordinate $[\mathrm m]$")
axes.set_ylabel(r"$y$-coordinate $[\mathrm m]$")
axes.set_yticks(np.linspace(-W/2, W/2, 5))
plt.tight_layout()
for ext in ("png", "pdf"):
    plt.savefig(os.path.join(plot_dir, ".".join(["speed", ext])))

# Plot elevation
fig, axes = plt.subplots(figsize=(14, 6))
cbar = fig.colorbar(tricontourf(eta_proj, axes=axes, levels=50, cmap='coolwarm'), ax=axes)
cbar.set_label(r"Elevation [$\mathrm m$]")
axes.set_xlim([-L/2, L/2])
axes.set_ylim([-W/2, W/2])
axes.set_xlabel(r"$x$-coordinate $[\mathrm m]$")
axes.set_ylabel(r"$y$-coordinate $[\mathrm m]$")
axes.set_yticks(np.linspace(-W/2, W/2, 5))
plt.tight_layout()
for ext in ("png", "pdf"):
    plt.savefig(os.path.join(plot_dir, ".".join(["elevation", ext])))
