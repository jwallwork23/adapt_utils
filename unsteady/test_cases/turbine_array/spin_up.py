from thetis import *

import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
from time import perf_counter

from adapt_utils.plotting import *
from adapt_utils.unsteady.swe.turbine.solver import AdaptiveTurbineProblem
from adapt_utils.unsteady.test_cases.turbine_array.options import TurbineArrayOptions


# --- Parse arguments

parser = argparse.ArgumentParser()
parser.add_argument("-reynolds_number", help="Target mesh Reynolds number")
parser.add_argument("-min_viscosity", help="Minimum tolerated viscosity (default 0).")
parser.add_argument("-plot_pdf", help="Toggle plotting to .pdf")
parser.add_argument("-plot_png", help="Toggle plotting to .png")
parser.add_argument("-plot_pvd", help="Toggle plotting to .pvd")
parser.add_argument("-plot_all", help="Toggle plotting to .pdf, .png and .pvd")
parser.add_argument("-plot_only", help="Just plot using saved data")
args = parser.parse_args()


# --- Set parameters

approach = 'fixed_mesh'
plot_pvd = bool(args.plot_pvd or False)
plot_pdf = bool(args.plot_pdf or False)
plot_png = bool(args.plot_png or False)
plot_all = bool(args.plot_all or False)
plot_only = bool(args.plot_only or False)
if plot_all:
    plot_pvd = plot_pdf = plot_png = True
plot_any = plot_pdf or plot_png
extensions = []
if plot_pdf:
    extensions.append('pdf')
if plot_png:
    extensions.append('png')
kwargs = {
    'approach': approach,
    'target_mesh_reynolds_number': None if args.reynolds_number is None else float(args.reynolds_number),
    'min_viscosity': float(args.min_viscosity or 0.0),
    'plot_pvd': plot_pvd,
}
op = TurbineArrayOptions(**kwargs)
L = op.domain_length
W = op.domain_width
op.end_time = op.T_ramp
plot_only = bool(args.plot_only or False)
ramp_dir = create_directory(os.path.join(os.path.dirname(__file__), "data", "ramp"))
op.di = ramp_dir
plot_dir = create_directory(os.path.join(ramp_dir, "plots"))
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

# Do not attempt to plot in parallel
nproc = COMM_WORLD.size
if nproc > 1:
    msg = "Will not attempt to plot with {:d} processors. Run again in serial flagging -plot_only."
    print_output(msg.format(nproc))
    sys.exit(0)
plt.rc('font', **{'size': 18})

# Load power output data
op.spun = np.all([os.path.isfile(os.path.join(ramp_dir, f + ".h5")) for f in ('velocity', 'elevation')])
sea_water_density = 1030.0
power_watts = [np.array([]) for i in range(15)]
if op.spun:
    for i, turbine in enumerate(op.farm_ids):
        fname = os.path.join(ramp_dir, "power_output_{:d}_00000.npy".format(turbine))
        power_watts[i] = np.append(power_watts[i], np.load(fname)*sea_water_density)
else:
    raise ValueError("Spin-up data not found.")
num_timesteps = len(power_watts[0])
power_watts = np.array(power_watts).reshape((3, 5, num_timesteps))

# Get columnwise power
columnar_power_watts = np.sum(power_watts, axis=0)
columnar_power_kilowatts = columnar_power_watts/1.0e+03

# Get total power
array_power_watts = np.sum(columnar_power_watts, axis=0)
array_power_kilowatts = array_power_watts/1.0e+03


# --- Plot power timeseries of whole array

# Convert to appropriate units and plot
fig, axes = plt.subplots(figsize=(8, 3.5))
time_seconds = np.linspace(0, op.T_ramp, num_timesteps)
time_hours = time_seconds/3600
axes.plot(time_hours, array_power_kilowatts, color="grey")
axes.set_xlabel("Time [h]")
axes.set_ylabel("Array power output [kW]")
axes.set_xlim([0, op.T_ramp/3600])

# Add second x-axis with non-dimensionalised time
non_dimensionalise = lambda time: 3600*time/op.T_tide
dimensionalise = lambda time: 3600*time*op.T_tide
secax = axes.secondary_xaxis('top', functions=(non_dimensionalise, dimensionalise))
secax.set_xlabel("Time/Tidal period")

# Save
savefig("array_power_output_ramp", plot_dir, extensions=extensions)


# --- Plot power timeseries of each column of the array

# Convert to appropriate units and plot
fig, axes = plt.subplots(figsize=(8, 3.5))
greys = ['k', 'dimgrey', 'grey', 'darkgrey', 'silver', 'lightgrey']
for i, (linestyle, colour) in enumerate(zip(["-", "--", ":", "--", "-"], greys)):
    axes.plot(time_hours, columnar_power_kilowatts[i, :],
              label="{:d}".format(i+1), linestyle=linestyle, color=colour)
axes.set_xlabel("Time [h]")
axes.set_ylabel("Power output [kW]")
axes.set_xlim([0.5*op.T_ramp/3600, op.T_ramp/3600])
axes.legend(bbox_to_anchor=(1.05, 1.2), fontsize=16)

# Add second x-axis with non-dimensionalised time
secax = axes.secondary_xaxis('top', functions=(non_dimensionalise, dimensionalise))
secax.set_xlabel("Time/Tidal period")

# Save
savefig("columnar_power_output_ramp", plot_dir, extensions=extensions)


# --- Plot the spun-up hydrodynamics

# Get fluid speed and elevation in P1 space
u, eta = swp.fwd_solutions[0].split()
speed = interpolate(sqrt(dot(u, u)), swp.P1[0])
eta_proj = project(eta, swp.P1[0])

# Plot fluid speed
fig, axes = plt.subplots(figsize=(10, 5))
levels = np.linspace(0.0, 1.25, 201)
im = tricontourf(speed, axes=axes, levels=levels, cmap='coolwarm')
cbar = fig.colorbar(im, ax=axes, orientation="horizontal", pad=0.04, aspect=40)
cbar.set_label(r"Fluid speed [$\mathrm{m\,s}^{-1}$]", fontsize=24)
cbar.set_ticks([0, 0.25, 0.5, 0.75, 1.0, 1.25])
cbar.ax.tick_params(labelsize=22)
axes.set_xlim([-L/2, L/2])
axes.set_ylim([-W/2, W/2])
axes.xaxis.tick_top()
for axis in (axes.xaxis, axes.yaxis):
    axis.set_tick_params(labelsize=22)
axes.set_yticks(np.linspace(-W/2, W/2, 5))
savefig("speed", plot_dir, extensions=extensions)

# Plot elevation
fig, axes = plt.subplots(figsize=(10, 5))
levels = np.linspace(-0.5, 0.5, 201)
im = tricontourf(eta_proj, axes=axes, levels=levels, cmap='coolwarm')
cbar = fig.colorbar(im, ax=axes, orientation="horizontal", pad=0.04, aspect=40)
cbar.set_label(r"Elevation [$\mathrm m$]", fontsize=24)
cbar.set_ticks([-0.5, -0.25, 0.0, 0.25, 0.5])
cbar.ax.tick_params(labelsize=22)
axes.set_xlim([-L/2, L/2])
axes.set_ylim([-W/2, W/2])
axes.xaxis.tick_top()
for axis in (axes.xaxis, axes.yaxis):
    axis.set_tick_params(labelsize=22)
axes.set_yticks(np.linspace(-W/2, W/2, 5))
savefig("elevation", plot_dir, extensions=extensions)
