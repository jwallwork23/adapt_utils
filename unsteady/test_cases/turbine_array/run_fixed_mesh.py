from thetis import create_directory, print_output, COMM_WORLD

import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from time import perf_counter

from adapt_utils.unsteady.swe.turbine.solver import AdaptiveTurbineProblem
from adapt_utils.unsteady.test_cases.turbine_array.options import TurbineArrayOptions
from adapt_utils.plotting import *  # NOQA


# --- Parse arguments

parser = argparse.ArgumentParser()
parser.add_argument("-plot_only", help="If True, the QoI is plotted and no simulations are run")
args = parser.parse_args()


# --- Set parameters

approach = 'fixed_mesh'
plot_only = bool(args.plot_only or False)
kwargs = {
    'approach': approach,
    'plot_pvd': True,
    'debug': True,  # TODO: parse
}
plotting_kwargs = {
    "annotation_clip": False,
    "color": "b",
    "arrowprops": {
        "arrowstyle": "<->",
        "color": "b",
    },
}
op = TurbineArrayOptions(**kwargs)

# Create directories and check if spun-up solution exists
data_dir = create_directory(os.path.join(os.path.dirname(__file__), "data"))
ramp_dir = create_directory(os.path.join(data_dir, "ramp"))
data_dir = create_directory(os.path.join(data_dir, approach))
spun = np.all([os.path.isfile(os.path.join(ramp_dir, f + ".h5")) for f in ('velocity', 'elevation')])
if not spun:
    op.end_time += op.T_ramp
sea_water_density = 1030.0
power_watts = [np.array([]) for i in range(15)]


# --- Run model

# Run forward model and save QoI timeseries
if not plot_only:
    swp = AdaptiveTurbineProblem(op, callback_dir=data_dir)

    # Set initial condition
    if spun:
        swp.load_state(0, ramp_dir)
        for i, turbine in enumerate(op.farm_ids):
            fname = os.path.join(ramp_dir, "power_output_{:d}.npy".format(turbine))
            power_watts[i] = np.append(power_watts[i], np.load(fname)*sea_water_density)
    else:
        swp.set_initial_condition()
    swp.setup_solver_forward(0)

    # Solve forward problem
    cpu_timestamp = perf_counter()
    swp.solve_forward(0)
    cpu_time = perf_counter() - cpu_timestamp
    msg = "Total CPU time: {:.1f} seconds / {:.1f} minutes / {:.3f} hours"
    print_output(msg.format(cpu_time, cpu_time/60, cpu_time/3600))
    average_power = swp.quantity_of_interest()/op.end_time
    print_output("Average power output of array: {:.1f}W".format(average_power))

# Do not attempt to plot in parallel
nproc = COMM_WORLD.size
if nproc > 1:
    msg = "Will not attempt to plot with {:d} processors. Run again in serial flagging -plot_only."
    print_output(msg.format(nproc))
    sys.exit(0)

# Adjust timeseries to account for density of water and assemble as an array
for i, turbine in enumerate(op.farm_ids):
    fname = os.path.join(data_dir, "power_output_{:d}.npy".format(turbine))
    if not os.path.exists(fname):
        raise IOError("Need to run the model in order to get power output timeseries.")
    power_watts[i] = np.append(power_watts[i], np.load(fname)*sea_water_density)
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
fig, axes = plt.subplots(figsize=(8, 4))
time_seconds = np.linspace(0.0, op.end_time, num_timesteps) - op.T_ramp
time_hours = time_seconds/3600
time_hours = time_hours[:num_timesteps]
axes.plot(time_hours, array_power_kilowatts, color="grey")
axes.set_xlabel("Time [h]")
axes.set_ylabel("Array power output [kW]")
r = op.T_ramp/3600
axes.set_xlim([-r, op.end_time/3600 - r])

# Add a dashed line when the ramp period is over
axes.axvline(0.0, linestyle='--', color="b")
axes.annotate("", xy=(-r, -25), xytext=(0, -25), **plotting_kwargs)
axes.annotate("Spin-up period", xy=(-0.8*r, -35), xytext=(-0.8*r, -35), color="b", annotation_clip=False)

# Add second x-axis with non-dimensionalised time
non_dimensionalise = lambda time: 3600*time/op.T_tide
dimensionalise = lambda time: 3600*time*op.T_tide
secax = axes.secondary_xaxis('top', functions=(non_dimensionalise, dimensionalise))
secax.set_xlabel("Time/Tidal period")

# Save
plot_dir = create_directory(os.path.join(os.path.dirname(__file__), "plots"))
plt.tight_layout()
for ext in ("png", "pdf"):
    plt.savefig(os.path.join(plot_dir, '_'.join([approach, ".".join(["array_power_output", ext])])))


# --- Plot power timeseries of each column of the array

# Convert to appropriate units and plot
fig, axes = plt.subplots(figsize=(8, 4))
greys = ['k', 'dimgrey', 'grey', 'darkgrey', 'silver', 'lightgrey']
for i, (linestyle, colour) in enumerate(zip(["-", "--", ":", "--", "-"], greys)):
    axes.plot(time_hours, columnar_power_kilowatts[i, :],
              label="Column {:d}".format(i+1), linestyle=linestyle, color=colour)
axes.set_xlabel("Time [h]")
axes.set_ylabel("Power output [kW]")
axes.set_xlim([-r, op.end_time/3600 - r])
axes.legend(loc="upper left")

# Add a dashed line when the ramp period is over
axes.axvline(0.0, linestyle='--', color="b")
axes.annotate("", xy=(-r, -6), xytext=(0.0, -6), **plotting_kwargs)
axes.annotate("Spin-up period", xy=(-0.8*r, -9), xytext=(-0.8*r, -9), color="b", annotation_clip=False)

# Add second x-axis with non-dimensionalised time
secax = axes.secondary_xaxis('top', functions=(non_dimensionalise, dimensionalise))
secax.set_xlabel("Time/Tidal period")

# Save
plt.tight_layout()
for ext in ("png", "pdf"):
    plt.savefig(os.path.join(plot_dir, '_'.join([approach, ".".join(["columnar_power_output", ext])])))
