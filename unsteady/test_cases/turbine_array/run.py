from thetis import create_directory, print_output

import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
from time import perf_counter

from adapt_utils.unsteady.swe.turbine.solver import AdaptiveTurbineProblem
from adapt_utils.unsteady.test_cases.turbine_array.options import TurbineArrayOptions


# --- Set parameters

parser = argparse.ArgumentParser()
parser.add_argument("-approach", help="Mesh adaptation strategy")
parser.add_argument("-plot_only", help="If True, the QoI is plotted and no simulations are run")
args = parser.parse_args()

approach = args.approach or 'fixed_mesh'
plot_only = bool(args.plot_only or False)

# Set up parameters
kwargs = {
    'plot_pvd': True,
}
plotting_kwargs = {
    "arrowprops": {
        "arrowstyle": "<->",
    },
}
op = TurbineArrayOptions(approach=approach)
op.update(kwargs)


# --- Run model

# Run forward model and save QoI timeseries
data_dir = create_directory(os.path.join(os.path.dirname(__file__), "data", approach))
if not plot_only:
    tp = AdaptiveTurbineProblem(op, callback_dir=data_dir)
    cpu_timestamp = perf_counter()
    tp.solve()
    cpu_time = perf_counter() - cpu_timestamp
    msg = "Total CPU time: {:.1f} seconds / {:.1f} minutes / {:.3f} hours"
    print_output(msg.format(cpu_time, cpu_time/60, cpu_time/3600))
    average_power = tp.quantity_of_interest()/op.end_time
    print_output("Average power output of array: {:.1f}W".format(average_power))

# Adjust timeseries to account for density of water and assemble as an array
power_watts = []
sea_water_density = 1030.0
for turbine in op.farm_ids:
    fname = os.path.join(data_dir, "power_output_{:d}.npy".format(turbine))
    if not os.path.exists(fname):
        raise IOError("Need to run the model in order to get power output timeseries.")
    power_watts.append(np.load(fname)*sea_water_density)
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
fig, axes = plt.subplots()
time_seconds = np.linspace(0.0, op.end_time, num_timesteps) - op.T_ramp
time_hours = time_seconds/3600
time_hours = time_hours[:num_timesteps]
axes.plot(time_hours, array_power_kilowatts)
axes.set_xlabel("Time [h]")
axes.set_ylabel("Array power output [kW]")

# Add a dashed line when the ramp period is over
axes.axvline(0.0, linestyle='--', color='k')
r = op.T_ramp/3600
axes.annotate("", xy=(-r, 0.0), xytext=(0.0, 0.0), annotation_clip=False, **plotting_kwargs)
axes.annotate("Spin-up period", xy=(-0.8*r, -3.2), xytext=(-0.8*r, -3.2), annotation_clip=False)

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
fig, axes = plt.subplots()
for i in range(5):
    axes.plot(time_hours, columnar_power_kilowatts[i, :], label="Column {:d}".format(i+1))
axes.set_xlabel("Time [h]")
axes.set_ylabel("Power output [kW]")
axes.legend(loc="upper left")

# Add a dashed line when the ramp period is over
axes.axvline(0.0, linestyle='--', color='k')
axes.annotate("", xy=(-r, 0.0), xytext=(0.0, 0.0), annotation_clip=False, **plotting_kwargs)
axes.annotate("Spin-up period", xy=(-0.8*r, -3.2), xytext=(-0.8*r, -3.2), annotation_clip=False)

# Add second x-axis with non-dimensionalised time
secax = axes.secondary_xaxis('top', functions=(non_dimensionalise, dimensionalise))
secax.set_xlabel("Time/Tidal period")

# Save
plt.tight_layout()
for ext in ("png", "pdf"):
    plt.savefig(os.path.join(plot_dir, '_'.join([approach, ".".join(["columnar_power_output", ext])])))
