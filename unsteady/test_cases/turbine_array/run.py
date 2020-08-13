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
op = TurbineArrayOptions(approach=approach)
op.update(kwargs)


# --- Run model

# Run forward model and save QoI timeseries
data_dir = create_directory(os.path.join(os.path.dirname(__file__), "data"))
fname = os.path.join(data_dir, "_".join([approach, "power_output.npy"]))
if not plot_only:
    tp = AdaptiveTurbineProblem(op)
    cpu_timestamp = perf_counter()
    tp.solve()
    cpu_time = perf_counter() - cpu_timestamp
    msg = "Total CPU time: {:.1f} seconds / {:.1f} minutes / {:.3f} hours"
    print_output(msg.format(cpu_time, cpu_time/60, cpu_time/3600))
    np.save(fname, tp.callbacks[0]["timestep"]["power_output_everywhere"].timeseries)
    average_power = tp.quantity_of_interest()/op.end_time
    print_output("Average power output of array: {:.1f}W".format(average_power))

# Adjust timeseries to account for density of water
if not os.path.exists(fname):
    raise IOError("Need to run the model in order to get power output timeseries.")
sea_water_density = 1030.0
power_watts = np.load(fname)*sea_water_density
power_kilowatts = power_watts/1.0e+03


# --- Plot power timeseries

fig, axes = plt.subplots()
num_timesteps = len(power_watts)

# Convert to appropriate units and plot
r = op.T_ramp
time_seconds = np.linspace(0.0, op.end_time, num_timesteps) - op.T_ramp
time_hours = time_seconds/3600
axes.plot(time_hours, power_kilowatts)
axes.set_xlabel("Time [h]")
axes.set_ylabel("Power output [kW]")

# Add a dashed line when the ramp period is over
ylim = axes.get_ylim()
axes.axvline(0.0, *ylim, linestyle='--', color='k')
axes.set_ylim(ylim)
plotting_kwargs = {
    "arrowprops": {
        "arrowstyle": "<->",
    },
}
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
    plt.savefig(os.path.join(plot_dir, '_'.join([approach, ".".join(["power_output", ext])])))
