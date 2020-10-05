from thetis import create_directory, print_output, COMM_WORLD

import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from time import perf_counter

from adapt_utils.io import get_date, index_string
from adapt_utils.plotting import *
from adapt_utils.unsteady.swe.turbine.solver import AdaptiveTurbineProblem
from adapt_utils.unsteady.test_cases.turbine_array.options import TurbineArrayOptions


# --- Parse arguments

parser = argparse.ArgumentParser()
parser.add_argument("-num_meshes", help="Number of meshes (for debugging)")
parser.add_argument("-end_time", help="End time of simulation in seconds")

# Physics
parser.add_argument("-max_reynolds_number", help="Maximum tolerated mesh Reynolds number")
parser.add_argument("-base_viscosity", help="Base viscosity (default 1).")
parser.add_argument("-target_viscosity", help="Target viscosity (defaults to base value).")

# I/O and debugging
parser.add_argument("-extension", help="Optional extension for output directory")
parser.add_argument("-load_mesh", help="Load meshes from a previous run")
parser.add_argument("-plot_pdf", help="Toggle plotting to .pdf")
parser.add_argument("-plot_png", help="Toggle plotting to .png")
parser.add_argument("-plot_pvd", help="Toggle plotting to .pvd")
parser.add_argument("-plot_all", help="Toggle plotting to .pdf, .png and .pvd")
parser.add_argument("-plot_only", help="Just plot using saved data")
parser.add_argument("-debug", help="Print all debugging statements")
parser.add_argument("-debug_mode", help="Choose debugging mode from 'basic' and 'full'")

args = parser.parse_args()


# --- Set parameters

approach = 'fixed_mesh'
load_mesh = None if args.load_mesh is None else 'plex'
plot_pvd = bool(args.plot_pvd or False)
plot_pdf = bool(args.plot_pdf or False)
plot_png = bool(args.plot_png or False)
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
base_viscosity = float(args.base_viscosity or 0.0)
kwargs = {
    'approach': approach,
    'num_meshes': int(args.num_meshes or 1),
    'target_viscosity': float(args.target_viscosity or base_viscosity),
    'plot_pvd': plot_pvd,
    'debug': bool(args.debug or False),
    'debug_mode': args.debug_mode or 'basic',
}
if args.end_time is not None:
    kwargs['end_time'] = float(args.end_time)
op = TurbineArrayOptions(base_viscosity, **kwargs)
if args.max_reynolds_number is not None:
    op.max_reynolds_number = float(args.max_reynolds_number)
index_str = index_string(op.num_meshes)

# Create directories and check if spun-up solution exists
ramp_dir = os.path.join(os.path.dirname(__file__), "data", "ramp")
if args.extension is not None:
    ramp_dir = "_".join([ramp_dir, args.extension])
op.di = create_directory(os.path.join(os.path.dirname(__file__), "outputs", approach))
if args.extension is not None:
    op.di = "_".join([op.di, args.extension])
op.spun = np.all([os.path.isfile(os.path.join(ramp_dir, f + ".h5")) for f in ('velocity', 'elevation')])


# --- Run model

# Run forward model and save QoI timeseries
if not plot_only:
    if not op.spun:
        raise ValueError("Spin-up data not found.")
    swp = AdaptiveTurbineProblem(op, callback_dir=op.di, ramp_dir=ramp_dir, load_mesh=load_mesh)

    # Solve forward problem
    cpu_timestamp = perf_counter()
    swp.solve_forward()
    cpu_time = perf_counter() - cpu_timestamp
    logstr = "Total CPU time: {:.1f} seconds / {:.1f} minutes / {:.3f} hours\n"
    logstr = logstr.format(cpu_time, cpu_time/60, cpu_time/3600)
    energy_output = op.sea_water_density*swp.energy_output()
    logstr += "Total energy output of array: {:.1f}J\n".format(energy_output)
    average_power_output = energy_output/op.end_time
    logstr += "Average power output of array: {:.1f}W".format(average_power_output)
    print_output(logstr)
    # TODO: Peak power output
    with open(os.path.join(op.di, 'log_{:s}'.format(get_date())), 'w+') as logfile:
        logfile.write(logstr)

# Do not attempt to plot in parallel
nproc = COMM_WORLD.size
if nproc > 1:
    msg = "Will not attempt to plot with {:d} processors. Run again in serial flagging -plot_only."
    print_output(msg.format(nproc))
    sys.exit(0)
elif not plot_any:
    sys.exit(0)
plt.rc('font', **{'size': 18})

# Adjust timeseries to account for density of water and assemble as an array
power_watts = [np.array([]) for i in range(15)]
for i, turbine in enumerate(op.farm_ids):
    timeseries = np.array([])
    for n in range(op.num_meshes):
        fname = os.path.join(op.di, "power_output_{:d}_{:s}.npy".format(turbine, index_string(n)))
        if not os.path.exists(fname):
            raise IOError("Need to run the model in order to get power output timeseries.")
        timeseries = np.append(timeseries, np.load(fname))
    power_watts[i] = np.append(power_watts[i], timeseries*op.sea_water_density)
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
fig, axes = plt.subplots(figsize=(6, 4))
time_seconds = np.linspace(0, op.end_time, num_timesteps)
time_hours = time_seconds/3600
axes.plot(time_hours, array_power_kilowatts, color="grey")
axes.set_xlabel("Time [h]")
axes.set_ylabel("Array power output [kW]")
axes.set_xlim([0, op.end_time/3600])

# Add second x-axis with non-dimensionalised time
non_dimensionalise = lambda time: 3600*time/op.T_tide
dimensionalise = lambda time: 3600*time*op.T_tide
secax = axes.secondary_xaxis('top', functions=(non_dimensionalise, dimensionalise))
secax.set_xlabel("Time/Tidal period")

# Save
plot_dir = create_directory(os.path.join(os.path.dirname(__file__), "plots"))
fname = approach
if args.extension is not None:
    fname = '_'.join([fname, args.extension])
savefig('_'.join([fname, "array_power_output", index_str]), plot_dir, extensions=extensions)


# --- Plot power timeseries of each column of the array

# Convert to appropriate units and plot
fig, axes = plt.subplots(figsize=(6, 4))
greys = ['k', 'dimgrey', 'grey', 'darkgrey', 'silver', 'lightgrey']
for i, (linestyle, colour) in enumerate(zip(["-", "--", ":", "--", "-"], greys)):
    axes.plot(time_hours, columnar_power_kilowatts[i, :],
              label="{:d}".format(i+1), linestyle=linestyle, color=colour)
axes.set_xlabel("Time [h]")
axes.set_ylabel("Power output [kW]")
axes.set_xlim([0, op.end_time/3600])
axes.legend(bbox_to_anchor=(1.25, 0.9), handlelength=1, fontsize=16)

# Add second x-axis with non-dimensionalised time
secax = axes.secondary_xaxis('top', functions=(non_dimensionalise, dimensionalise))
secax.set_xlabel("Time/Tidal period")

# Save
savefig('_'.join([fname, "columnar_power_output", index_str]), plot_dir, extensions=extensions)
