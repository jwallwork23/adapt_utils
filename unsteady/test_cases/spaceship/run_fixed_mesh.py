from thetis import create_directory, print_output, File, COMM_WORLD

import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from time import perf_counter

from adapt_utils.plotting import *  # NOQA
from adapt_utils.swe.turbine.solver import AdaptiveTurbineProblem
from adapt_utils.unsteady.test_cases.spaceship.options import SpaceshipOptions


# --- Parse arguments

parser = argparse.ArgumentParser()
parser.add_argument("-num_meshes", help="Number of meshes (for debugging)")
parser.add_argument("-viscosity_sponge_type", help="""
    If set, a viscosity sponge is used to the forced boundary. Choose from 'linear' or 'exponential'.
    """)
parser.add_argument("-stabilisation", help="""
    If set, must be 'lax_friedrichs'. Otherwise, no stabilisation is used.
    """)
parser.add_argument("-plot_pdf", help="Toggle plotting to .pdf")
parser.add_argument("-plot_png", help="Toggle plotting to .png")
parser.add_argument("-plot_pvd", help="Toggle plotting to .pvd")
parser.add_argument("-plot_all", help="Toggle plotting to .pdf, .png and .pvd")
parser.add_argument("-plot_only", help="Just plot using saved data")
parser.add_argument("-debug", help="Toggle debugging mode")
parser.add_argument("-debug_mode", help="Choose debugging mode from 'basic' and 'full'")
args = parser.parse_args()


# --- Set parameters

approach = 'fixed_mesh'
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
kwargs = {
    'approach': approach,
    'num_meshes': int(args.num_meshes or 1),

    # Model
    "stabilisation": args.stabilisation,
    "viscosity_sponge_type": args.viscosity_sponge_type,  # NOTE: Defaults to None
    "family": "dg-cg",

    # I/O and debugging
    "plot_pvd": plot_pvd,
    "debug": bool(args.debug or False),
    'debug_mode': args.debug_mode or 'basic',
}
plt.rc('font', **{'size': 18})
op = SpaceshipOptions(approach=approach)
op.update(kwargs)
if op.viscosity_sponge_type is not None:
    op.di = create_directory(os.path.join(op.di, op.viscosity_sponge_type))
if op.debug:
    op.solver_parameters_momentum['snes_monitor'] = None
    op.solver_parameters_pressure['snes_monitor'] = None

# Create directories and check if spun-up solution exists
data_dir = create_directory(os.path.join(os.path.dirname(__file__), "data"))
ramp_dir = create_directory(os.path.join(data_dir, "ramp"))
data_dir = create_directory(os.path.join(data_dir, approach, index_str))
op.spun = np.all([os.path.isfile(os.path.join(ramp_dir, f + ".h5")) for f in ('velocity', 'elevation')])
sea_water_density = 1030.0
power_watts = [np.array([]) for i in range(15)]
if op.spun:
    for i, turbine in enumerate(op.farm_ids):
        fname = os.path.join(ramp_dir, "power_output_{:d}_00000.npy".format(turbine))
        power_watts[i] = np.append(power_watts[i], np.load(fname)*sea_water_density)
else:
    print_output("Spin-up data not found. Spinning up now.")
    op.end_time += op.T_ramp


# --- Run model

# Create solver object
swp = AdaptiveTurbineProblem(op, callback_dir=data_dir, ramp_dir=ramp_dir, load_mesh=load_mesh)

# Plot bathymetry and viscosity
swp.bathymetry_file.write(swp.bathymetry[0])
File(os.path.join(op.di, "viscosity.pvd")).write(swp.fields[0].horizontal_viscosity)

# Run forward model and save QoI timeseries
if not plot_only:
    cpu_timestamp = perf_counter()
    swp.solve_forward()
    cpu_time = perf_counter() - cpu_timestamp
    msg = "Total CPU time: {:.1f} seconds / {:.1f} minutes / {:.3f} hours"
    print_output(msg.format(cpu_time, cpu_time/60, cpu_time/3600))
    print_output("Average power output of array: {:.1f}W".format(swp.average_power_output()))
if not op.spun:
    op.end_time -= op.T_ramp

# Do not attempt to plot in parallel
nproc = COMM_WORLD.size
if nproc > 1:
    msg = "Will not attempt to plot with {:d} processors. Run again in serial flagging -plot_only."
    print_output(msg.format(nproc))
    sys.exit(0)
elif not plot_any:
    sys.exit(0)

# Adjust timeseries to account for density of water and assemble as an array
for i, turbine in enumerate(op.farm_ids):
    timeseries = np.array([])
    for n in range(op.num_meshes):
        fname = os.path.join(data_dir, "power_output_{:d}_{:s}.npy".format(turbine, index_string(n)))
        if not os.path.exists(fname):
            raise IOError("Need to run the model in order to get power output timeseries.")
        timeseries = np.append(timeseries, np.load(fname))
    power_watts[i] = np.append(power_watts[i], timeseries*sea_water_density)
num_timesteps = len(power_watts[0])
power_watts = np.array(power_watts).reshape((2, num_timesteps))

# Get total power
array_power_watts = np.sum(power_watts, axis=0)
array_power_kilowatts = array_power_watts/1.0e+03


# --- Plot power timeseries

# Convert to appropriate units and plot
fig, axes = plt.subplots()
time_seconds = np.linspace(-op.T_ramp, op.end_time, num_timesteps)
time_hours = time_seconds/3600
axes.plot(time_hours, power_kilowatts, color="grey")
axes.set_xlabel("Time [h]")
axes.set_ylabel("Power output [kW]")

# Add a dashed line when the ramp period is over
axes.axvline(0.0, linestyle='--', color="b")
# TODO: Annotate

# Add second x-axis with non-dimensionalised time
non_dimensionalise = lambda time: 3600*time/op.T_tide
dimensionalise = lambda time: 3600*time*op.T_tide
secax = axes.secondary_xaxis('top', functions=(non_dimensionalise, dimensionalise))
secax.set_xlabel("Time/Tidal period")

# Save
plot_dir = create_directory(os.path.join(os.path.dirname(__file__), "plots"))
for ext in extensions:
    plt.savefig(os.path.join(plot_dir, '_'.join([approach, "power_output." + ext])))
