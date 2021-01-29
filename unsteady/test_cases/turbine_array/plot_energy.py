import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

from adapt_utils.plotting import *
from adapt_utils.unsteady.test_cases.turbine_array.options import TurbineArrayOptions


# --- Parse arguments

parser = argparse.ArgumentParser()
parser.add_argument("-plot_pdf", help="Toggle plotting to .pdf")
parser.add_argument("-plot_png", help="Toggle plotting to .png")
parser.add_argument("-plot_all", help="Toggle plotting to .pdf, .png and .pvd")
parser.add_argument("-plot_only", help="Just plot using saved data")
args = parser.parse_args()


# --- Set parameters

# Parsed arguments
plot_pdf = bool(args.plot_pdf or False)
plot_png = bool(args.plot_png or False)
plot_all = bool(args.plot_all or False)
plot_only = bool(args.plot_only or False)
if plot_only:
    plot_all = True
if plot_all:
    plot_pdf = plot_png = True
plot_any = plot_pdf or plot_png
extensions = []
if plot_pdf:
    extensions.append('pdf')
if plot_png:
    extensions.append('png')
if len(extensions) == 0:
    print("Nothing to plot.")
    sys.exit(0)

# Create parameters object
op = TurbineArrayOptions(1)
dt_per_cycle = int(op.T_tide/op.dt)
data_dir = os.path.join(os.path.dirname(__file__), 'data')
ramp_dir = os.path.join(data_dir, 'ramp_3.855cycle_nu0.001_ReMax1000_dxfarm{:d}')
plot_dir = os.path.join(os.path.dirname(__file__), 'plots')


# --- Plot energy output over last tidal cycle in spin up  # TODO: First cycle of actual simulation

resolutions = [24, 12, 6, 5, 4, 3]
elements = [24726, 33682, 51610, 58816, 149794, 186374]  # TODO: Automate
elements_used = []
columnar_energy_megajoules = []
array_energy_megajoules = []
for i, dxfarm in enumerate(resolutions):
    di = ramp_dir.format(dxfarm)

    # Load power output data
    if np.all([os.path.isfile(os.path.join(di, f + ".h5")) for f in ('velocity', 'elevation')]):
        elements_used.append(elements[i])
    else:
        print("Need to spin up dxfarm = {:d}".format(dxfarm))
        continue
    power_watts = [np.array([]) for i in range(15)]
    for i, turbine in enumerate(op.farm_ids):
        fname = os.path.join(di, "power_output_{:d}_00000.npy".format(turbine))
        power_watts[i] = np.append(power_watts[i], np.load(fname)*op.sea_water_density)
    num_timesteps = len(power_watts[0])
    power_watts = np.array(power_watts).reshape((3, 5, num_timesteps))

    # Get power timeseries
    columnar_power_watts = np.sum(power_watts, axis=0)
    array_power_watts = np.sum(columnar_power_watts, axis=0)

    # Compute energy outputs over final cycle
    columnar_energy_joules = np.sum(columnar_power_watts[-dt_per_cycle:], axis=1)
    columnar_energy_megajoules.append(columnar_energy_joules/1.0e+06)
    array_energy_joules = np.sum(array_power_watts[-dt_per_cycle:])
    array_energy_megajoules.append(array_energy_joules/1.0e+06)
columnar_energy_megajoules = np.transpose(columnar_energy_megajoules)

# Plot energy output for whole array
fig, axes = plt.subplots(figsize=(8, 3.5))
axes.semilogx(elements_used, array_energy_megajoules, '-x', color="grey")
axes.set_xlabel("Element count")
axes.set_ylabel(r"Energy output [$\mathrm{MJ}$]")
axes.grid(True)
savefig("array_energy_output_ramp", plot_dir, extensions=extensions)

# Plot energy output for each column
fig, axes = plt.subplots(figsize=(8, 3.5))
greys = ['k', 'dimgrey', 'grey', 'darkgrey', 'silver', 'lightgrey']
for i, (linestyle, colour) in enumerate(zip(["-", "--", ":", "--", "-"], greys)):
    axes.semilogx(elements_used, columnar_energy_megajoules[i, :],
                  label="{:d}".format(i+1), linestyle=linestyle, marker='x', color=colour)
axes.set_xlabel("Element count")
axes.set_ylabel(r"Energy output [$\mathrm{MJ}$]")
axes.legend(bbox_to_anchor=(1.05, 1.2), fontsize=16)
axes.grid(True)
savefig("columnar_energy_output_ramp", plot_dir, extensions=extensions)
