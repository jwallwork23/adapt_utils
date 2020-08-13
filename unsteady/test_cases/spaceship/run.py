from thetis import create_directory, print_output, File

import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
from time import perf_counter

from adapt_utils.unsteady.swe.turbine.solver import AdaptiveTurbineProblem
from adapt_utils.unsteady.test_cases.spaceship.options import SpaceshipOptions


# --- Set parameters

parser = argparse.ArgumentParser()
parser.add_argument("-approach", help="Mesh adaptation strategy")
parser.add_argument("-plot_only", help="If True, the QoI is plotted and no simulations are run")
parser.add_argument("-viscosity_sponge_type", help="""
If set, a viscosity sponge is used to the forced boundary. Choose from 'linear' or 'exponential'.""")
parser.add_argument("-stabilisation", help="""
If set, must be 'lax_friedrichs'. Otherwise, no stabilisation is used.""")
args = parser.parse_args()

approach = args.approach or 'fixed_mesh'
plot_only = bool(args.plot_only or False)

kwargs = {
    "solver_parameters": {  # TODO: Temporary
        "shallow_water": {
            # "mat_type": "aij",
            # "snes_monitor": None,
            "snes_converged_reason": None,
            "ksp_type": "gmres",
            # "ksp_type": "preonly",
            # "ksp_monitor": None,
            "ksp_converged_reason": None,
            # "pc_type": "fieldsplit",
            # "pc_type": "lu",
            # "pc_factor_mat_solver_type": "mumps",
            "pc_fieldsplit_type": "multiplicative",
            "fieldsplit_U_2d": {
                "ksp_type": "preonly",
                "ksp_max_it": 10000,
                "ksp_rtol": 1.0e-05,
                "pc_type": "sor",
                # "ksp_view": None,
                # "ksp_converged_reason": None,
            },
            "fieldsplit_H_2d": {
                "ksp_type": "preonly",
                "ksp_max_it": 10000,
                "ksp_rtol": 1.0e-05,
                # "pc_type": "sor",
                "pc_type": "jacobi",
                # "ksp_view": None,
                # "ksp_converged_reason": None,
            },
        },
    },

    # Model
    "stabilisation": args.stabilisation,
    "viscosity_sponge_type": args.viscosity_sponge_type,
    "family": "dg-cg",

    # I/O
    'plot_pvd': True,
}

op = SpaceshipOptions(approach=approach)
op.update(kwargs)
if op.viscosity_sponge_type is not None:
    op.di = create_directory(os.path.join(op.di, op.viscosity_sponge_type))


# --- Run model

# I/O
data_dir = create_directory(os.path.join(os.path.dirname(__file__), 'data', 'fixed_mesh'))
fname = os.path.join(data_dir, approach, 'power_output.npy')

# Create solver object
tp = AdaptiveTurbineProblem(op, callback_dir=data_dir)

# Plot bathymetry and viscosity
tp.bathymetry_file.write(tp.bathymetry[0])
File(os.path.join(op.di, "viscosity.pvd")).write(tp.fields[0].horizontal_viscosity)

# Run forward model and save QoI timeseries
if not plot_only:
    cpu_timestamp = perf_counter()
    tp.solve()
    cpu_time = perf_counter() - cpu_timestamp
    msg = "Total CPU time: {:.1f} seconds / {:.1f} minutes / {:.3f} hours"
    print_output(msg.format(cpu_time, cpu_time/60, cpu_time/3600))
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
time_seconds = np.linspace(0.0, op.end_time, num_timesteps) - op.T_ramp
time_hours = time_seconds/3600
axes.plot(time_hours, power_kilowatts)
axes.set_xlabel("Time [h]")
axes.set_ylabel("Power output [kW]")

# Add second x-axis with non-dimensionalised time
non_dimensionalise = lambda time: 3600*time/op.T_tide
dimensionalise = lambda time: 3600*time*op.T_tide
secax = axes.secondary_xaxis('top', functions=(non_dimensionalise, dimensionalise))
secax.set_xlabel("Time/Tidal period")

# Save
plot_dir = create_directory(os.path.join(os.path.dirname(__file__), 'plots'))
plt.savefig(os.path.join(plot_dir, '_'.join([approach, 'power_output.pdf'])))
