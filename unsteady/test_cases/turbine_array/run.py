from thetis import create_directory, print_output

import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
from time import perf_counter

from adapt_utils.unsteady.swe.turbine.solver import AdaptiveTurbineProblem
from adapt_utils.unsteady.test_cases.turbine_array.options import TurbineArrayOptions


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

# Run forward model and save QoI timeseries
data_dir = create_directory(os.path.join(os.path.dirname(__file__), 'data'))
fname = os.path.join(data_dir, '_'.join([approach, 'power_output.npy']))
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
sea_water_density = 1030.0
power_watts = np.load(fname)*sea_water_density
power_kilowatts = power_watts/1.0e+03

# Plot power timeseries
fig, axes = plt.subplots()
num_timesteps = len(qoi_timeseries)
time_hours = np.linspace(0.0, op.end_time/3600, num_timesteps)
axes.plot(time_hours, power_kilowatts)
axes.set_xlabel("Time [h]")
axes.set_ylabel("Power output [kW]")
plot_dir = create_directory(os.path.join(os.path.dirname(__file__), 'plots'))
plt.savefig(os.path.join(plot_dir, '_'.join([approach, 'power_output.pdf'])))

# Plot power timeseries in non-dimensionalised time  # TODO: Just add a second x-axis to the above
fig, axes = plt.subplots()
time_non_dim = np.linspace(0.0, op.end_time/op.T_tide, num_timesteps)
axes.plot(time_non_dim, power_kilowatts)
axes.set_xlabel("Time/Tidal period")
axes.set_ylabel("Power output [kW]")
plt.savefig(os.path.join(plot_dir, '_'.join([approach, 'power_output_non_dimensional_time.pdf'])))
plt.show()

# TODO: Plot power output of individual turbines
# TODO: Version of plot without spin up period
