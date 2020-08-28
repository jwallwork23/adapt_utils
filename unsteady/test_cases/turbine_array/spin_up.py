from thetis import create_directory, print_output

import numpy as np
import os
from time import perf_counter

from adapt_utils.io import export_hydrodynamics
from adapt_utils.unsteady.swe.turbine.solver import AdaptiveTurbineProblem
from adapt_utils.unsteady.test_cases.turbine_array.options import TurbineArrayOptions


# Set up parameters
op = TurbineArrayOptions(approach='fixed_mesh', plot_pvd=False)
op.end_time = op.T_ramp

# Run forward model and save QoI timeseries
data_dir = create_directory(os.path.join(os.path.dirname(__file__), "data", "ramp"))
swp = AdaptiveTurbineProblem(op, callback_dir=data_dir)
cpu_timestamp = perf_counter()
swp.solve()
cpu_time = perf_counter() - cpu_timestamp

# Print / log stats  # TODO: Use Logger class
msg = "Total CPU time: {:.1f} seconds / {:.1f} minutes / {:.3f} hours"
msg = msg.format(cpu_time, cpu_time/60, cpu_time/3600)
average_power = swp.quantity_of_interest()/op.end_time
msg += "\nAverage power output of array: {:.1f}W".format(average_power)
print_output(msg)
with open(os.path.join(data_dir, "log"), "w+") as logfile:
    logfile.write(msg + "\n")

# Export final state to file to use later as an initial condition
op.plot_pvd = True
export_hydrodynamics(*swp.fwd_solutions[0].split(), data_dir, plexname=None, op=op)
