from thetis import create_directory

import matplotlib.pyplot as plt
import numpy as np
import os

from adapt_utils.plotting import *  # NOQA
from adapt_utils.unsteady.test_cases.turbine_array.options import TurbineArrayOptions


# Set parameters
plt.rc('font', **{'size': 16})
op = TurbineArrayOptions(1.0)

# Create parameter object and get time periods
time_seconds = np.linspace(0.0, 2*op.T_tide, 1001)
time_hours = time_seconds/3600

# Get forcing functions
hmax = op.max_amplitude
omega = 2*np.pi/op.T_tide
left_hand_forcing = lambda t: hmax*np.cos(omega*(t))
right_hand_forcing = lambda t: hmax*np.cos(omega*(t) + np.pi)

# Plot the forcings
fig, axes = plt.subplots(figsize=(8, 5))
axes.plot(
    time_hours, left_hand_forcing(time_seconds),
    label="Western boundary", color="C0",
)
axes.plot(
    time_hours, right_hand_forcing(time_seconds),
    label="Eastern boundary", color="C2",
)
axes.set_ylim([-0.6, 0.6])
axes.set_xlabel("Time [hours]")
axes.set_ylabel("Tidal forcing [m]")
axes.grid(True)
box = axes.get_position()
axes.set_position([box.x0, box.y0 + box.height*0.1, box.width, box.height])
axes.legend(loc='lower center', bbox_to_anchor=(0.5, -0.5), ncol=2, fontsize=18)

# Add second x-axis with non-dimensionalised time
non_dimensionalise = lambda time: 3600*time/op.T_tide
dimensionalise = lambda time: 3600*time*op.T_tide
secax = axes.secondary_xaxis('top', functions=(non_dimensionalise, dimensionalise))
secax.set_xlabel("Time/Tidal period")

# Save plot
plot_dir = create_directory(os.path.join(os.path.dirname(__file__), "plots"))
savefig("tidal_forcing", plot_dir, extensions=["png", "pdf"])
