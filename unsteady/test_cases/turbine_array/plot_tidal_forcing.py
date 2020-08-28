from thetis import create_directory

import matplotlib.pyplot as plt
import numpy as np
import os

from adapt_utils.plotting import *
from adapt_utils.unsteady.test_cases.turbine_array.options import TurbineArrayOptions


# Set parameters
plt.rc('font', **{'size': 16})
plotting_kwargs = {
    "annotation_clip": False,
    "color": "b",
    "arrowprops": {
        "color": "b",
        "arrowstyle": "<->",
    },
}
op = TurbineArrayOptions()

# Create parameter object and get time periods
time_seconds = np.linspace(0.0, op.end_time, 1001) - op.T_ramp
time_hours = time_seconds/3600

# Get forcing functions
hmax = op.max_amplitude
omega = 2*np.pi/op.T_tide
left_hand_forcing = lambda t: hmax*np.cos(omega*(t - op.T_ramp))
right_hand_forcing = lambda t: hmax*np.cos(omega*(t - op.T_ramp) + np.pi)

# Plot the forcings
fig, axes = plt.subplots(figsize=(8, 4))
axes.plot(time_hours, left_hand_forcing(time_seconds), label="Western boundary")
axes.plot(time_hours, right_hand_forcing(time_seconds), label="Eastern boundary")
axes.set_ylim([-0.6, 0.6])
axes.set_xlabel("Time [hours]")
axes.set_ylabel("Tidal forcing [m]")

# Add a dashed line when the ramp period is over
axes.axvline(0.0, linestyle='--', color='k')
r = op.T_ramp/3600
axes.annotate("", xy=(-r, -0.8), xytext=(0.0, -0.8), **plotting_kwargs)
axes.annotate("Spin-up period", xy=(-0.8*r, -0.9), xytext=(-0.8*r, -0.9), color="b", annotation_clip=False)
axes.legend(loc='upper left')

# Add second x-axis with non-dimensionalised time
non_dimensionalise = lambda time: 3600*time/op.T_tide
dimensionalise = lambda time: 3600*time*op.T_tide
secax = axes.secondary_xaxis('top', functions=(non_dimensionalise, dimensionalise))
secax.set_xlabel("Time/Tidal period")

# Save plot
plot_dir = create_directory(os.path.join(os.path.dirname(__file__), "plots"))
plt.tight_layout()
for ext in ("png", "pdf"):
    plt.savefig(os.path.join(plot_dir, ".".join(["tidal_forcing", ext])))
