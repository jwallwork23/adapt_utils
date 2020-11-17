from thetis import create_directory

import matplotlib.pyplot as plt
import numpy as np
import os

from adapt_utils.plotting import *
from adapt_utils.unsteady.test_cases.idealised_desalination.options import *


# Set parameters
plt.rc('font', **{'size': 16})
op = IdealisedDesalinationOutfallOptions(spun=True)
time_seconds = np.linspace(0.0, op.end_time, 1001)
time_hours = time_seconds/3600

# Plot the forcing
fig, axes = plt.subplots(figsize=(8, 5))
A = op.characteristic_speed.dat.data[0]
axes.plot(time_hours, A*np.sin(op.omega*time_seconds), color="C0")
axes.set_xlim([0.0, op.end_time/3600])
axes.set_ylim([-A, A])
axes.set_xlabel("Time [hours]")
axes.set_ylabel("Velocity [m/s]")
axes.grid(True)

# Add second x-axis with non-dimensionalised time
non_dimensionalise = lambda time: 3600*time/op.T_tide
dimensionalise = lambda time: 3600*time*op.T_tide
secax = axes.secondary_xaxis('top', functions=(non_dimensionalise, dimensionalise))
secax.set_xlabel("Time/Tidal period")

# Save plot
plot_dir = create_directory(os.path.join(os.path.dirname(__file__), "plots"))
savefig("tidal_velocity_forcing", plot_dir, extensions=["png", "pdf"])
