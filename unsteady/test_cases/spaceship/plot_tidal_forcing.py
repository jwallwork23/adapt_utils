from thetis import create_directory

import matplotlib.pyplot as plt
import numpy as np
import os

from adapt_utils.unsteady.test_cases.spaceship.options import SpaceshipOptions


plt.rc('text', usetex=True)

# Interpolate forcing onto time range
op = SpaceshipOptions()
time_seconds = np.linspace(0.0, op.tidal_forcing_end_time, 1001)
time_hours = time_seconds/3600
time_days = time_hours/24
forcing = op.tidal_forcing_interpolator(time_seconds)

# Plot the spin-up period only
fig, axes = plt.subplots(figsize=(8, 6))
axes.plot(time_hours, forcing)
axes.set_xlabel(r"Time $[h]$")
axes.set_ylabel(r"Tidal forcing $[m]$")
axes.set_xlim([0, op.T_ramp/3600])
axes.set_xticks([0, 6, 12, 18, 24])
plot_dir = create_directory(os.path.join(os.path.dirname(__file__), 'plots'))
plt.tight_layout()
for ext in ("png", "pdf"):
    plt.savefig(os.path.join(plot_dir, ".".join(["spin_up", ext])))

# Plot the forcing over the whole time range
fig, axes = plt.subplots(figsize=(18, 6))
axes.plot(time_days, forcing)
axes.set_xlabel("Time [days]")
axes.set_ylabel("Tidal forcing [m]")

# Annotate the spin-up period
axes.set_xticks(range(17))
ylim = axes.get_ylim()
r = op.T_ramp/3600/24
axes.axvline(r, *ylim, linestyle='--', color='k')
axes.set_xlim([0, 16])
axes.set_ylim(ylim)
plotting_kwargs = {
    "arrowprops": {
        "arrowstyle": "<->",
    },
}
axes.annotate("", xy=(0.0, -3.7), xytext=(r, -3.7), annotation_clip=False, **plotting_kwargs)
axes.annotate("Spin-up", xy=(0.2*r, -3.95), xytext=(0.2*r, -3.95), annotation_clip=False)
axes.annotate("period", xy=(0.2*r, -4.25), xytext=(0.2*r, -4.25), annotation_clip=False)

# Add second x-axis with non-dimensionalised time
non_dimensionalise = lambda time: 24*3600*time/op.T_tide
dimensionalise = lambda time: 24*3600*time*op.T_tide
secax = axes.secondary_xaxis('top', functions=(non_dimensionalise, dimensionalise))
secax.set_xlabel("Time/Tidal period")
secax.set_xticks(range(32))

# Save plot
plt.tight_layout()
for ext in ("png", "pdf"):
    plt.savefig(os.path.join(plot_dir, ".".join(["tidal_forcing", ext])))