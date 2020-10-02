import matplotlib.pyplot as plt
import numpy as np
import os

from adapt_utils.plotting import *
from adapt_utils.unsteady.test_cases.turbine_array.options import TurbineArrayOptions


# Data
data_dir = os.path.join(os.path.dirname(__file__), 'data')
fname = os.path.join(data_dir, "vorticity_{:s}_{:d}.npy")

# Create parameters object
op = TurbineArrayOptions(1)
num_timesteps = int(op.T_ramp/op.dt/op.dt_per_export) + 1
time = np.linspace(0, op.T_ramp, num_timesteps)/60

# Plotting parameters
resolutions = [24, 12, 6, 3]
resolutions_used = []
colours = ['r', 'b', 'k', 'g']

# Plotting
fig, axes = plt.subplots(figsize=(10, 5))
for dxfarm, colour in zip(resolutions, colours):
    try:
        vorticity_min = np.load(fname.format('min', dxfarm))
        vorticity_max = np.load(fname.format('max', dxfarm))
        resolutions_used.append(dxfarm)
    except FileNotFoundError:
        print("Need to extract peak vorticity for dxfarm = {:d}".format(dxfarm))
        continue
    axes.plot(time, vorticity_max, color=colour, label=r"$\Delta x_{\mathrm{farm}} = %d$" % dxfarm)
    axes.plot(time, vorticity_min, '-.', color=colour)
axes.grid(True)
box = axes.get_position()
axes.set_position([box.x0, box.y0, box.width, box.height * 0.9])
axes.legend(loc='upper center', bbox_to_anchor=(0.5, 1.2), ncol=len(resolutions_used), fontsize=18)
axes.set_xlabel(r"Time [$\mathrm h$]")
axes.set_ylabel(r"Peak vorticity [$\mathrm s^{-1}$]")
savefig("plots/peak_viscosity")
