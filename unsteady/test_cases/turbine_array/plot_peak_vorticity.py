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
time = np.linspace(0, op.T_ramp/60, num_timesteps)

# Plotting parameters
resolutions = {
    24: {'colour': 'r'}, 12: {'colour': 'b'}, 6: {'colour': 'k'},
    5: {'colour': 'g'}, 4: {'colour': 'y'}, 3: {'colour': 'o'},
}
resolutions_used = []

# Plot peak vorticities
fig, axes = plt.subplots(figsize=(10, 5))
label = r"$\Delta x_{\mathrm{farm}} = %d \mathrm m$"
for dxfarm in resolutions:
    resolutions[dxfarm]['data'] = {}
    try:
        resolutions[dxfarm]['data']['min'] = np.load(fname.format('min', dxfarm))
        resolutions[dxfarm]['data']['max'] = np.load(fname.format('max', dxfarm))
        resolutions_used.append(dxfarm)
    except FileNotFoundError:
        print("Need to extract peak vorticity for dxfarm = {:d}".format(dxfarm))
        continue
    colour = resolutions[dxfarm]['colour']
    data = resolutions[dxfarm]['data']
    axes.plot(time, data['max'], color=colour, label=label % dxfarm, linewidth=1)
    axes.plot(time, data['min'], '-.', color=colour, linewidth=1)
axes.set_xlim([0, op.T_ramp/60])
axes.grid(True)
box = axes.get_position()
axes.set_position([box.x0, box.y0, box.width, box.height * 0.9])
axes.legend(loc='upper center', bbox_to_anchor=(0.5, 1.2), ncol=len(resolutions_used), fontsize=18)
axes.set_xlabel(r"Time [$\mathrm h$]")
axes.set_ylabel(r"Peak vorticity [$\mathrm s^{-1}$]")
savefig("plots/peak_vorticity")

# Plot over a single tidal cycle
axes.set_xlim([(op.T_ramp - op.T_tide)/60, op.T_ramp/60])
savefig("plots/peak_vorticity_cycle")

# Get non-dimensionalised time
time = np.linspace(0, op.T_ramp/op.T_tide, num_timesteps)

# Plot relative peak vorticities
fig, axes = plt.subplots(figsize=(10, 5))
for dxfarm in resolutions_used:
    colour = resolutions[dxfarm]['colour']
    data = resolutions[dxfarm]['data']
    axes.plot(time, data['max']/data['max'].max(), color=colour, label=label % dxfarm, linewidth=1)
    axes.plot(time, data['min']/np.abs(data['min']).max(), '-.', color=colour, linewidth=1)
axes.set_xlim([0, op.T_ramp/op.T_tide])
axes.set_xticks(np.linspace(0, 3.5, 8))
axes.set_yticks(np.linspace(-1, 1, 9))
axes.grid(True)
box = axes.get_position()
axes.set_position([box.x0, box.y0, box.width, box.height * 0.9])
axes.legend(loc='upper center', bbox_to_anchor=(0.5, 1.2), ncol=len(resolutions_used), fontsize=18)
axes.set_xlabel(r"Time/Tidal period")
axes.set_ylabel(r"Relative peak vorticity")
savefig("plots/relative_peak_vorticity")

# Plot over a single tidal cycle
axes.set_xlim([op.T_ramp/op.T_tide - 1, op.T_ramp/op.T_tide])
savefig("plots/relative_peak_vorticity_cycle")
