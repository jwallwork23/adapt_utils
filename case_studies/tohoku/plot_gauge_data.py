import matplotlib.pyplot as plt
import numpy as np
import os

from adapt_utils.case_studies.tohoku.options.options import TohokuOptions


# Plotting parameters
fontsize = 22
fontsize_tick = 18
plotting_kwargs = {
    'markevery': 5,
}

# Create output directories
dirname = os.path.dirname(__file__)
di = os.path.join(dirname, 'outputs')
if not os.path.exists(di):
    os.makedirs(di)

# Instantiate TohokuOptions object and setup interpolator
op = TohokuOptions()
gauges = list(op.gauges)
num_gauges = len(gauges)
for gauge in gauges:
    sample = 1 if gauge[0] in ('2', '8') else 60
    op.sample_timeseries(gauge, sample=sample)
    op.gauges[gauge]["data"] = []

# Interpolate timeseries data from file
t = 0.0
t_epsilon = 1.0e-05
time_seconds = []
while t < op.end_time - t_epsilon:
    time_seconds.append(t)
    for gauge in gauges:
        op.gauges[gauge]["data"].append(float(op.gauges[gauge]["interpolator"](t)))
    t += op.dt

# Convert time to minutes
time_seconds = np.array(time_seconds)
time_minutes = time_seconds/60.0

# Plot timeseries data
N = int(np.ceil(np.sqrt(num_gauges)))
fig, axes = plt.subplots(nrows=N, ncols=N, figsize=(16, 12))
for i, gauge in enumerate(gauges):
    ax = axes[i//N, i % N]
    ax.plot(time_minutes, op.gauges[gauge]['data'], '--x', label=gauge, **plotting_kwargs)
    ax.legend(loc='upper left')
    ax.set_xlabel('Time (min)', fontsize=fontsize)
    ax.set_ylabel('Elevation (m)', fontsize=fontsize)
    plt.xticks(fontsize=fontsize_tick)
    plt.yticks(fontsize=fontsize_tick)
    ax.grid()
for i in range(num_gauges, N*N):
    axes[i//N, i % N].axis(False)
plt.tight_layout()
plt.savefig(os.path.join(di, 'gauge_data.pdf'))
