import matplotlib.pyplot as plt
import numpy as np
import os

from adapt_utils.case_studies.tohoku.options.options import TohokuOptions
from adapt_utils.plotting import *


# Plotting parameters
fontsize = 20
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
for smoothed in (True, False):
    for gauge in gauges:
        sample = 60 if smoothed and (gauge[0] == 'P' or 'PG' in gauge) else 1
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
        ax.legend(loc='best', fontsize=fontsize_tick)
        ax.set_xlabel(r'Time [$\mathrm{min}$]', fontsize=fontsize)
        ax.set_ylabel(r'Elevation [$\mathrm m$]', fontsize=fontsize)
        t0 = op.gauges[gauge]["arrival_time"]/60
        tf = op.gauges[gauge]["departure_time"]/60
        ax.set_xlim([t0, tf])
        ax.tick_params(axis='x', labelsize=fontsize_tick)
        ax.tick_params(axis='y', labelsize=fontsize_tick)
        ax.grid()
    for i in range(num_gauges, N*N):
        axes[i//N, i % N].axis(False)
    plt.tight_layout()
    fname = os.path.join(di, 'gauge_data')
    if smoothed:
        fname = fname + '_smoothed'
    plt.savefig(fname + '.pdf')
    plt.savefig(fname + '.png')
