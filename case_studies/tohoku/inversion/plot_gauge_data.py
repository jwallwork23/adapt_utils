from thetis import create_directory

import matplotlib.pyplot as plt
import numpy as np
import os

from adapt_utils.case_studies.tohoku.options.options import TohokuInversionOptions
from adapt_utils.plotting import *  # NOQA


# Plotting parameters
fontsize = 22
fontsize_tick = 20
plotting_kwargs = {
    'markevery': 5,
}

# Create output directories
dirname = os.path.dirname(__file__)
di = create_directory(os.path.join(dirname, 'plots'))

# Instantiate TohokuOptions object and setup interpolator
op = TohokuInversionOptions()
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
    fig, axes = plt.subplots(nrows=N, ncols=N, figsize=(17, 13))
    for i, gauge in enumerate(gauges):
        ax = axes[i//N, i % N]
        plotting_kwargs['color'] = op.gauges[gauge]['colour']
        ax.plot(time_minutes, op.gauges[gauge]['data'], '-', label=gauge, **plotting_kwargs)
        leg = ax.legend(handlelength=0, handletextpad=0, fontsize=fontsize_tick)
        for item in leg.legendHandles:
            item.set_visible(False)
        if i//N == 3:
            ax.set_xlabel(r'Time [$\mathrm{min}$]', fontsize=fontsize)
        if i % N == 0:
            ax.set_ylabel(r'Elevation [$\mathrm m$]', fontsize=fontsize)
        ax.xaxis.set_tick_params(labelsize=fontsize_tick)
        ax.yaxis.set_tick_params(labelsize=fontsize_tick)
        ax.set_yticks(ax.get_yticks().tolist())  # Avoid matplotlib error
        ax.set_yticklabels(["{:.1f}".format(tick) for tick in ax.get_yticks()])
        t0 = op.gauges[gauge]["arrival_time"]/60
        tf = op.gauges[gauge]["departure_time"]/60
        ax.set_xlim([t0, tf])
        ax.grid()
    for i in range(num_gauges, N*N):
        axes[i//N, i % N].axis(False)
    plt.tight_layout()
    fname = 'gauge_data'
    if smoothed:
        fname = fname + '_smoothed'
    savefig(fname, di, extensions=['png', 'pdf'])
