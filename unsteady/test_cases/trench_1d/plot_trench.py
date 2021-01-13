import matplotlib.pyplot as plt
import numpy as np
import os

from adapt_utils.io import create_directory
from adapt_utils.plotting import *


depth_init = 0.397
depth_riv = depth_init - 0.397
depth_trench = depth_riv - 0.15
depth_diff = depth_trench - depth_riv

N = 1000
X = np.linspace(0.0, 16.0, N)


def trench(x):
    if x <= 5.0:
        return depth_riv
    elif x <= 6.5:
        return depth_diff*(x - 6.5)/1.5 + depth_trench
    elif x <= 9.5:
        return depth_trench
    elif x <= 11.0:
        return depth_diff*(11.0 - x)/1.5 + depth_riv
    else:
        return depth_riv


fig, axes = plt.subplots(figsize=(10, 3))
axes.plot(X, [trench(Xi) for Xi in X])
yticks = [-0.15, -0.1, -0.05, 0]
yticklabels = ["0.15", "0.10", "0.05", "0.00"]
axes.set_xticks([0, 5, 6.5, 9.5, 11, 16])
axes.set_yticks(yticks)
axes.set_yticklabels(yticklabels)
axes.set_xlabel(r"$x$-coordinate [$\mathrm m$]")
axes.set_ylabel(r"Depth [$\mathrm m$]")
axes.set_xlim([0, 16])
plot_dir = create_directory(os.path.join(os.path.dirname(__file__), 'plots'))
savefig('trench', plot_dir, extensions=['pdf'])
