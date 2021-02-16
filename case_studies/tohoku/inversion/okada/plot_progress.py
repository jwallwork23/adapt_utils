import argparse
import matplotlib.pyplot as plt
import numpy as np
import sys

from adapt_utils.norms import vecnorm
from adapt_utils.plotting import *  # NOQA


parser = argparse.ArgumentParser()
parser.add_argument("level")
args = parser.parse_args()

level = int(args.level)
mode = 'discrete'
try:
    control_trajectory = np.load('data/opt_progress_{:s}_{:d}_ctrl.npy'.format(mode, level))
    functional_trajectory = np.load('data/opt_progress_{:s}_{:d}_func.npy'.format(mode, level))
    gradient_trajectory = np.load('data/opt_progress_{:s}_{:d}_grad.npy'.format(mode, level))
    line_search_trajectory = np.load('data/opt_progress_{:s}_{:d}_ls.npy'.format(mode, level))
except Exception:
    print("Cannot load {:s} data for level {:d}.".format(mode, level))
    sys.exit(0)
i = 0
indices = [0]
for j, ctrl in enumerate(control_trajectory):
    if i == len(line_search_trajectory):
        break
    if np.allclose(ctrl, line_search_trajectory[i]):
        indices.append(j)
        i += 1
functional_trajectory = [functional_trajectory[i] for i in indices]
gradient_trajectory = [gradient_trajectory[i] for i in indices]
gradient_norm_trajectory = [vecnorm(g, order=np.Inf) for g in gradient_trajectory]

fig, axes = plt.subplots(figsize=(8, 8))
axes.loglog(functional_trajectory)
axes.set_xlabel("Iteration")
axes.set_ylabel("Quantity of Interest")
axes.grid(True, which='both')
plt.tight_layout()

fig, axes = plt.subplots(figsize=(8, 8))
axes.loglog(gradient_norm_trajectory)
axes.set_xlabel("Iteration")
axes.set_ylabel("Computed gradient")
axes.grid(True, which='both')
plt.tight_layout()

plt.show()
