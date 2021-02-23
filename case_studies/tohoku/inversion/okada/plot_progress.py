import argparse
import matplotlib.pyplot as plt
import numpy as np

from adapt_utils.norms import vecnorm
from adapt_utils.plotting import *  # NOQA


parser = argparse.ArgumentParser()
parser.add_argument("level")
parser.add_argument("-categories")
args = parser.parse_args()

level = int(args.level)
gauge_classifications = (
    'all',
    'near_field_gps',
    'near_field_pressure',
    'mid_field_pressure',
    'far_field_pressure',
    'southern_pressure',
)
if 'all' in args.categories:
    categories = 'all'
    gauge_classifications_to_consider = gauge_classifications[1:]
else:
    categories = args.categories.split(',')
    gauge_classifications_to_consider = []
    for category in categories:
        assert category in gauge_classifications
        gauge_classifications_to_consider.append(category)
    categories = '_'.join(categories)
fname = 'data/opt_progress_discrete_{:d}_{:s}'.format(level, categories) + '_{:s}'
print(fname.format('ctrl') + '.npy')
control_trajectory = np.load(fname.format('ctrl') + '.npy')
functional_trajectory = np.load(fname.format('func') + '.npy')
gradient_trajectory = np.load(fname.format('grad') + '.npy')
line_search_trajectory = np.load(fname.format('ls') + '.npy')
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
