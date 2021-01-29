import argparse
import matplotlib.pyplot as plt
import numpy as np
import os

from adapt_utils.plotting import *


# --- Parse arguments

parser = argparse.ArgumentParser()
parser.add_argument("-approach", help="Choose adaptive approach")
parser.add_argument("-geometry", help="Choose from 'circle' or 'square'")
args = parser.parse_args()


# --- Set parameters

# Parsed arguments
geometry = args.geometry or 'circle'
approach = args.approach or 'fixed_mesh'

# Plotting parameters
fontsize = 18
legend_fontsize = 16
kwargs = {'linestyle': '--', 'marker': 'x'}

# Shapes to consider
shapes = ('Gaussian', 'Cone', 'Slotted Cylinder')


# --- Load data from log files

elements = []
dat = {}
for shape in shapes:
    dat[shape] = {'qois': [], 'errors': []}
for n in (1, 2, 4, 8, 16):
    with open(os.path.join('outputs', approach, '{:s}_{:d}.log'.format(geometry, n)), 'r') as f:
        f.readline()
        f.readline()
        f.readline()
        for shape in shapes:
            line = f.readline().split()
            approx = float(line[-3])
            exact = float(line[-5])
            dat[shape]['qois'].append(approx)
            dat[shape]['errors'].append(abs(1.0 - approx/exact))
        f.readline()
        elements.append(int(f.readline().split()[-1]))


# --- Plot relative errors

fig, ax = plt.subplots(figsize=(6, 5))
for shape in shapes:
    ax.loglog(elements, dat[shape]['errors'], label=shape.replace(' ', '\n'), **kwargs)
ax.set_xlabel('Element Count', fontsize=fontsize)
ax.set_ylabel('Relative Error', fontsize=fontsize)
ax.set_xticks([1.0e+04, 1.0e+05, 1.0e+06])
yticks = [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1]
ax.set_yticks(yticks)
ax.set_ylim([yticks[0], yticks[-1]])
ytick_labels = [int(np.log10(yt)) for yt in yticks]
ax.set_yticklabels([r"$10^{{{:d}}}$".format(l) for l in ytick_labels])
ax.grid(True)
ax.legend(fontsize=legend_fontsize)
fname = 'convergence_' + geometry
savefig(fname, os.path.join('outputs', approach), extensions=['pdf', 'png'])
