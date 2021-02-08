import h5py
import matplotlib.pyplot as plt
import numpy as np
import os

from adapt_utils.io import create_directory
from adapt_utils.plotting import *


characteristics = {
    "fixed_mesh": {"label": "Uniform refinement", "marker": "*"},
    "integrate": {"label": "Integration", "marker": "v"},
    "intersect": {"label": "Intersection", "marker": "^"},
}
approaches = []
dofs = {}
average_dofs = {}
l2_error = {}
cons_error = {}
time = {}
plot_dir = create_directory(os.path.join(os.path.dirname(__file__), 'plots'))
for approach in characteristics:
    di = os.path.join(os.path.dirname(__file__), 'outputs')
    if approach in ('integrate', 'intersect'):
        di = os.path.join(di, 'hessian')
    di = os.path.join(di, approach)
    fname = os.path.join(di, "convergence.h5")
    if not os.path.isfile(fname):
        print("Cannot find convergence data {:s}".format(fname))
        continue
    approaches.append(approach)
    with h5py.File(fname, 'r') as outfile:
        dofs[approach] = np.array(outfile['dofs'])
        if approach != 'fixed_mesh':
            average_dofs[approach] = np.average(dofs[approach], axis=1)
        time[approach] = np.array(outfile['time'])
        l2_error[approach] = np.array(outfile['l2_error'])
        cons_error[approach] = np.array(outfile['cons_error'])

# Plot L2 error
fig, axes = plt.subplots()
for approach in approaches:
    if approach == 'fixed_mesh':
        axes.semilogx(dofs[approach], l2_error[approach], **characteristics[approach])
    else:
        axes.semilogx(average_dofs[approach], l2_error[approach], **characteristics[approach])
axes.set_xlabel("Mean spatial DoFs")
axes.set_ylabel(r"Relative $\mathcal L_2$ error (\%)")
axes.grid(True)
savefig("l2_error", plot_dir, extensions=["pdf"])

# Plot conservation error
fig, axes = plt.subplots()
for approach in approaches:
    if approach == 'fixed_mesh':
        axes.semilogx(dofs[approach], cons_error[approach], **characteristics[approach])
    else:
        axes.semilogx(average_dofs[approach], cons_error[approach], **characteristics[approach])
axes.set_xlabel("Mean spatial DoFs")
axes.set_ylabel(r"$\mathcal L_1$ conservation error (\%)")
axes.grid(True)
savefig("cons_error", plot_dir, extensions=["pdf"])

# Plot CPU time
fig, axes = plt.subplots()
for approach in approaches:
    if approach == 'fixed_mesh':
        axes.loglog(dofs[approach], time[approach], **characteristics[approach])
    else:
        axes.loglog(average_dofs[approach], time[approach], **characteristics[approach])
axes.set_xlabel("Mean spatial DoFs")
axes.set_ylabel(r"CPU time [$\mathrm s$]")
axes.grid(True, which='both')
savefig("time", plot_dir, extensions=["pdf"])

# Plot legend
fig2, axes2 = plt.subplots()
lines, labels = axes.get_legend_handles_labels()
legend = axes2.legend(lines, labels, fontsize=18, frameon=False, ncol=3)
fig2.canvas.draw()
axes2.set_axis_off()
bbox = legend.get_window_extent().transformed(fig2.dpi_scale_trans.inverted())
savefig('legend', plot_dir, bbox_inches=bbox, extensions=['pdf'], tight=False)
