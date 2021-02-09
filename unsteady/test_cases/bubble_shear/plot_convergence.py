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
levels = 4
approaches = ['fixed_mesh']
alldofs = {'integrate': None, 'intersect': None}
dofs = {'integrate': None, 'intersect': None}
l2_error = {'integrate': None, 'intersect': None}
cons_error = {'integrate': None, 'intersect': None}
time = {'integrate': None, 'intersect': None}
average_dofs = {}
plot_dir = create_directory(os.path.join(os.path.dirname(__file__), 'plots'))

# Load fixed mesh data
di = os.path.join(os.path.dirname(__file__), 'outputs', 'fixed_mesh')
fname = os.path.join(di, "convergence.h5")
if not os.path.isfile(fname):
    print("Cannot find convergence data {:s}".format(fname))
with h5py.File(fname, 'r') as outfile:
    dofs['fixed_mesh'] = np.array(outfile['dofs'])
    time['fixed_mesh'] = np.array(outfile['time'])
    l2_error['fixed_mesh'] = np.array(outfile['l2_error'])
    cons_error['fixed_mesh'] = np.array(outfile['cons_error'])

# Load adaptive mesh data
concat = lambda a, b: b if a is None else np.concatenate((a, b))
for approach in characteristics:
    if approach == 'fixed_mesh':
        continue
    di = os.path.join(os.path.dirname(__file__), 'outputs', 'hessian', approach)
    fname = os.path.join(di, "convergence_{:d}.h5")
    approaches.append(approach)
    for i in range(levels):
        if not os.path.isfile(fname.format(i)):
            print("Cannot find convergence data {:s}".format(fname.format(i)))
            continue
        with h5py.File(fname.format(i), 'r') as outfile:
            alldofs[approach] = concat(alldofs[approach], [np.array(outfile['dofs'])])
            dofs[approach] = concat(dofs[approach], [np.average(outfile['dofs'][0])])
            time[approach] = concat(time[approach], np.array(outfile['time']))
            l2_error[approach] = concat(l2_error[approach], np.array(outfile['l2_error']))
            cons_error[approach] = concat(cons_error[approach], np.array(outfile['cons_error']))

# Plot DoF distribution
for approach in ('integrate', 'intersect'):
    fig, axes = plt.subplots()
    for i in [3, 2, 1, 0]:
        label = r'$\mathcal C_T={{{:.0f}}}$'.format(1000*2**i)
        axes.bar(np.linspace(1, 51, 50), alldofs[approach][i], label=label)
    axes.set_xlabel("Subinterval")
    axes.set_ylabel("DoF count")
    axes.set_xlim([0.5, 50.5])
    axes.set_xticks([1, 10, 20, 30, 40, 50])
    axes.grid(True, axis='y')
    savefig("dofs_{:s}".format(approach), plot_dir, extensions=["pdf"])
fig2, axes2 = plt.subplots()
lines, labels = axes.get_legend_handles_labels()
legend = axes2.legend(reversed(lines), reversed(labels), fontsize=18, frameon=False, ncol=4)
fig2.canvas.draw()
axes2.set_axis_off()
bbox = legend.get_window_extent().transformed(fig2.dpi_scale_trans.inverted())
savefig('legend_dofs', plot_dir, bbox_inches=bbox, extensions=['pdf'], tight=False)

# Plot L2 error
fig, axes = plt.subplots()
for approach in approaches:
    axes.semilogx(dofs[approach], l2_error[approach], **characteristics[approach])
axes.set_xlabel("Mean spatial DoFs")
axes.set_ylabel(r"Relative $\mathcal L_2$ error")
yticks = [20, 40, 60, 80]
axes.set_yticks(yticks)
axes.set_yticklabels([r"{{{:.0f}}}\%".format(yt) for yt in yticks])
axes.set_ylim([10, 80])
axes.grid(True, which='both')
savefig("l2_error", plot_dir, extensions=["pdf"])

# Plot conservation error
fig, axes = plt.subplots()
for approach in approaches:
    axes.semilogx(dofs[approach], cons_error[approach], **characteristics[approach])
axes.set_xlabel("Mean spatial DoFs")
axes.set_ylabel(r"$\mathcal L_1$ conservation error (\%)")
axes.grid(True)
savefig("cons_error", plot_dir, extensions=["pdf"])

# Plot CPU time
fig, axes = plt.subplots()
for approach in approaches:
    axes.loglog(dofs[approach], time[approach], **characteristics[approach])
axes.set_xlabel("Mean spatial DoFs")
axes.set_ylabel(r"CPU time [$\mathrm s$]")
yticks = [100, 1000, 10000]
axes.set_yticks(yticks)
axes.set_ylim([60, 10000])
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
