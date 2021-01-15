import matplotlib.pyplot as plt
from mpltools import annotation
import numpy as np
import os
import pickle

from adapt_utils.plotting import *


di = os.path.dirname(__file__)
input_dir = os.path.join(di, 'outputs/dwr/enrichment')
plot_dir = os.path.join(di, 'plots')

for alignment in ('aligned', 'offset'):
    fname = os.path.join(input_dir, '{:s}.p'.format(alignment))
    out = pickle.load(open(fname, 'rb'))

    out['GE_hp']['label'] = 'GE$_{hp}$'
    out['GE_h']['label'] = 'GE$_h$'
    out['GE_p']['label'] = 'GE$_p$'
    out['DQ']['label'] = 'DQ'

    fig, axes = plt.subplots(figsize=(6, 5))
    for method in out.keys():
        time = out[method]['time'][1:]
        # axes.loglog(out[method]['num_cells'][1:], time, '--x', label=out[method]['label'])
        axes.loglog(out[method]['dofs'][1:], time, '--x', label=out[method]['label'])
    axes.set_xlabel("Degrees of freedom")
    axes.set_ylabel(r"Computation time [$\mathrm s$]")
    axes.grid(True)
    annotation.slope_marker((3.0e+05, 1), 1, invert=False, ax=axes, size_frac=0.2)
    annotation.slope_marker((1.0e+06, 1.0e+04), 1.5, invert=True, ax=axes, size_frac=0.2)
    savefig("enrichment_time_{:s}".format(alignment), plot_dir, extensions=["pdf"])

    fig, axes = plt.subplots(figsize=(6, 5))
    for method in out.keys():
        I_eff = np.array(out[method]['effectivity'][1:])
        # axes.semilogx(out[method]['num_cells'], I_eff, '--x', label=out[method]['label'])
        # axes.loglog(out[method]['num_cells'][1:], I_eff, '--x', label=out[method]['label'])
        axes.loglog(out[method]['dofs'][1:], I_eff, '--x', label=out[method]['label'])
    axes.set_xlabel("Degrees of freedom")
    axes.set_ylabel("Effectivity index")
    axes.grid(True)
    annotation.slope_marker((2.0e+05, 4), 0.5, invert=False, ax=axes, size_frac=0.2)
    annotation.slope_marker((2.0e+05, 2.0e+04), 1.5, invert=False, ax=axes, size_frac=0.2)
    savefig("enrichment_effectivity_{:s}".format(alignment), plot_dir, extensions=["pdf"])

    # Save legend to file
    if alignment == 'aligned':
        fig2, axes2 = plt.subplots()
        lines, labels = axes.get_legend_handles_labels()
        lines = [lines[-1]] + lines[:-1]
        labels = [labels[-1]] + labels[:-1]
        legend = axes2.legend(lines, labels, fontsize=18, frameon=False, ncol=4)
        fig2.canvas.draw()
        axes2.set_axis_off()
        bbox = legend.get_window_extent().transformed(fig2.dpi_scale_trans.inverted())
        savefig('legend_enrichment', plot_dir, extensions=['pdf'], bbox_inches=bbox, tight=False)
