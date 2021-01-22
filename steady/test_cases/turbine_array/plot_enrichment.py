import matplotlib.pyplot as plt
from mpltools import annotation
import os
import pickle

from adapt_utils.plotting import *


di = os.path.dirname(__file__)
input_dir = os.path.join(di, 'outputs/dwr/enrichment')
plot_dir = os.path.join(di, 'plots')

for alignment in ('aligned', 'offset'):

    # Plot computation time
    fig, axes = plt.subplots(figsize=(6, 5))
    for nonlinear_method in ('prolong', 'solve'):
        linestyle = '--' if nonlinear_method == 'prolong' else ':'
        fname = os.path.join(input_dir, '{:s}_{:s}.p'.format(alignment, nonlinear_method))
        out = pickle.load(open(fname, 'rb'))
        out['GE_hp']['label'] = 'GE$_{hp}$ (' + nonlinear_method + ')'
        out['GE_h']['label'] = 'GE$_h$ (' + nonlinear_method + ')'
        out['GE_p']['label'] = 'GE$_p$ (' + nonlinear_method + ')'
        out['GE_hp']['colour'] = 'C0'
        out['GE_h']['colour'] = 'C1'
        out['GE_p']['colour'] = 'C2'
        for method in out.keys():
            time = out[method]['time']
            kwargs = dict(marker='x', color=out[method]['colour'], label=out[method]['label'])
            axes.loglog(out[method]['dofs'], time, linestyle, **kwargs)
    axes.set_xlabel("Degrees of freedom")
    axes.set_ylabel(r"Computation time [$\mathrm s$]")
    axes.grid(True)
    annotation.slope_marker((1.0e+05, 10), 1, invert=False, ax=axes, size_frac=0.2)
    savefig("enrichment_time_{:s}".format(alignment), plot_dir, extensions=["pdf"])

    # Save legend to file
    if alignment == 'aligned':
        fig2, axes2 = plt.subplots()
        lines, labels = axes.get_legend_handles_labels()
        # lines = [lines[-1]] + lines[:-1]
        # labels = [labels[-1]] + labels[:-1]
        legend = axes2.legend(lines, labels, fontsize=18, frameon=False, ncol=1)
        fig2.canvas.draw()
        axes2.set_axis_off()
        bbox = legend.get_window_extent().transformed(fig2.dpi_scale_trans.inverted())
        savefig('legend_enrichment', plot_dir, extensions=['pdf'], bbox_inches=bbox, tight=False)
