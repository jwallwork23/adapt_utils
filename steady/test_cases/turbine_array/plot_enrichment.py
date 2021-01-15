import matplotlib.pyplot as plt
from mpltools import annotation
import numpy as np
import os
import pickle

from adapt_utils.plotting import *


di = os.path.dirname(__file__)
input_dir = os.path.join(di, 'outputs/dwr/enrichment')
plot_dir = os.path.join(di, 'plots')

for nonlinear_method in ('prolong', 'solve'):
    for alignment in ('aligned', 'offset'):
        fname = os.path.join(input_dir, '{:s}_{:s}.p'.format(alignment, nonlinear_method))
        out = pickle.load(open(fname, 'rb'))

        out['GE_hp']['label'] = 'GE$_{hp}$'
        out['GE_h']['label'] = 'GE$_h$'
        out['GE_p']['label'] = 'GE$_p$'

        fig, axes = plt.subplots(figsize=(6, 5))
        for method in out.keys():
            time = out[method]['time']
            axes.loglog(out[method]['dofs'], time, '--x', label=out[method]['label'])
        axes.set_xlabel("Degrees of freedom")
        axes.set_ylabel(r"Computation time [$\mathrm s$]")
        axes.grid(True)
        if nonlinear_method == 'prolong':
            annotation.slope_marker((1.0e+05, 20), 1, invert=False, ax=axes, size_frac=0.2)
        savefig("enrichment_time_{:s}_{:s}".format(alignment, nonlinear_method), plot_dir, extensions=["pdf"])

        # Save legend to file
        if alignment == 'aligned' and nonlinear_method == 'prolong':
            fig2, axes2 = plt.subplots()
            lines, labels = axes.get_legend_handles_labels()
            lines = [lines[-1]] + lines[:-1]
            labels = [labels[-1]] + labels[:-1]
            legend = axes2.legend(lines, labels, fontsize=18, frameon=False, ncol=4)
            fig2.canvas.draw()
            axes2.set_axis_off()
            bbox = legend.get_window_extent().transformed(fig2.dpi_scale_trans.inverted())
            savefig('legend_enrichment', plot_dir, extensions=['pdf'], bbox_inches=bbox, tight=False)
