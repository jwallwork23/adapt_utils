import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from mpltools import annotation
import numpy as np
import os
import pickle

from adapt_utils.plotting import *


di = os.path.dirname(__file__)
input_dir = os.path.join(di, 'outputs/dwr/enrichment')
plot_dir = os.path.join(di, 'plots')

for alignment in ('aligned', 'offset'):

    # Plot computation time
    fig, axes = plt.subplots(figsize=(6, 5))
    for nonlinear_method, linestyle in zip(('prolong', 'solve'), ('--', ':')):
        fname = os.path.join(input_dir, '{:s}_{:s}.p'.format(alignment, nonlinear_method))
        out = pickle.load(open(fname, 'rb'))
        out['GE_hp']['label'] = 'GE$_{hp}$ (' + nonlinear_method + ')'
        out['GE_h']['label'] = 'GE$_h$ (' + nonlinear_method + ')'
        out['GE_p']['label'] = 'GE$_p$ (' + nonlinear_method + ')'
        out['GE_hp']['colour'] = 'C0'
        out['GE_h']['colour'] = 'C1'
        out['GE_p']['colour'] = 'C2'
        for method, marker in zip(out.keys(), ('^', 's', 'o')):
            time = out[method]['time']
            kwargs = dict(marker=marker, color=out[method]['colour'], label=out[method]['label'])
            axes.plot(out[method]['dofs'], time, linestyle, **kwargs)
    axes.set_xscale('log')
    axes.set_yscale('log')
    minor = ticker.LogLocator(base=10.0, subs=np.arange(1.0, 10.0)*0.1, numticks=10)
    axes.xaxis.set_minor_locator(minor)
    axes.xaxis.set_minor_formatter(ticker.NullFormatter())
    axes.yaxis.set_minor_locator(minor)
    axes.yaxis.set_minor_formatter(ticker.NullFormatter())
    axes.set_xlabel("Degrees of freedom")
    axes.set_ylabel(r"CPU time [$\mathrm s$]")
    axes.grid(True, which='both')
    annotation.slope_marker((1.0e+05, 10), 1, invert=False, ax=axes, size_frac=0.2)
    savefig("enrichment_time_{:s}".format(alignment), plot_dir, extensions=["pdf"])

    # Save legend to file
    if alignment == 'aligned':
        fig2, axes2 = plt.subplots()
        lines, labels = axes.get_legend_handles_labels()
        legend = axes2.legend(lines, labels, fontsize=18, frameon=False, ncol=1)
        fig2.canvas.draw()
        axes2.set_axis_off()
        bbox = legend.get_window_extent().transformed(fig2.dpi_scale_trans.inverted())
        savefig('legend_enrichment', plot_dir, extensions=['pdf'], bbox_inches=bbox, tight=False)

    # Plot computation time
    baseline = None
    fig, axes = plt.subplots(figsize=(6, 5))
    for nonlinear_method, linestyle in zip(('prolong', 'solve'), ('--', ':')):
        fname = os.path.join(input_dir, '{:s}_{:s}.p'.format(alignment, nonlinear_method))
        out = pickle.load(open(fname, 'rb'))
        baseline = baseline if baseline is not None else np.array(out['GE_p']['time'])
        out['GE_hp']['label'] = 'GE$_{hp}$ (' + nonlinear_method + ')'
        out['GE_h']['label'] = 'GE$_h$ (' + nonlinear_method + ')'
        out['GE_p']['label'] = 'GE$_p$ (' + nonlinear_method + ')'
        out['GE_hp']['colour'] = 'C0'
        out['GE_h']['colour'] = 'C1'
        out['GE_p']['colour'] = 'C2'
        for method, marker in zip(out.keys(), ('^', 's', 'o')):
            time = np.array(out[method]['time'])/baseline
            kwargs = dict(marker=marker, color=out[method]['colour'], label=out[method]['label'])
            axes.plot(out[method]['dofs'], time, linestyle, **kwargs)
    axes.set_xscale('log')
    minor = ticker.LogLocator(base=10.0, subs=np.arange(1.0, 10.0)*0.1, numticks=10)
    axes.xaxis.set_minor_locator(minor)
    axes.xaxis.set_minor_formatter(ticker.NullFormatter())
    axes.set_yticks([1, 5, 10, 15])
    axes.set_ylim([0, 15])
    axes.yaxis.set_minor_locator(ticker.LinearLocator(numticks=16))
    axes.set_xlabel("Degrees of freedom")
    axes.set_ylabel(r"CPU time / GE$_p$ (prolong)")
    axes.grid(True, which='both')
    savefig("relative_enrichment_time_{:s}".format(alignment), plot_dir, extensions=["pdf"])
