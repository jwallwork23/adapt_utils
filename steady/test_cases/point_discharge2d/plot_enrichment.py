import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import os
import pickle

from adapt_utils.plotting import *


di = os.path.dirname(__file__)
input_dir = os.path.join(di, 'outputs/dwr/enrichment')
plot_dir = os.path.join(di, 'plots')

methods = ('GE_hp', 'GE_h', 'GE_p', 'DQ')
markers = ('^', 'x', 'o', 's')
minor = ticker.LogLocator(base=10.0, subs=np.arange(1.0, 10.0)*0.1, numticks=10)
for alignment in ('aligned', 'offset'):
    fname = os.path.join(input_dir, '{:s}.p'.format(alignment))
    out = pickle.load(open(fname, 'rb'))

    out['GE_hp']['label'] = 'GE$_{hp}$'
    out['GE_h']['label'] = 'GE$_h$'
    out['GE_p']['label'] = 'GE$_p$'
    out['DQ']['label'] = 'DQ'

    # Plot effectivity
    fig, axes = plt.subplots(figsize=(6, 5))
    for method, marker in zip(methods, markers):
        if method == 'DQ':
            continue
        I_eff = np.array(out[method]['effectivity'][1:])
        axes.plot(out[method]['dofs'][1:], I_eff, '--', label=out[method]['label'], marker=marker)
    axes.set_xscale('log')
    axes.set_yscale('log')
    for ax in (axes.xaxis, axes.yaxis):
        ax.set_minor_locator(minor)
        ax.set_minor_formatter(ticker.NullFormatter())
    axes.set_xticks([10**i for i in range(4, 7)])
    axes.set_yticks([10**i for i in range(-1, 3)])
    axes.set_xlabel("Degrees of freedom")
    axes.set_ylabel("Effectivity index")
    axes.grid(True, which='both')
    savefig("enrichment_effectivity_{:s}".format(alignment), plot_dir, extensions=["pdf"])

    # Get average time for an adjoint solve
    time_adj = sum(np.array(out[method]['time_adj']) for method in out.keys())/len(methods)

    # Plot CPU time
    fig, axes = plt.subplots(figsize=(6, 5))
    for method, marker in zip(methods, markers):
        time = out[method]['time'][1:]
        axes.plot(out[method]['dofs'][1:], time, '--', label=out[method]['label'], marker=marker)
    axes.set_xscale('log')
    axes.set_yscale('log')
    for ax in (axes.xaxis, axes.yaxis):
        ax.set_minor_locator(minor)
        ax.set_minor_formatter(ticker.NullFormatter())
    axes.set_xlabel("Degrees of freedom")
    axes.set_ylabel(r"CPU time [$\mathrm s$]")
    axes.set_xticks([10**i for i in range(4, 7)])
    axes.set_yticks([10**i for i in range(5)])
    axes.grid(True, which='both')
    # annotation.slope_marker((1.5e+05, 1.5), 1, invert=False, ax=axes, size_frac=0.2)
    # annotation.slope_marker((2.0e+05, 1.0e+03), 1.5, invert=True, ax=axes, size_frac=0.2)
    savefig("enrichment_time_{:s}".format(alignment), plot_dir, extensions=["pdf"])

    # Plot relative CPU time
    fig, axes = plt.subplots(figsize=(6, 5))
    for method, marker in zip(methods, markers):
        time = np.array(out[method]['time'][1:])/time_adj[1:]
        axes.plot(out[method]['dofs'][1:], time, '--', label=out[method]['label'], marker=marker)
    axes.set_xscale('log')
    axes.set_yscale('log')
    for ax in (axes.xaxis, ):
        ax.set_minor_locator(minor)
        ax.set_minor_formatter(ticker.NullFormatter())
    axes.set_xlabel("Degrees of freedom")
    axes.set_ylabel(r"CPU time / adjoint solve")
    axes.set_xticks([10**i for i in range(4, 7)])
    axes.grid(True, which='both')
    savefig("relative_enrichment_time_{:s}".format(alignment), plot_dir, extensions=["pdf"])

    # Save legend to file
    if alignment == 'aligned':
        fig2, axes2 = plt.subplots()
        lines, labels = axes.get_legend_handles_labels()
        legend = axes2.legend(lines, labels, fontsize=18, frameon=False, ncol=4)
        fig2.canvas.draw()
        axes2.set_axis_off()
        bbox = legend.get_window_extent().transformed(fig2.dpi_scale_trans.inverted())
        savefig('legend_enrichment', plot_dir, extensions=['pdf'], bbox_inches=bbox, tight=False)
