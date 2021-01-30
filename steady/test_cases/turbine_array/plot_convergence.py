import argparse
import h5py
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import numpy as np
import os

from adapt_utils.plotting import *


# --- Parse arguments

parser = argparse.ArgumentParser()
parser.add_argument("-loglog")
parser.add_argument("-errorline")
parser.add_argument("-enrichment_method")
args = parser.parse_args()


# --- Set parameters

# Formatting
fontsize = 18
fontsize_legend = 16
loglog = bool(args.loglog)
xlabel = "Degrees of freedom"
ylabel = r"Power output $(\mathrm{MW})$"
ylabel2 = r"Relative error (\%)"
errorline = float(args.errorline or 0.0)

# Curve plotting kwargs
characteristics = {
    'fixed_mesh': {'label': 'Uniform refinement', 'marker': 'x', 'color': 'cornflowerblue'},
    'isotropic_dwr': {'label': 'Isotropic DWR', 'marker': '^', 'color': 'orange'},
    'anisotropic_dwr': {'label': 'Anisotropic DWR', 'marker': 'o', 'color': 'g'},
    'weighted_hessian': {'label': 'Weighted Hessian', 'marker': 's', 'color': 'magenta'},
}
for approach in list(characteristics.keys()):
    characteristics[approach]['linestyle'] = '-'
approaches = ['isotropic_dwr', 'anisotropic_dwr', 'weighted_hessian']
plot_dir = os.path.join(os.path.dirname(__file__), 'plots')
method = args.enrichment_method or 'GE_h'

# --- Plot convergence curves

di = os.path.join(os.path.dirname(__file__), 'outputs')
for offset in (0, 1):
    fig, axes = plt.subplots()
    axes.grid(True)

    # Read converged QoI value from file
    fname = os.path.join(di, 'fixed_mesh', 'hdf5', 'qoi_offset_{:d}.h5'.format(offset))
    with h5py.File(fname, 'r') as f:
        exact = np.array(f['qoi'])[-1]

    # Conversion functions
    power2error = lambda x: 100*abs(x - exact)/exact
    error2power = lambda x: x*exact/100 + exact

    # Plot fixed mesh
    approach = 'fixed_mesh'
    fname = os.path.join(di, approach, 'hdf5', 'qoi_offset_{:d}.h5'.format(offset))
    with h5py.File(fname, 'r') as f:
        dofs, qois = np.array(f['dofs']), np.array(f['qoi'])
    if loglog:
        axes.loglog(np.array(dofs[:-1]), power2error(np.array(qois[:-1])), **characteristics[approach])
    else:
        axes.semilogx(dofs, qois, **characteristics[approach])

    # Plot convergence curves
    for approach in approaches:
        for forward in ('prolong', 'solve'):

            # Read data from file
            fname = os.path.join(di, approach, 'hdf5', 'qoi_offset_{:d}_{:s}'.format(offset, method))
            if method not in ('PR', 'DQ'):
                fname += '_{:s}'.format(forward)
            fname += '.h5'
            if not os.path.exists(fname):
                print("File {:s} does not exist.".format(fname))
                continue
            with h5py.File(fname, 'r') as f:
                dofs, qois = np.array(f['dofs']), np.array(f['qoi'])

            # Plot convergence curves
            chars = characteristics[approach].copy()
            if method not in ('PR', 'DQ'):
                chars['label'] += " ({:s})".format(forward)
                if forward == 'solve':
                    chars['linestyle'] = '--'
            if loglog:
                axes.loglog(dofs, power2error(qois), **chars)
            else:
                axes.semilogx(dofs, qois, **chars)

    if loglog:
        yticks = [0.001, 0.01, 0.1, 1]
        axes.set_yticks(yticks)
        axes.set_yticklabels(["0.001", "0.01", "0.1", "1"])
    else:
        if offset == 0:
            axes.set_ylim([21.80, 22.03])

        # Annotate with an error line
        xticks = [1.0e+04, 1.0e+05, 1.0e+06, 1.0e+07]
        hlines = [exact]
        if errorline > 1e-3:
            hlines.append((1.0 + errorline/100)*exact)
        label = r'{:.1f}\% relative error'.format(errorline)
        plt.hlines(hlines, xticks[0], xticks[-1], color='k', linestyles='dashed', label=label)
        axes.set_xticks(xticks)
        axes.set_xlim([xticks[0], xticks[-1]])
        axes.yaxis.set_minor_locator(AutoMinorLocator())

        # Add a secondary axis
        secax = axes.secondary_yaxis('right', functions=(power2error, error2power))
        secax.set_ylabel(ylabel2, fontsize=fontsize)
        secax.yaxis.set_minor_locator(AutoMinorLocator())

    # Save to file
    axes.set_xlabel(xlabel, fontsize=fontsize)
    if not loglog:
        axes.set_ylabel(ylabel, fontsize=fontsize)
    fname = 'convergence_{:d}_{:s}'.format(offset, method)
    if loglog:
        fname = '_'.join([fname, 'loglog'])
    savefig(fname, plot_dir, extensions=["pdf"])

    # Save legend to file
    if offset == 0 and not loglog:
        fig2, axes2 = plt.subplots()
        lines, labels = axes.get_legend_handles_labels()
        lines = [lines[-1]] + lines[:-1]
        labels = [labels[-1]] + labels[:-1]
        legend = axes2.legend(lines, labels, fontsize=fontsize_legend, frameon=False, ncol=3)
        fig2.canvas.draw()
        axes2.set_axis_off()
        bbox = legend.get_window_extent().transformed(fig2.dpi_scale_trans.inverted())
        savefig('legend', plot_dir, bbox_inches=bbox, extensions=['pdf'], tight=False)
