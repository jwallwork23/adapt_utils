import argparse
import h5py
import matplotlib.pyplot as plt
<<<<<<< HEAD
from matplotlib.ticker import AutoMinorLocator
=======
>>>>>>> dfe1c0b3a34dfef1765835b64b574a69fe60dd9a
import numpy as np
import os

from adapt_utils.plotting import *


# --- Parse arguments

parser = argparse.ArgumentParser()
parser.add_argument("-loglog")
<<<<<<< HEAD
parser.add_argument("-errorline")
parser.add_argument("-enrichment_method")
=======
parser.add_argument("-round")
parser.add_argument("-errorline")
parser.add_argument('-plot_pdf', help="Save plots to .pdf (default False).")
parser.add_argument('-plot_png', help="Save plots to .png (default False).")
parser.add_argument('-plot_all', help="Plot to .pdf, .png and .pvd (default False).")
>>>>>>> dfe1c0b3a34dfef1765835b64b574a69fe60dd9a
args = parser.parse_args()


# --- Set parameters

<<<<<<< HEAD
=======
# Parsed arguments
plot_pdf = bool(args.plot_pdf or False)
plot_png = bool(args.plot_png or False)
if bool(args.plot_all or False):
    plot_pdf = plot_png = True
extensions = []
if plot_pdf:
    extensions.append('pdf')
if plot_png:
    extensions.append('png')
if len(extensions) == 0:
    raise ValueError("Nothing to plot. Please specify a file extension.")

>>>>>>> dfe1c0b3a34dfef1765835b64b574a69fe60dd9a
# Formatting
fontsize = 18
fontsize_legend = 16
loglog = bool(args.loglog)
<<<<<<< HEAD
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
=======
xlabel = "Degrees of freedom (DOFs)"
ylabel = r"Power output $(\mathrm{kW})$"
ylabel2 = r"Relative error in power output (\%)"
if loglog:
    ylabel = ylabel2
errorline = float(args.errorline or 0.0)

# Conversion functions
power2error = lambda x: 100*(x - exact)/exact
error2power = lambda x: x*exact/100 + exact

# Curve plotting kwargs
characteristics = {
    'fixed_mesh': {'label': 'Uniform refinement', 'marker': 'o', 'color': 'cornflowerblue'},
    'carpio_isotropic': {'label': 'Isotropic adaptation', 'marker': '^', 'color': 'orange'},
    'carpio': {'label': 'Anisotropic adaptation', 'marker': 'x', 'color': 'g'},
}
for approach in characteristics:
    characteristics['linestyle'] = '-'

>>>>>>> dfe1c0b3a34dfef1765835b64b574a69fe60dd9a

# --- Plot convergence curves

di = os.path.join(os.path.dirname(__file__), 'outputs')
for offset in (0, 1):
    fig, axes = plt.subplots()
    axes.grid(True)

    # Read converged QoI value from file
<<<<<<< HEAD
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
=======
    with h5py.File(os.path.join(di, 'fixed_mesh/hdf5/qoi_offset_{:d}.h5'.format(offset)), 'r') as f:
        exact = np.array(f['qoi'])[-1]
        if bool(args.round or False):
            exact = np.around(exact, decimals=-2)

    # Plot convergence curves
    for approach in ('fixed_mesh', 'carpio_isotropic', 'carpio'):

        # Read data from file
        fname = os.path.join(di, '{:s}/hdf5/qoi_offset_{:d}.h5'.format(approach, offset))
        if not os.path.exists(fname):
            raise IOError("File {:s} does not exist.".format(fname))
        with h5py.File(fname, 'r') as f:
            dofs, qois = np.array(f['dofs']), np.array(f['qoi'])
            if loglog and approach == 'fixed_mesh':
                dofs, qois = np.array(dofs[:-1]), np.array(qois[:-1])

        # Plot convergence curves
        if loglog:
            axes.loglog(dofs, power2error(qois), **characteristics[approach])
        else:
            axes.semilogx(dofs, qois, **characteristics[approach])

    # Annotate with an error line
    xlim = axes.get_xlim()
    hlines = [exact, ]
    if not loglog:
        if errorline > 1e-3:
            hlines.append((1.0 + errorline/100)*exact)
        label = r'{:.1f}\% relative error'.format(errorline)
        plt.hlines(hlines, *xlim, linestyles='dashed', label=label)
    axes.set_xlim(xlim)
    ax.set_xlim(xlim)
    ytick = r"{:.2f}\%" if loglog else "{:.2f}"
    scale = 1.0 if loglog else 1e-3
    yticks = [ytick.format(scale*i) for i in ax.get_yticks().tolist()]
    axes.set_yticklabels(yticks)
    plt.xlabel(xlabel, fontsize=fontsize)
    plt.ylabel(ylabel, fontsize=fontsize)
    plt.legend(fontsize=fontsize_legend)

    # Add a secondary axis
    if not loglog:
        secax = axes.secondary_yaxis('right', functions=(power2error, error2power))
        secax.set_ylabel(ylabel2, fontsize=fontsize)
        yticks = [r"{:.2f}\%".format(i) for i in secax.get_yticks().tolist()]
        secax.set_yticklabels(yticks)

    # Save to file
    plt.tight_layout()
    fname = os.path.join(di, 'convergence_{:d}'.format(offset))
    if loglog:
        fname = '_'.join([fname, 'loglog'])
    for ext in extensions:
        plt.savefig('.'.join([fname, ext]))

plt.show()
>>>>>>> dfe1c0b3a34dfef1765835b64b574a69fe60dd9a
