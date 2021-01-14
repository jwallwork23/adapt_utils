import argparse
import h5py
import matplotlib.pyplot as plt
import numpy as np
import os

from adapt_utils.plotting import *


# --- Parse arguments

parser = argparse.ArgumentParser()
parser.add_argument("-loglog")
parser.add_argument("-errorline")
parser.add_argument('-plot_pdf', help="Save plots to .pdf (default False).")
parser.add_argument('-plot_png', help="Save plots to .png (default False).")
parser.add_argument('-plot_all', help="Plot to .pdf, .png and .pvd (default False).")
args = parser.parse_args()


# --- Set parameters

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

# Formatting
fontsize = 18
fontsize_legend = 16
loglog = bool(args.loglog)
xlabel = "Degrees of freedom"
ylabel = r"Power output $(\mathrm{MW})$"
ylabel2 = r"Relative discretisation error (\%)"
if loglog:
    ylabel = ylabel2
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


# --- Plot convergence curves

di = os.path.join(os.path.dirname(__file__), 'outputs')
for offset in (0, 1):
    fig, axes = plt.subplots()
    axes.grid(True)

    # Read converged QoI value from file
    with h5py.File(os.path.join(di, 'fixed_mesh/hdf5/qoi_offset_{:d}.h5'.format(offset)), 'r') as f:
        exact = np.array(f['qoi'])[-1]

    # Conversion functions
    power2error = lambda x: 100*(x - exact)/exact
    error2power = lambda x: x*exact/100 + exact

    # Plot convergence curves
    for approach in characteristics:

        # Read data from file
        fname = os.path.join(di, '{:s}/hdf5/qoi_offset_{:d}.h5'.format(approach, offset))
        if not os.path.exists(fname):
            print("File {:s} does not exist.".format(fname))
            continue
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

    # Add a secondary axis
    if not loglog:
        secax = axes.secondary_yaxis('right', functions=(power2error, error2power))
        secax.set_ylabel(ylabel2, fontsize=fontsize)

    # Save to file
    axes.set_xlabel(xlabel, fontsize=fontsize)
    axes.set_ylabel(ylabel, fontsize=fontsize)
    axes.legend(fontsize=fontsize_legend)
    fname = 'convergence_{:d}'.format(offset)
    if loglog:
        fname = '_'.join([fname, 'loglog'])
    savefig(fname, 'plots', extensions=extensions)
