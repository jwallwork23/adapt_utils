import argparse
import h5py
import matplotlib.pyplot as plt
import numpy as np
import os

from adapt_utils.plotting import *


# --- Parse arguments

parser = argparse.ArgumentParser()
parser.add_argument("-loglog")
parser.add_argument("-round")
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


# --- Plot convergence curves

di = os.path.join(os.path.dirname(__file__), 'outputs')
for offset in (0, 1):
    fig, axes = plt.subplots()
    axes.grid(True)

    # Read converged QoI value from file
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
