import argparse
import h5py
import matplotlib.pyplot as plt
import numpy as np
import os

from adapt_utils.plotting import *


# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('approach', help="Mesh adaptation approach.")
parser.add_argument('-family', help="Finite element family.")
parser.add_argument('-stabilisation', help="Stabilisation method to use.")
parser.add_argument('-anisotropic_stabilisation', help="Use anisotropic cell size measure?")
parser.add_argument('-norm_order', help="Metric normalisation order.")
parser.add_argument('-convergence_rate', help="Convergence rate for anisotropic DWR.")
args = parser.parse_args()
approach = args.approach
p = 'inf' if args.norm_order == 'inf' else float(args.norm_order or 1)
alpha = float(args.convergence_rate or 2)

# Get filenames
family = args.family or 'cg'
assert family in ('cg', 'dg')
ext = family
anisotropic_stabilisation = bool(args.anisotropic_stabilisation or False)
if ext == 'dg':
    if args.stabilisation in ('lf', 'LF', 'lax_friedrichs'):
        ext += '_lf'
else:
    if args.stabilisation in ('su', 'SU'):
        ext += '_su'
    if args.stabilisation in ('supg', 'SUPG'):
        ext += '_supg'
di = os.path.join(os.path.dirname(__file__), 'outputs', approach, '{:s}', 'hdf5')
plot_dir = os.path.join(os.path.dirname(__file__), 'plots')

methods = {
    'fixed_mesh': {'label': 'Uniform', 'marker': '*'},
    'GE_hp': {'label': r'GE$_{hp}$', 'marker': '^'},
    'GE_h': {'label': r'GE$_h$', 'marker': 'h'},
    'GE_p': {'label': r'GE$_p$', 'marker': 's'},
    'DQ': {'label': 'DQ', 'marker': 'x'},
}
for alignment in ('aligned', 'offset'):

    # TODO: Neaten
    filename = 'qoi_{:s}'.format(ext)
    if anisotropic_stabilisation:
        filename += '_anisotropic'
    filename = '{:s}_{:s}.h5'.format(filename, alignment)

    with h5py.File(os.path.join('outputs', 'fixed_mesh', 'hdf5', filename), 'r') as outfile:
        dofs = np.array(outfile['dofs'])
        qoi = np.array(outfile['qoi'])
        qoi_exact = np.array(outfile['qoi'][-1])
    absolute_error = np.abs(qoi - qoi_exact)
    relative_error = absolute_error/np.abs(qoi_exact)

    fig, axes = plt.subplots()
    label = methods['fixed_mesh']['label']
    marker = methods['fixed_mesh']['marker']
    axes.semilogx(dofs, relative_error, '--', label=label, marker=marker)

    # TODO: Neaten
    filename = 'qoi_{:s}'.format(ext)
    if anisotropic_stabilisation:
        filename += '_anisotropic'
    if 'isotropic_dwr' in approach:
        filename += '_{:.0f}'.format(alpha)
    else:
        filename += '_inf' if p == 'inf' else '_{:.0f}'.format(p)
    filename = '{:s}_{:s}.h5'.format(filename, alignment)

    # Plot convergence curves
    for method in methods:
        fname = os.path.join(di.format(method), filename)
        if not os.path.isfile(fname):
            print(fname)
            msg = "Cannot find convergence data for {:}-norm {:s} adaptation in the {:s} setup."
            print(msg.format(p, approach, alignment))
            continue
        with h5py.File(fname, 'r') as outfile:
            dofs = np.array(outfile['dofs'])
            qoi = np.array(outfile['qoi'])
        absolute_error = np.abs(qoi - qoi_exact)
        relative_error = absolute_error/np.abs(qoi_exact)
        label = methods[method]['label']
        marker = methods[method]['marker']
        axes.semilogx(dofs, relative_error, '--', label=label, marker=marker)
    axes.set_xlabel("Degrees of Freedom")
    axes.set_ylabel("Relative error")
    yticks = np.linspace(0, 0.5, 6)
    axes.set_yticks(yticks)
    axes.set_yticklabels([r"{{{:d}}}\%".format(int(yt*100)) for yt in yticks])
    axes.set_ylim([-0.01, 0.31])
    xlim = axes.get_xlim()
    axes.hlines(y=0.01, xmin=xlim[0], xmax=xlim[1], color='k', linestyle='-', label=r'1.0\% error')
    axes.set_xlim(xlim)
    axes.grid(True)
    axes.grid(True, which='minor', axis='y')

    # Save to file
    filename = 'qoi_enrichment_{:s}_{:s}'.format(approach, ext)
    if anisotropic_stabilisation:
        filename += '_anisotropic'
    filename += '_inf' if p == 'inf' else '_{:.0f}'.format(p)
    filename += '_{:.0f}'.format(alpha)
    savefig('_'.join([filename, alignment]), plot_dir, extensions=["pdf"])

    # Save legend to file
    if alignment == 'aligned':
        fig2, axes2 = plt.subplots()
        lines, labels = axes.get_legend_handles_labels()
        lines = [lines[-1]] + lines[:-1]
        labels = [labels[-1]] + labels[:-1]
        legend = axes2.legend(lines, labels, fontsize=18, frameon=False, ncol=3)
        fig2.canvas.draw()
        axes2.set_axis_off()
        bbox = legend.get_window_extent().transformed(fig2.dpi_scale_trans.inverted())
        fig2.savefig(os.path.join(plot_dir, 'legend_enrichment.pdf'), dpi='figure', bbox_inches=bbox)
