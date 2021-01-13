import argparse
import h5py
import matplotlib.pyplot as plt
import numpy as np
import os

from adapt_utils.plotting import *


# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('family', help="Finite element family.")
parser.add_argument('-stabilisation', help="Stabilisation method to use.")
parser.add_argument('-anisotropic_stabilisation', help="Use anisotropic cell size measure?")
parser.add_argument('-norm_order', help="Metric normalisation order.")
parser.add_argument('-convergence_rate', help="Convergence rate for anisotropic DWR.")
args = parser.parse_args()
p = 'inf' if args.norm_order == 'inf' else float(args.norm_order or 1)
alpha = float(args.convergence_rate or 10)

# Get filenames
ext = args.family
assert ext in ('cg', 'dg')
anisotropic_stabilisation = bool(args.anisotropic_stabilisation or False)
if ext == 'dg':
    if args.stabilisation in ('lf', 'LF', 'lax_friedrichs'):
        ext += '_lf'
else:
    if args.stabilisation in ('su', 'SU'):
        ext += '_su'
    if args.stabilisation in ('supg', 'SUPG'):
        ext += '_supg'
di = os.path.join(os.path.dirname(__file__), 'outputs', '{:s}', 'hdf5')
plot_dir = os.path.join(os.path.dirname(__file__), 'plots')

approaches = {
    'fixed_mesh': {'label': 'Uniform', 'marker': '*'},
    'dwr': {'label': 'Isotropic DWR', 'marker': '^'},
    'anisotropic_dwr': {'label': 'Anisotropic DWR', 'marker': 'h'},
    'weighted_hessian': {'label': 'Weighted Hessian', 'marker': 's'},
    'weighted_gradient': {'label': 'Weighted Gradient', 'marker': 'x'},
    # 'dwr_adjoint': {'label': 'Isotropic DWR adjoint', 'marker': 'v'},
    # 'dwr_avg': {'label': 'Isotropic DWR adjoint', 'marker': '>'},
}
for alignment in ('aligned', 'offset'):
    fig, axes = plt.subplots()

    # Plot convergence curves
    for approach in approaches:
        filename = 'qoi_{:s}'.format(ext)
        if anisotropic_stabilisation:
            filename += '_anisotropic'
        if approach != 'fixed_mesh':
            if approach == 'anisotropic_dwr':
                filename += '_{:.0f}'.format(alpha)
            else:
                filename += '_inf' if p == 'inf' else '_{:.0f}'.format(p)
        fname = os.path.join(di.format(approach), '{:s}_{:s}.h5'.format(filename, alignment))
        if not os.path.isfile(fname):
            msg = "Cannot find convergence data for {:}-norm {:s} adaptation in the {:s} setup."
            print(msg.format(p, approach, alignment))
            continue
        with h5py.File(fname, 'r') as outfile:
            elements = np.array(outfile['elements'])
            dofs = np.array(outfile['dofs'])
            qoi = np.array(outfile['qoi'])
            if approach == 'fixed_mesh':
                # qoi_exact = np.array(outfile['qoi_exact'][-1])
                qoi_exact = np.array(outfile['qoi'][-1])
            if approach == 'dwr':
                estimators = np.abs(np.array(outfile['estimators']))
        absolute_error = np.abs(qoi - qoi_exact)
        relative_error = absolute_error/np.abs(qoi_exact)
        if approach in ('dwr', 'dwr_adjoint', 'dwr_avg'):
            effectivity = estimators/absolute_error
            print(approach, " effectivity indices: ", effectivity)  # FIXME
            # print(approach, "effectivity indices: ", effectivity/elements)
        label = approaches[approach]['label']
        marker = approaches[approach]['marker']
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
    filename = 'qoi_{:s}'.format(ext)
    if anisotropic_stabilisation:
        filename += '_anisotropic'
    filename += '_inf' if p == 'inf' else '_{:.0f}'.format(p)
    filename += '_{:.0f}'.format(alpha)
    savefig('_'.join([filename, alignment]), plot_dir, extensions=['pdf', 'png'])

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
        fig2.savefig(os.path.join(plot_dir, 'legend.png'), dpi='figure', bbox_inches=bbox)
        fig2.savefig(os.path.join(plot_dir, 'legend.pdf'), dpi='figure', bbox_inches=bbox)
