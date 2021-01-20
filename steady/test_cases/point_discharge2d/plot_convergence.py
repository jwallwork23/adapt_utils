import argparse
import h5py
import matplotlib.pyplot as plt
import numpy as np
import os

from adapt_utils.plotting import *


# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('mode', help="""
    Choose from {'forward', 'adjoint', 'avg', 'int', 'dwr', 'anisotropic_dwr, 'weighed_hessian',
    'weighted_gradient', 'isotropic'}.
    """)
parser.add_argument('-family', help="Finite element family.")
parser.add_argument('-stabilisation', help="Stabilisation method to use.")
parser.add_argument('-anisotropic_stabilisation', help="Use anisotropic cell size measure?")
parser.add_argument('-norm_order', help="Metric normalisation order.")
parser.add_argument('-convergence_rate', help="Convergence rate for anisotropic DWR.")
parser.add_argument('-enrichment_method', help="Choose from {'GE_hp', 'GE_h', 'GE_p', 'PR', 'DQ'}.")
parser.add_argument('-loglog', help="Use a log-log scale")
args = parser.parse_args()
mode = args.mode
assert mode in (
    'forward', 'adjoint', 'avg', 'int', 'dwr',
    'anisotropic_dwr', 'weighted_hessian', 'weighted_gradient', 'isotropic'
)
p = 'inf' if args.norm_order == 'inf' else float(args.norm_order or 1)
alpha = float(args.convergence_rate or 2)
enrichment_method = args.enrichment_method or 'GE_h'
loglog = bool(args.loglog or False)

# Get filenames
family = args.family or 'cg'
assert family in ('cg', 'dg')
ext = family
stabilisation = args.stabilisation or 'supg'
anisotropic_stabilisation = False if args.anisotropic_stabilisation == "0" else True
if ext == 'dg':
    if stabilisation in ('lf', 'LF', 'lax_friedrichs'):
        ext += '_lf'
else:
    if stabilisation in ('su', 'SU'):
        ext += '_su'
    if stabilisation in ('supg', 'SUPG'):
        ext += '_supg'
fixed_di = os.path.join(os.path.dirname(__file__), 'outputs', 'fixed_mesh', 'hdf5')
di = os.path.join(os.path.dirname(__file__), 'outputs', '{:s}', enrichment_method, 'hdf5')
plot_dir = os.path.join(os.path.dirname(__file__), 'plots')

label_ext = ''
if mode == 'adjoint':
    label_ext = '_adjoint'
elif mode == 'avg':
    label_ext = '_int'
elif mode == 'int':
    label_ext = '_int'
approaches = {
    'fixed_mesh': {'label': 'Uniform', 'marker': '*'},
    'dwr' + label_ext: {'label': 'Isotropic DWR', 'marker': '^'},
    'anisotropic_dwr' + label_ext: {'label': 'Anisotropic DWR', 'marker': 'h'},
    'weighted_hessian' + label_ext: {'label': 'Weighted Hessian', 'marker': 's'},
    'weighted_gradient' + label_ext: {'label': 'Weighted Gradient', 'marker': 'x'},
}
if mode in ('dwr', 'anisotropic_dwr', 'weighted_hessian', 'weighted_gradient'):
    approaches = {'fixed_mesh': {'label': 'Uniform', 'marker': '*'}}
    approaches[mode] = {'label': 'Forward', 'marker': '^'}
    approaches[mode + '_adjoint'] = {'label': 'Adjoint', 'marker': 'h'}
    approaches[mode + '_avg'] = {'label': 'Averaged', 'marker': 's'}
    approaches[mode + '_int'] = {'label': 'Intersected', 'marker': 'x'}
elif mode == 'isotropic':
    approaches = {'fixed_mesh': {'label': 'Uniform', 'marker': '*'}}
    approaches['dwr'] = {'label': 'Vertex-based', 'marker': '^'}
    approaches['isotropic_dwr'] = {'label': 'Element-based', 'marker': 'h'}
for alignment in ('aligned', 'offset'):
    fig, axes = plt.subplots()

    # Plot convergence curves
    for approach in approaches:
        filename = 'qoi_{:s}'.format(ext)
        if anisotropic_stabilisation:
            filename += '_anisotropic'
        if approach == 'fixed_mesh':
            fpath = fixed_di
        else:
            fpath = di.format(approach)
            if 'isotropic_dwr' in approach:
                filename += '_{:.0f}'.format(alpha)
            else:
                filename += '_inf' if p == 'inf' else '_{:.0f}'.format(p)
        fname = os.path.join(fpath, '{:s}_{:s}.h5'.format(filename, alignment))
        if not os.path.isfile(fname):
            print(fname)
            msg = "Cannot find convergence data for {:}-norm {:s} adaptation in the {:s} setup."
            print(msg.format(p, approach, alignment))
            continue
        with h5py.File(fname, 'r') as outfile:
            elements = np.array(outfile['elements'])
            dofs = np.array(outfile['dofs'])
            qoi = np.array(outfile['qoi'])
            if approach == 'fixed_mesh':
                qoi_exact = np.array(outfile['qoi'][-1])  # Discretisation error
                # qoi_exact = np.array(outfile['qoi_exact'][-1])  # Total error
        absolute_error = np.abs(qoi - qoi_exact)
        relative_error = absolute_error/np.abs(qoi_exact)
        label = approaches[approach]['label']
        marker = approaches[approach]['marker']
        if approach == 'fixed_mesh':
            dofs = dofs[:-1]
            relative_error = relative_error[:-1]
        if loglog:
            axes.loglog(dofs, relative_error, '--', label=label, marker=marker)
        else:
            axes.semilogx(dofs, relative_error, '--', label=label, marker=marker)
    axes.set_xlabel("Degrees of Freedom")
    axes.set_ylabel("Relative error")
    if not loglog:
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
    filename = 'qoi_{:s}_{:s}'.format(mode, ext)
    if anisotropic_stabilisation:
        filename += '_anisotropic'
    filename += '_inf' if p == 'inf' else '_{:.0f}'.format(p)
    filename += '_{:.0f}'.format(alpha)
    savefig('_'.join([filename, alignment, enrichment_method]), plot_dir, extensions=['pdf'])

    # Save legend to file
    if alignment == 'aligned':
        fig2, axes2 = plt.subplots()
        lines, labels = axes.get_legend_handles_labels()
        if not loglog:
            lines = [lines[-1]] + lines[:-1]
            labels = [labels[-1]] + labels[:-1]
        legend = axes2.legend(lines, labels, fontsize=18, frameon=False, ncol=3)
        fig2.canvas.draw()
        axes2.set_axis_off()
        bbox = legend.get_window_extent().transformed(fig2.dpi_scale_trans.inverted())
        savefig('legend_{:s}'.format(mode), plot_dir, bbox_inches=bbox, extensions=['pdf'])
