import argparse
import h5py
<<<<<<< HEAD
import matplotlib.ticker as ticker
=======
>>>>>>> dfe1c0b3a34dfef1765835b64b574a69fe60dd9a
import matplotlib.pyplot as plt
import numpy as np
import os

from adapt_utils.plotting import *


# Parse arguments
parser = argparse.ArgumentParser()
<<<<<<< HEAD
parser.add_argument('mode', help="""
    Choose from {'forward', 'adjoint', 'avg', 'int', 'dwr', 'anisotropic_dwr, 'weighed_hessian',
    'weighted_gradient', 'isotropic', 'enrichment'}.
    """)
parser.add_argument('-family', help="Finite element family.")
=======
parser.add_argument('family', help="Finite element family.")
>>>>>>> dfe1c0b3a34dfef1765835b64b574a69fe60dd9a
parser.add_argument('-stabilisation', help="Stabilisation method to use.")
parser.add_argument('-anisotropic_stabilisation', help="Use anisotropic cell size measure?")
parser.add_argument('-norm_order', help="Metric normalisation order.")
parser.add_argument('-convergence_rate', help="Convergence rate for anisotropic DWR.")
<<<<<<< HEAD
parser.add_argument('-enrichment_method', help="Choose from {'GE_hp', 'GE_h', 'GE_p', 'PR', 'DQ'}.")
parser.add_argument('-loglog', help="Use a log-log scale")
args = parser.parse_args()
mode = args.mode
assert mode in (
    'forward', 'adjoint', 'avg', 'int', 'dwr', 'anisotropic_dwr', 'weighted_hessian',
    'weighted_gradient', 'isotropic', 'enrichment'
)
p = 'inf' if args.norm_order == 'inf' else float(args.norm_order or 1)
alpha = float(args.convergence_rate or 2)
loglog = bool(args.loglog or False)

adjoint = mode not in ('forward', 'isotropic', 'enrichment')
enrichment_method = args.enrichment_method or 'GE_p' if adjoint else 'DQ'

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
output_di = os.path.join(os.path.dirname(__file__), 'outputs')
fixed_di = os.path.join(output_di, 'fixed_mesh', 'hdf5')
plot_dir = os.path.join(os.path.dirname(__file__), 'plots')

label_ext = ''
if mode == 'adjoint':
    label_ext = '_adjoint'
elif mode == 'avg':
    label_ext = '_avg'
elif mode == 'int':
    label_ext = '_int'
approaches = {
    'fixed_mesh': {'label': 'Uniform', 'marker': '*'},
    'dwr' + label_ext: {'label': 'Isotropic DWR', 'marker': '^'},
    'anisotropic_dwr' + label_ext: {'label': 'Anisotropic DWR', 'marker': 'h'},
    # 'isotropic_dwr' + label_ext: {'label': 'Isotropic DWR', 'marker': 'h'},
    'weighted_hessian' + label_ext: {'label': 'Weighted Hessian', 'marker': 's'},
    'weighted_gradient' + label_ext: {'label': 'Weighted Gradient', 'marker': 'x'},
}
if mode == 'avg':
    approaches['dwr_both'] = {'label': '$2^{nd}$ order isotropic DWR', 'marker': 'v'}
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
elif mode == 'enrichment':
    approaches = {'fixed_mesh': {'label': 'Uniform', 'marker': '*'}}
    approaches['GE_hp'] = {'label': r'GE$_{hp}$', 'marker': '^'}
    approaches['GE_h'] = {'label': r'GE$_h$', 'marker': 'h'}
    approaches['GE_p'] = {'label': r'GE$_p$', 'marker': 's'}
    approaches['DQ'] = {'label': 'DQ', 'marker': 'x'}
colours = ['C{:d}'.format(i) for i in range(6)]

=======
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
}
>>>>>>> dfe1c0b3a34dfef1765835b64b574a69fe60dd9a
for alignment in ('aligned', 'offset'):
    fig, axes = plt.subplots()

    # Plot convergence curves
<<<<<<< HEAD
    for approach, colour in zip(approaches, colours):
        if mode == 'enrichment':
            enrichment_method = approach
        di = os.path.join(output_di, '{:s}', enrichment_method, 'hdf5')
        filename = 'qoi_{:s}'.format(ext)
        if anisotropic_stabilisation:
            filename += '_anisotropic'
        if approach == 'fixed_mesh':
            fpath = fixed_di
        else:
            fpath = di.format('dwr' if mode == 'enrichment' else approach)
            if 'isotropic_dwr' in approach:
                filename += '_{:.0f}'.format(alpha)
            else:
                filename += '_inf' if p == 'inf' else '_{:.0f}'.format(p)
        fname = os.path.join(fpath, '{:s}_{:s}.h5'.format(filename, alignment))
        if not os.path.isfile(fname):
            print(fname)
=======
    for approach in approaches:
        filename = 'qoi_{:s}'.format(ext)
        if approach != 'fixed_mesh':
            if anisotropic_stabilisation:
                filename += '_anisotropic'
            if approach == 'anisotropic_dwr':
                filename += '_{:.0f}'.format(alpha)
            else:
                filename += '_inf' if p == 'inf' else '_{:.0f}'.format(p)
        fname = os.path.join(di.format(approach), '{:s}_{:s}.h5'.format(filename, alignment))
        if not os.path.isfile(fname):
>>>>>>> dfe1c0b3a34dfef1765835b64b574a69fe60dd9a
            msg = "Cannot find convergence data for {:}-norm {:s} adaptation in the {:s} setup."
            print(msg.format(p, approach, alignment))
            continue
        with h5py.File(fname, 'r') as outfile:
            elements = np.array(outfile['elements'])
<<<<<<< HEAD
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
            axes.loglog(dofs, relative_error, '--', label=label, marker=marker, color=colour)
        else:
            axes.plot(dofs, relative_error, '--', label=label, marker=marker, color=colour)
            axes.set_xscale('log')
            x_minor = ticker.LogLocator(base=10.0, subs=np.arange(1.0, 10.0)*0.1, numticks=10)
            axes.xaxis.set_minor_locator(x_minor)
            axes.xaxis.set_minor_formatter(ticker.NullFormatter())
            y_minor = ticker.LinearLocator(numticks=23)
            axes.yaxis.set_minor_locator(y_minor)
    axes.set_xlabel("Degrees of Freedom")
    axes.set_ylabel("Relative error")
    if not loglog:
        yticks = np.linspace(0, 0.5, 6)
        axes.set_yticks(yticks)
        axes.set_yticklabels([r"{{{:d}}}\%".format(int(yt*100)) for yt in yticks])
        axes.set_ylim([-0.01, 0.21])
        xlim = axes.get_xlim()
        axes.set_xticks([1.0e+03, 1.0e+04, 1.0e+05, 1.0e+06])
        axes.hlines(y=0.01, xmin=xlim[0], xmax=xlim[1], color='k', linestyle='-', label=r'1.0\% error')
        axes.set_xlim(xlim)
    axes.grid(True)

    # Save to file
    filename = 'qoi_{:s}_{:s}'.format(mode, ext)
=======
            qoi = np.array(outfile['qoi'])
            if approach == 'fixed_mesh':
                # qoi_exact = np.array(outfile['qoi_exact'][-1])
                qoi_exact = np.array(outfile['qoi'][-1])
            if approach == 'dwr':
                estimators = np.abs(np.array(outfile['estimators']))
        absolute_error = np.abs(qoi - qoi_exact)
        relative_error = absolute_error/np.abs(qoi_exact)
        if approach == 'dwr':
            effectivity = estimators/absolute_error
            print("Effectivity indices: ", effectivity)  # FIXME
            # print("Effectivity indices: ", effectivity/elements)
        label = approaches[approach]['label']
        marker = approaches[approach]['marker']
        axes.semilogx(elements, relative_error, '--', label=label, marker=marker)
    axes.set_xlabel("Element count")
    axes.set_ylabel("Relative error")
    axes.set_xticks([1.0e+03, 1.0e+04, 1.0e+05, 1.0e+06])
    yticks = np.linspace(0, 0.5, 6)
    axes.set_yticks(yticks)
    axes.set_yticklabels([r"{{{:d}}}\%".format(int(yt*100)) for yt in yticks])
    axes.set_ylim([-0.01, 0.31])
    # xlim = axes.get_xlim()
    xlim = [0.8e+03, 1.2e+06]
    axes.hlines(y=0.01, xmin=xlim[0], xmax=xlim[1], color='k', linestyle='-', label=r'1.0\% error')
    axes.set_xlim(xlim)
    if alignment == 'aligned':
        axes.legend(bbox_to_anchor=(0.5, 0.3), fontsize=18)
    axes.grid(True)
    axes.grid(True, which='minor', axis='y')

    # Save to file
    filename = 'qoi_{:s}'.format(ext)
>>>>>>> dfe1c0b3a34dfef1765835b64b574a69fe60dd9a
    if anisotropic_stabilisation:
        filename += '_anisotropic'
    filename += '_inf' if p == 'inf' else '_{:.0f}'.format(p)
    filename += '_{:.0f}'.format(alpha)
<<<<<<< HEAD
    savefig('_'.join([filename, alignment, enrichment_method]), plot_dir, extensions=['pdf'])

    # Save legend to file
    if alignment == 'aligned':
        fig2, axes2 = plt.subplots()
        lines, labels = axes.get_legend_handles_labels()
        if not loglog:
            lines = [lines[-1]] + lines[:-1]
            labels = [labels[-1]] + labels[:-1]
        legend = axes2.legend(lines, labels, fontsize=18, frameon=False, ncol=4 if mode == 'avg' else 3)
        fig2.canvas.draw()
        axes2.set_axis_off()
        bbox = legend.get_window_extent().transformed(fig2.dpi_scale_trans.inverted())
        savefig('legend_{:s}'.format(mode), plot_dir, bbox_inches=bbox, extensions=['pdf'], tight=False)
=======
    savefig('_'.join([filename, alignment]), plot_dir, extensions=['pdf', 'png'])
>>>>>>> dfe1c0b3a34dfef1765835b64b574a69fe60dd9a
