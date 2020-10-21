import argparse
import h5py
import matplotlib.pyplot as plt
import numpy as np
import os

from adapt_utils.plotting import *


# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('approach', help="Mesh adaptation approach.")
parser.add_argument('family', help="Finite element family.")
parser.add_argument('-stabilisation', help="Stabilisation method to use.")
args = parser.parse_args()

# Get filenames
ext = args.family
assert ext in ('cg', 'dg')
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

assert args.approach in ('dwr', 'a_posteriori', 'a_priori')
for alignment in ('aligned', 'offset'):
    fig, axes = plt.subplots()

    filename = 'qoi_{:s}'.format(ext)
    fname = os.path.join(di.format('fixed_mesh'), '{:s}_{:s}.h5'.format(filename, alignment))
    with h5py.File(fname, 'r') as outfile:
        elements = np.array(outfile['elements'])
        qoi = np.array(outfile['qoi'])
        # qoi_exact = np.array(outfile['qoi_exact'][-1])
        qoi_exact = np.array(outfile['qoi'][-1])
        absolute_error = np.abs(qoi - qoi_exact)
        relative_error = absolute_error/np.abs(qoi_exact)
    axes.semilogx(elements, relative_error, '--x', label="Uniform")

    # Plot convergence curves
    for p in [1, 2, 4, 8]:
        filename = 'qoi_{:s}_{:.0f}'.format(ext, p)
        fname = os.path.join(di.format(args.approach), '{:s}_{:s}.h5'.format(filename, alignment))
        if not os.path.isfile(fname):
            msg = "Cannot find convergence data for {:}-norm {:s} adaptation in the {:s} setup."
            print(msg.format(p, args.approach, alignment))
            continue
        with h5py.File(fname, 'r') as outfile:
            elements = np.array(outfile['elements'])
            qoi = np.array(outfile['qoi'])
        absolute_error = np.abs(qoi - qoi_exact)
        relative_error = absolute_error/np.abs(qoi_exact)
        axes.semilogx(elements, relative_error, '--x', label=r"$p={{{:.0f}}}$".format(p))
    axes.set_xlabel("Element count")
    axes.set_ylabel("Relative error")
    axes.set_xticks([1.0e+03, 1.0e+04, 1.0e+05, 1.0e+06])
    yticks = np.linspace(0, 0.5, 6)
    axes.set_yticks(yticks)
    axes.set_yticklabels([r"{{{:d}}}\%".format(int(yt*100)) for yt in yticks])
    axes.set_ylim([-0.01, 0.51])
    xlim = axes.get_xlim()
    axes.hlines(y=0.01, xmin=xlim[0], xmax=xlim[1], color='k', linestyle=':', label=r'1.0\% error')
    axes.set_xlim(xlim)
    axes.legend(loc='upper right', fontsize=18)
    axes.grid(True)
    axes.grid(True, which='minor', axis='y')
    fname = '_'.join(['qoi', ext, args.approach, alignment, 'norms'])
    savefig(fname, plot_dir, extensions=['pdf', 'png'])
