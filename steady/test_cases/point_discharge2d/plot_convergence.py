import argparse
import h5py
from matplotlib.ticker import MultipleLocator
import matplotlib.pyplot as plt
import numpy as np
import os

from adapt_utils.plotting import *


# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('family')
parser.add_argument('-stabilisation')
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
filename = 'qoi_{:s}'.format(ext)
di = os.path.join(os.path.dirname(__file__), 'outputs', '{:s}', 'hdf5')
plot_dir = os.path.join(os.path.dirname(__file__), 'plots')

approaches = {
    'fixed_mesh': 'Uniform',
    'dwr': 'Isotropic',
    'a_posteriori': 'A posteriori',
    # 'a_priori': 'A priori',  # TODO
}
for alignment in ('aligned', 'offset'):
    fig, axes = plt.subplots()

    # Plot convergence curves
    for approach in approaches:
        fname = os.path.join(di.format(approach), '{:s}_{:s}.h5'.format(filename, alignment))
        assert os.path.isfile(fname)
        with h5py.File(fname, 'r') as outfile:
            elements = np.array(outfile['elements'])
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
            print("Effectivity indices: ", effectivity)
        axes.semilogx(elements, relative_error, '--x', label=approaches[approach])
    axes.set_xlabel("Element count")
    axes.set_ylabel("Relative error")
    axes.legend(loc='upper right')
    yticks = np.linspace(0, 1, 6)
    axes.set_yticks(yticks)
    axes.set_yticklabels([r"{{{:d}}}\%".format(int(yt*100)) for yt in yticks])
    axes.yaxis.set_minor_locator(MultipleLocator(yticks[1]/2))
    axes.grid(True)
    axes.grid(True, which='minor', axis='y')
    savefig('_'.join([filename, alignment]), plot_dir, extensions=['pdf', 'png'])
