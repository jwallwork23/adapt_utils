import argparse
import h5py
import matplotlib.pyplot as plt
import numpy as np
import os

from adapt_utils.plotting import *


# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('family')
parser.add_argument('-stabilisation')
parser.add_argument('-use_automatic_sipg_parameter')
args = parser.parse_args()

# Get filenames
ext = args.family
assert ext in ('cg', 'dg')
auto_sipg = bool(args.use_automatic_sipg_parameter or False)
if ext == 'dg':
    if args.stabilisation in ('lf', 'LF', 'lax_friedrichs'):
        ext += '_lf'
    if auto_sipg:
        ext += '_sipg'
else:
    if args.stabilisation in ('su', 'SU'):
        ext += '_su'
    if args.stabilisation in ('supg', 'SUPG'):
        ext += '_supg'
filename = 'qoi_{:s}'.format(ext)
di = os.path.join(os.path.dirname(__file__), 'outputs', 'fixed_mesh', 'hdf5')
plot_dir = os.path.join(os.path.dirname(__file__), 'plots')

for alignment in ('aligned', 'offset'):

    # Read data from file
    fname = os.path.join(di, '{:s}_{:s}.h5'.format(filename, alignment))
    assert os.path.isfile(fname)
    with h5py.File(fname, 'r') as outfile:
        elements = np.array(outfile['elements'])
        qoi = np.array(outfile['qoi'])
        qoi_exact = np.array(outfile['qoi_exact'][-1])
    relative_error = np.abs(qoi - qoi_exact)/np.abs(qoi_exact)

    # Plot convergence curves
    fig, axes = plt.subplots()
    axes.semilogx(elements, relative_error)
    axes.set_xlabel("Element count")
    axes.set_ylabel("Relative error")
    axes.grid(True)
    plt.tight_layout()
    savefig('_'.join([filename, alignment]), plot_dir, extensions=['pdf', 'png'])
