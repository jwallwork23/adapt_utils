import argparse
import matplotlib.pyplot as plt
import numpy as np
import os

from adapt_utils.plotting import *


# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('level')
parser.add_argument('family')
parser.add_argument('-stabilisation')
parser.add_argument('-use_automatic_sipg_parameter')
args = parser.parse_args()
assert args.family in ('cg', 'dg')
auto_sipg = bool(args.use_automatic_sipg_parameter or False)

# Get output directory
di = os.path.join(os.path.dirname(__file__), 'outputs', 'fixed_mesh')
di = os.path.join(di, args.stabilisation or args.family)
if args.family == 'dg' and auto_sipg:
    di += '_sipg'
plot_dir = os.path.join(os.path.dirname(__file__), 'plots')

# Load progress arrays
ext = args.family
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
ext += "_" + args.level
fname = "_".join(["{:s}", ext])
fname += ".npy"
controls = np.load(os.path.join(di, fname.format("control")))
functionals = np.load(os.path.join(di, fname.format("functional")))
gradients = np.load(os.path.join(di, fname.format("gradient")))

# Plot progress
fig, axes = plt.subplots()
fname = os.path.join(di, "parameter_space_{:s}.npy".format(args.level))
if os.path.isfile(fname):
    axes.semilogy(np.linspace(0.01, 0.15, 31), np.load(fname), 'x', color='C0')
axes.semilogy(controls, functionals, 'o', color='C1')
axes.set_xlabel(r"Radius [$\mathrm m$]")
axes.set_ylabel(r"$J_{\mathrm{calibration}}$")
axes.set_xlim([0, 0.16])
axes.set_ylim([1e-2, 1e3])
axes.grid(True)
savefig("progress_{:s}".format(ext), plot_dir, extensions=["pdf", "png"])
