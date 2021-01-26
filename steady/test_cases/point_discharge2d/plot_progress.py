import argparse
import matplotlib.pyplot as plt
from math import log10
import numpy as np
import os

from adapt_utils.plotting import *


# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('level', help="Number of uniform refinements to apply to the initial mesh.")
parser.add_argument('family', help="Finite element family.")
parser.add_argument('-stabilisation', help="Stabilisation method to use.")
parser.add_argument('-anisotropic_stabilisation', help="Use anisotropic cell size measure?")
args = parser.parse_args()
assert args.family in ('cg', 'dg')

# Get output directory
di = os.path.join(os.path.dirname(__file__), 'outputs', 'fixed_mesh')
di = os.path.join(di, args.stabilisation or args.family)
plot_dir = os.path.join(os.path.dirname(__file__), 'plots')

# Load progress arrays
ext = args.family
anisotropic_stabilisation = bool(args.anisotropic_stabilisation or False)
if ext == 'dg':
    if args.stabilisation in ('lf', 'LF', 'lax_friedrichs'):
        ext += '_lf'
else:
    if args.stabilisation in ('su', 'SU'):
        ext += '_su'
    if args.stabilisation in ('supg', 'SUPG'):
        ext += '_supg'
    if anisotropic_stabilisation:
        ext += '_anisotropic'
ext += "_" + args.level
fname = "_".join(["{:s}", ext])
fname += ".npy"
controls = np.load(os.path.join(di, fname.format("control")))
functionals = np.load(os.path.join(di, fname.format("functional")))
# gradients = np.load(os.path.join(di, fname.format("gradient")))

# Plot progress
fig, axes = plt.subplots(figsize=(8, 4))
fname = os.path.join(di, "parameter_space_{:s}.npy".format(args.level))
parameter_space = np.linspace(0.01, 0.4, 100)
if os.path.isfile(fname):
    axes.plot(parameter_space, np.load(fname), ':', color='C0')
axes.scatter(controls, functionals, c=list(range(len(controls))), cmap='autumn')
axes.plot(controls, functionals, color='C1', linewidth=0.5)
axes.set_yscale('log')
for i in [0, 3, 5, 6, 7]:
    axes.annotate(
        "",
        xy=(0.5*sum(controls[i:i+2]), 10**(0.5*(log10(functionals[i]) + log10(functionals[i+1])))),
        xytext=(controls[i], functionals[i]),
        arrowprops=dict(arrowstyle="->, head_width=0.1", color='C1', lw=0.5))
axes.set_xlabel(r"Radius [$\mathrm m$]")
axes.set_ylabel(r"$J_{\mathrm{calibration}}$")
axes.set_xlim([0, 0.4])
axes.set_ylim([1e-2, 1e5])
axes.grid(True)
savefig("progress_{:s}".format(ext), plot_dir, extensions=["pdf", "jpg"])
