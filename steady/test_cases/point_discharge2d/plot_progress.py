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
fname = "_".join(["{:s}", ext, args.level])
fname += ".npy"
controls = np.load(os.path.join(di, fname.format("control")))
functionals = np.load(os.path.join(di, fname.format("functional")))
gradients = np.load(os.path.join(di, fname.format("gradient")))

# Plot progress
fig, axes = plt.subplots()
axes.plot(controls, functionals, 'o')
axes.set_xlabel("Radius [m]")
axes.set_ylabel("QoI")
plt.tight_layout()
savefig("progress_{:s}".format(ext), di, extensions=["pdf", "png"])
