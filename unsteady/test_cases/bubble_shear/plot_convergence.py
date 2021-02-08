import h5py
import matplotlib.pyplot as plt
import numpy as np
import os

from adapt_utils.io import create_directory
from adapt_utils.plotting import *


characteristics = {
    "fixed_mesh": {"label": "Uniform refinement", "marker": "*"},
    "integrate": {"label": "Integration", "marker": "v"},
    "intersect": {"label": "Intersection", "marker": "^"},
}
approaches = []
dofs = {}
l2_error = {}
cons_error = {}
time = {}
plot_dir = create_directory(os.path.join(os.path.dirname(__file__), 'plots'))
for approach in characteristics:
    di = os.path.join(os.path.dirname(__file__), 'outputs')
    if approach in ('integrate', 'intersect'):
        di = os.path.join(di, 'hessian')
    di = os.path.join(di, approach)
    fname = os.path.join(di, "convergence.h5")
    if not os.path.isfile(fname):
        print("Cannot find convergence data {:s}".format(fname))
        continue
    approaches.append(approach)
    with h5py.File(fname, 'r') as outfile:
        dofs[approach] = np.array(outfile['dofs'])
        time[approach] = np.array(outfile['time'])
        l2_error[approach] = np.array(outfile['l2_error'])
        cons_error[approach] = np.array(outfile['cons_error'])

fig, axes = plt.subplots()
for approach in approaches:
    axes.semilogx(dofs[approach], l2_error[approach], **characteristics[approach])
axes.set_xlabel("Degrees of freedom")
axes.set_ylabel(r"Relative $\mathcal L_2$ error (\%)")
axes.grid(True)
axes.legend()
savefig("l2_error", plot_dir, extensions=["pdf"])

fig, axes = plt.subplots()
for approach in approaches:
    axes.semilogx(dofs[approach], cons_error[approach], **characteristics[approach])
axes.set_xlabel("Degrees of freedom")
axes.set_ylabel(r"$\mathcal L_1$ conservation error (\%)")
axes.grid(True)
axes.legend()
savefig("cons_error", plot_dir, extensions=["pdf"])

fig, axes = plt.subplots()
for approach in approaches:
    axes.semilogx(dofs[approach], time[approach], **characteristics[approach])
axes.set_xlabel("Degrees of freedom")
axes.set_ylabel(r"CPU time [$\mathrm s$]")
axes.grid(True)
axes.legend()
plt.tight_layout()
savefig("time", plot_dir, extensions=["pdf"])
