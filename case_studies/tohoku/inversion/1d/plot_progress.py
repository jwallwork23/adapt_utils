import argparse
import matplotlib.pyplot as plt
import numpy as np
import scipy.interpolate as si
import sys

from adapt_utils.plotting import *  # NOQA


parser = argparse.ArgumentParser()
parser.add_argument("level")
parser.add_argument("mode")
args = parser.parse_args()

level = int(args.level)
mode = args.mode
assert mode in ('discrete', 'continuous')

try:
    control_trajectory = np.load('data/opt_progress_{:s}_{:d}_ctrl.npy'.format(mode, level))
    functional_trajectory = np.load('data/opt_progress_{:s}_{:d}_func.npy'.format(mode, level))
    gradient_trajectory = np.load('data/opt_progress_{:s}_{:d}_grad.npy'.format(mode, level))
    line_search_trajectory = np.load('data/opt_progress_{:s}_{:d}_ls.npy'.format(mode, level))
except Exception:
    print("Cannot load {:s} data for level {:d}.".format(mode, level))
    sys.exit(0)
i = 0
indices = [0]
for j, ctrl in enumerate(control_trajectory):
    if i == len(line_search_trajectory):
        break
    if np.isclose(ctrl, line_search_trajectory[i]):
        indices.append(j)
        i += 1
control_trajectory = [control_trajectory[i] for i in indices]
functional_trajectory = [functional_trajectory[i] for i in indices]
gradient_trajectory = [gradient_trajectory[i] for i in indices]

fig, axes = plt.subplots(figsize=(8, 8))
l = si.lagrange(control_trajectory[:3], functional_trajectory[:3])
dl = l.deriv()
print("Exact gradient at 10.0: {:.4f}".format(dl(10.0)))
print("Exact gradient at  5.0: {:.4f}".format(dl(5.0)))
l_min = -dl.coefficients[1]/dl.coefficients[0]
print("Minimiser of quadratic: {:.4f}".format(l_min))
xx = np.linspace(2, 10, 100)
axes.plot(xx, l(xx), ':', color='C0')
delta_m = 0.25
for m, f, g in zip(control_trajectory, functional_trajectory, gradient_trajectory):
    x = np.array([m - delta_m, m + delta_m])
    axes.plot(x, g*(x-m) + f, '-', color='C2', linewidth=5)
axes.plot(control_trajectory, functional_trajectory, 'o', color='C1', markersize=8)
axes.plot(l_min, l(l_min), '*', markersize=14, color='C0', label=r"$m^\star={:.4f}$".format(l_min))
axes.set_xlabel(r"Control parameter, $m$")
axes.set_ylabel("Quantity of Interest")
if mode == 'continuous':
    axes.yaxis.set_label_position("right")
    axes.yaxis.tick_right()
axes.grid(True)
axes.legend()
plt.tight_layout()
plt.savefig("plots/opt_progress_{:s}_{:d}.pdf".format(mode, level))

print("Line searches:        {:d}".format(len(line_search_trajectory)))
print(open("data/{:s}_{:d}.log".format(mode, level), "r").read())
