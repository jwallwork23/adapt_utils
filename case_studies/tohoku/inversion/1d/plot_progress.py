import matplotlib.pyplot as plt
import numpy as np
import scipy.interpolate as si

from adapt_utils.plotting import *


level = 1
mode = 'discrete'
# mode = 'continuous'

control_trajectory = np.load('data/opt_progress_{:s}_{:d}_ctrl.npy'.format(mode, level))
functional_trajectory = np.load('data/opt_progress_{:s}_{:d}_func.npy'.format(mode, level))
gradient_trajectory = np.load('data/opt_progress_{:s}_{:d}_grad.npy'.format(mode, level))


fig, axes = plt.subplots(figsize=(8, 8))
l = si.lagrange(control_trajectory[:3], functional_trajectory[:3])
dl = l.deriv()
print("Exact gradient at 10.0: {:.4f}".format(dl(10.0)))
print("Exact gradient at  5.0: {:.4f}".format(dl(5.0)))
l_min = -dl.coefficients[1]/dl.coefficients[0]
print("Minimiser of quadratic: {:.4f}".format(l_min))
xx = np.linspace(0, 10, 100)
axes.plot(xx, l(xx), '--x', color='C0', markevery=10)
delta_m = 0.25
for m, f, g in zip(control_trajectory, functional_trajectory, gradient_trajectory):
    x = np.array([m - delta_m, m + delta_m])
    axes.plot(x, g*(x-m) + f, '-', color='C2', linewidth=3)
axes.plot(control_trajectory, functional_trajectory, 'o', color='C1', markersize=8)
axes.set_xlabel(r"Control parameter, $m$")
axes.set_ylabel("QoI")
axes.grid(True)
plt.tight_layout()
plt.show()
# plt.savefig("plots/opt_progress_discrete_{:d}.pdf".format(level))
