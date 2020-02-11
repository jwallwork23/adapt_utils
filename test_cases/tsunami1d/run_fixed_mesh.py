import firedrake

import os
import matplotlib
import matplotlib.pyplot as plt

from adapt_utils.swe.spacetime.solver import SpaceTimeShallowWaterProblem
from adapt_utils.test_cases.tsunami1d.options import Tsunami1dOptions


matplotlib.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
matplotlib.rc('text', usetex=True)

# Parameters
debug = True
plot_pdf = False
n = 50
# n = 500
# n = 2000  # (Value used in original paper)
dt = 2000/n
op = Tsunami1dOptions(debug=debug, nx=500, dt=dt)

# Solve forward problem
swp = SpaceTimeShallowWaterProblem(op, discrete_adjoint=True)
swp.setup_solver_forward()
#swp.solve_forward()

# Solve adjoint problem
swp.setup_solver_adjoint()  # TODO: Try continuous adjoint
swp.solve_adjoint()
exit(0)  # TODO: temp

# Plot forward
fig = plt.figure(figsize=(3.2, 4.8))
ax = fig.add_subplot(111)
eta = swp.solution.split()[1]
firedrake.plot(firedrake.interpolate(abs(eta), eta.function_space()), axes=ax, vmin=0.1, vmax=0.1001, cmap=matplotlib.cm.Reds)
ax.invert_xaxis()
ymin = 0.0
ymax = op.end_time
plt.xlabel("Kilometres offshore")
plt.ylabel("Hours")
plt.tight_layout()
plt.axvline(50e+3, ymin=ymin, ymax=ymax, linestyle='--', color='k')
plt.xlim([400e+3, 0.0])
plt.ylim([ymin, ymax])
plt.xticks([50e+3, 150e+3, 250e+3, 350e+3], ["50", "150", "250", "350"])
plt.yticks([1800.0, 3600.0], ["0.5", "1.0"])
fname = os.path.join(op.di, "forward_{:d}".format(n))
plt.savefig(fname + ".png")
if plot_pdf:
    plt.savefig(fname + ".pdf")

# Plot adjoint
fig = plt.figure(figsize=(3.2, 4.8))
ax = fig.add_subplot(111)
zeta = swp.adjoint_solution.split()[1]
firedrake.plot(firedrake.interpolate(abs(zeta), zeta.function_space()), axes=ax, vmin=0.1, vmax=0.1001, cmap=matplotlib.cm.Blues)
ax.invert_xaxis()
ymin = 0.0
ymax = op.end_time
plt.xlabel("Kilometres offshore")
plt.ylabel("Hours")
plt.tight_layout()
plt.axvline(50e+3, ymin=ymin, ymax=ymax, linestyle='--', color='k')
plt.xlim([400e+3, 0.0])
plt.ylim([ymin, ymax])
plt.xticks([50e+3, 150e+3, 250e+3, 350e+3], ["50", "150", "250", "350"])
plt.yticks([], [])
fname = os.path.join(op.di, "adjoint_{:d}".format(n))
plt.savefig(fname + ".png")
if plot_pdf:
    plt.savefig(fname + ".pdf")
