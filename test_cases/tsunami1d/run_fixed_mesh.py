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
# n = 50
n = 500
# n = 2000  # (Value used in original paper)
dt = 2000/n

# Solve problem
op = Tsunami1dOptions(debug=debug, nx=500, dt=dt)
swp = SpaceTimeShallowWaterProblem(op)
swp.setup_solver_forward()
swp.solve_forward()

# Plot thresholded
fig = plt.figure(figsize=(3.2, 4.8))
ax = fig.add_subplot(111)
eta = swp.solution.split()[1]
firedrake.plot(firedrake.interpolate(abs(eta), eta.function_space()), axes=ax, vmin=0.1, vmax=0.1001, cmap=matplotlib.cm.Reds)
ax.invert_xaxis()
ymin = 0.0
ymax = op.end_time
plt.tight_layout()
plt.axvline(50e+3, ymin=ymin, ymax=ymax, linestyle='--', color='k')
plt.xlim([400e+3, 0.0])
plt.ylim([ymin, ymax])
plt.xticks([50e+3, 150e+3, 250e+3, 350e+3], ["50", "150", "250", "350"])
plt.yticks([1800.0, 3600.0], ["0.5", "1.0"])
plt.xlabel("Kilometres offshore")
plt.ylabel("Hours")
# plt.savefig(os.path.join(op.di, "forward_{:d}.pdf".format(n)))
plt.savefig(os.path.join(op.di, "forward_{:d}.png".format(n)))
