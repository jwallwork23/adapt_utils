import firedrake
from thetis import print_output

import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from adapt_utils.swe.spacetime.solver import SpaceTimeShallowWaterProblem
from adapt_utils.test_cases.tsunami1d.options import Tsunami1dOptions


matplotlib.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
matplotlib.rc('text', usetex=True)

# Parameters
debug = True
plot_pdf = False
plot_pvd = True
save_hdf5 = True
approach = 'dwr'

# Spatial discretisation
n = 250
# n = 2000  # (Value used in original paper)
dx = 1/n

# Time discretisation
celerity = 20.0*np.sqrt(9.81)
# dt = 40.0e+3*dx/celerity
dt = 3.0
# dt = 1.0  # (Value used in original paper)

# NOTE: Forward and adjoint relatively stable with n = 500 and dt = 1.5
op = Tsunami1dOptions(debug=debug, nx=n, dt=dt, approach=approach,
                      save_hdf5=save_hdf5, plot_pvd=plot_pvd,
                      horizontal_length_scale=1000.0, time_scale=10.0)
op.h_min = 100.0/op.L
op.h_max = 100.0e+3/op.L
op.target = 10000.0
op.num_adapt = 1
# op.norm_order = 1
# op.normalisation = 'error'

swp = SpaceTimeShallowWaterProblem(op, discrete_adjoint=False, levels=1)
swp.setup_solver_forward()
swp.solve_forward()
print_output("QoI before adaptation: {:.4e}".format(op.evaluate_qoi(swp.solution)))
swp.setup_solver_adjoint()
swp.solve_adjoint()
adjoint = 'adjoint' in op.approach
swp.dwr_indication(adjoint=adjoint)
if 'both' in op.approach:
    swp.dwr_indication(adjoint=not adjoint)
    swp.indicators['dwr_cell_both'] = swp.indicators['dwr_cell'].copy(deepcopy=True)
    swp.indicators['dwr_cell_both'] /= firedrake.norm(swp.indicators['dwr_cell'])
    swp.indicators['dwr_cell_both'] += swp.indicators['dwr_cell_adjoint']/firedrake.norm(swp.indicators['dwr_cell_adjoint'])
    swp.indicators['dwr_flux_both'] = swp.indicators['dwr_flux'].copy(deepcopy=True)
    swp.indicators['dwr_flux_both'] /= firedrake.norm(swp.indicators['dwr_flux'])
    swp.indicators['dwr_flux_both'] += swp.indicators['dwr_flux_adjoint']/firedrake.norm(swp.indicators['dwr_flux_adjoint'])
    swp.indicator = swp.indicators['dwr_cell_both'].copy(deepcopy=True)
    swp.indicator += swp.indicators['dwr_flux_both']
swp.indicator = firedrake.interpolate(abs(swp.indicator), swp.P0)
swp.plot()

v = 0.02
L = op.L  # Horizontal length scale
T = op.T  # Time scale

# Plot error estimator
fig = plt.figure(figsize=(3.2, 4.8))
ax = fig.add_subplot(111)
firedrake.plot(swp.indicator, axes=ax, vmin=v, vmax=1.001*v, cmap=matplotlib.cm.Greens)
# firedrake.plot(swp.indicator, axes=ax, cmap=matplotlib.cm.Greens)
ax.invert_xaxis()
ymin = op.start_time
ymax = op.end_time
plt.xlabel("Kilometres offshore")
plt.ylabel("Hours")
plt.tight_layout()
plt.axvline(50e+3/L, ymin=ymin, ymax=ymax, linestyle='--', color='k')
plt.xlim([400e+3/L, 0.0/L])
plt.ylim([ymin, ymax])
plt.xticks([50e+3/L, 150e+3/L, 250e+3/L, 350e+3/L], ["50", "150", "250", "350"])
plt.yticks([], [])
fname = os.path.join(op.di, "{:s}_{:d}".format(op.approach, n))
plt.savefig(fname + ".png")
if plot_pdf:
    plt.savefig(fname + ".pdf")

# Adapt mesh
swp.get_isotropic_metric()
swp.adapt_mesh()
