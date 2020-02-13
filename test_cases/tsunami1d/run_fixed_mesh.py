import firedrake

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
forward = False
adjoint = True

# Spatial discretisation
n = 500
# n = 2000  # (Value used in original paper)
dx = 1/n

# Time discretisation
celerity = 20.0*np.sqrt(9.81)
# dt = 2000.0*dx/celerity
dt = 2.0
# dt = 1.0  # (Value used in original paper)

# NOTE: Forward and adjoint relatively stable with n = 500 and dt = 2
op = Tsunami1dOptions(debug=debug, nx=n, dt=dt, save_hdf5=save_hdf5, plot_pvd=plot_pvd)
swp = SpaceTimeShallowWaterProblem(op, discrete_adjoint=False)

if forward:
    # Solve forward problem
    swp.setup_solver_forward()
    swp.solve_forward()
    eta = swp.solution.split()[1]

    # Plot forward
    fig = plt.figure(figsize=(3.2, 4.8))
    ax = fig.add_subplot(111)
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

if adjoint:
    # Solve adjoint problem
    swp.setup_solver_adjoint()
    swp.solve_adjoint()
    zeta = swp.adjoint_solution.split()[1]

    # Plot adjoint
    fig = plt.figure(figsize=(3.2, 4.8))
    ax = fig.add_subplot(111)
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

if forward and adjoint:
    # Take inner product of forward and adjoint solutions
    swp.dwp_indication()
    dwp = swp.indicator

    # Plot inner product
    fig = plt.figure(figsize=(3.2, 4.8))
    ax = fig.add_subplot(111)
    firedrake.plot(firedrake.interpolate(abs(dwp), dwp.function_space()), axes=ax, vmin=0.1, vmax=0.1001, cmap=matplotlib.cm.Greens)
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
    fname = os.path.join(op.di, "dwp_{:d}".format(n))
    plt.savefig(fname + ".png")
    if plot_pdf:
        plt.savefig(fname + ".pdf")
