import firedrake
from thetis import print_output

import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from adapt_utils.swe.spacetime.solver import SpaceTimeShallowWaterProblem
from adapt_utils.swe.tsunami.tsunami1d.options import Tsunami1dOptions


matplotlib.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
matplotlib.rc('text', usetex=True)

# Parameters
debug = True
plot_pdf = False
plot_pvd = True
forward = True
adjoint = True
L = 1000.0
T = 10.0
v = 0.02

n = 500
# n = 2000  # (Value used in original paper)
dt = 1.5
# dt = 1.0  # (Value used in original paper)

# NOTE: Forward and adjoint relatively stable with n = 500 and dt = 1.5
op = Tsunami1dOptions(debug=debug, approach='dwp', nx=n, dt=dt, plot_pvd=plot_pvd,
                      horizontal_length_scale=L, time_scale=T)
op.h_min = 100.0/L
op.h_max = 100.0e+3/L
op.target = 10000.0
op.num_adapt = 1
# op.norm_order = 1
# op.normalisation = 'error'

swp = SpaceTimeShallowWaterProblem(op, discrete_adjoint=False, levels=0)
swp.setup_solver_forward()
swp.solve_forward()
print_output("QoI before adaptation: {:.4e}".format(op.evaluate_qoi(swp.solution)))
swp.setup_solver_adjoint()
swp.solve_adjoint()
swp.dwp_indication()
swp.indicator.interpolate(abs(swp.indicator))
swp.get_isotropic_metric()
swp.adapt_mesh()

# FIXME: Solution of equations on new mesh

if forward:
    # Solve forward problem
    swp.setup_solver_forward()
    swp.solve_forward()
    print_output("QoI after adaptation: {:.4e}".format(op.evaluate_qoi(swp.solution)))
    eta = swp.solution.split()[1]

    # Plot forward
    fig = plt.figure(figsize=(3.2, 4.8))
    ax = fig.add_subplot(111)
    # firedrake.plot(firedrake.interpolate(abs(eta), eta.function_space()), axes=ax, vmin=v, vmax=v+0.0001, cmap=matplotlib.cm.Reds)
    firedrake.plot(firedrake.interpolate(abs(eta), eta.function_space()), axes=ax, cmap=matplotlib.cm.Reds)
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
    plt.yticks([1800.0/T, 3600.0/T], ["0.5", "1.0"])
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
    # firedrake.plot(firedrake.interpolate(abs(zeta), zeta.function_space()), axes=ax, vmin=v, vmax=v+0.0001, cmap=matplotlib.cm.Blues)
    firedrake.plot(firedrake.interpolate(abs(zeta), zeta.function_space()), axes=ax, cmap=matplotlib.cm.Blues)
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
    fname = os.path.join(op.di, "adjoint_{:d}".format(n))
    plt.savefig(fname + ".png")
    if plot_pdf:
        plt.savefig(fname + ".pdf")

if forward and adjoint:
    # Take inner product of forward and adjoint solutions
    swp.dwp_indication()
    swp.plot()
    dwp = swp.indicator

    # Plot inner product
    fig = plt.figure(figsize=(3.2, 4.8))
    ax = fig.add_subplot(111)
    # firedrake.plot(firedrake.interpolate(abs(dwp), dwp.function_space()), axes=ax, vmin=v, vmax=v+0.0001, cmap=matplotlib.cm.Greens)
    firedrake.plot(firedrake.interpolate(abs(dwp), dwp.function_space()), axes=ax, cmap=matplotlib.cm.Greens)
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
    fname = os.path.join(op.di, "dwp_{:d}".format(n))
    plt.savefig(fname + ".png")
    if plot_pdf:
        plt.savefig(fname + ".pdf")
