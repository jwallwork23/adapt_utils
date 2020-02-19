import firedrake
from thetis import print_output

import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import argparse

from adapt_utils.swe.spacetime.solver import SpaceTimeShallowWaterProblem
from adapt_utils.test_cases.tsunami1d.options import Tsunami1dOptions


# TODO: put more parameters in parser
parser = argparse.ArgumentParser()
parser.add_argument("-n", help="Number of points in spatial dimension")
parser.add_argument("-dt", help="Timestep")
parser.add_argument("-end_time", help="End time of simulation")
parser.add_argument("-plot_png")
parser.add_argument("-plot_pdf")
parser.add_argument("-forward", help="Solve forward problem")
parser.add_argument("-adjoint", help="Solve adjoint problem")
parser.add_argument("-quads", help="Use quadrilateral elements")
args = parser.parse_args()

debug = True
n = int(args.n or 2000)     # (Value used in original paper)
dt = float(args.dt or 1.0)  # (Value used in original paper)
end_time = float(args.end_time or 4200.0)
plot_png = bool(args.plot_png)
plot_pdf = bool(args.plot_pdf)
forward = bool(args.forward)
adjoint = bool(args.adjoint)
assert forward or adjoint
quads = bool(args.quads)
L = 1000.0  # Horizontal length scale
T = 10.0    # Time scale
v = 0.02    # For plotting

if plot_png or plot_pdf:
    matplotlib.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
    matplotlib.rc('text', usetex=True)

# NOTE: Forward and adjoint relatively stable with:
#   * n = 500, dt = 1.5 and quads = False
#   * n = 1000, dt = 3.0 and quads = False
op = Tsunami1dOptions(debug=debug, nx=n, dt=dt, end_time=end_time, plot_pvd=True,
                      horizontal_length_scale=L, time_scale=T, quads=quads)
swp = SpaceTimeShallowWaterProblem(op, discrete_adjoint=False)

if forward:
    # Solve forward problem
    swp.setup_solver_forward()
    swp.solve_forward()
    print_output("QoI: {:.4f}".format(op.evaluate_qoi(swp.solution)))
    eta = swp.solution.split()[1]

    # Plot forward
    if plot_png or plot_pdf:
        fig = plt.figure(figsize=(3.2, 4.8))
        ax = fig.add_subplot(111)
        firedrake.plot(firedrake.interpolate(abs(eta), eta.function_space()), axes=ax, vmin=v, vmax=1.001*v, cmap=matplotlib.cm.Reds)
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
        if plot_png:
           plt.savefig(fname + ".png")
        if plot_pdf:
            plt.savefig(fname + ".pdf")

if adjoint:
    # Solve adjoint problem
    swp.setup_solver_adjoint()
    swp.solve_adjoint()
    zeta = swp.adjoint_solution.split()[1]

    # Plot adjoint
    if plot_png or plot_pdf:
        fig = plt.figure(figsize=(3.2, 4.8))
        ax = fig.add_subplot(111)
        firedrake.plot(firedrake.interpolate(abs(zeta), zeta.function_space()), axes=ax, vmin=v, vmax=1.001*v, cmap=matplotlib.cm.Blues)
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
        if plot_png:
            plt.savefig(fname + ".png")
        if plot_pdf:
            plt.savefig(fname + ".pdf")

if forward and adjoint:
    # Take inner product of forward and adjoint solutions
    swp.dwp_indication()
    swp.plot()
    dwp = swp.indicator

    # Plot inner product
    if plot_png or plot_pdf:
        fig = plt.figure(figsize=(3.2, 4.8))
        ax = fig.add_subplot(111)
        firedrake.plot(firedrake.interpolate(abs(dwp), dwp.function_space()), axes=ax, vmin=v, vmax=1.001*v, cmap=matplotlib.cm.Greens)
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
        if plot_png:
            plt.savefig(fname + ".png")
        if plot_pdf:
            plt.savefig(fname + ".pdf")
