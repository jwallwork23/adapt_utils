from firedrake import *

import matplotlib.pyplot as plt

from adapt_utils.interpolation import point_interpolate
from adapt_utils.plotting import *


plot = True

# Setup two function spaces and a source
Hs = UnitSquareMesh(20, 25, diagonal='left')
Ht = UnitSquareMesh(20, 20, diagonal='right')
Vs = FunctionSpace(Hs, "CG", 1)
Vt = FunctionSpace(Ht, "CG", 1)
xs, ys = SpatialCoordinate(Hs)
s_init = Function(Vs, name="Initial source")
s_init.interpolate(sin(pi*xs)*sin(pi*ys))
s = Function(Vs, name="Source").assign(s_init)
t = Function(Vt, name="Target")

# Ping pong test
N = 100
mass_init = assemble(s_init*dx)
l2_error = []
mass_error = []
s.assign(s_init)
for i in range(N):
    point_interpolate(s, t)
    point_interpolate(t, s)
    l2_error.append(errornorm(s, s_init)/norm(s_init))
    mass_error.append(abs(assemble(s*dx) - mass_init)/abs(mass_init))
assert np.isclose(l2_error[-2], l2_error[-1])
assert np.isclose(mass_error[-2], mass_error[-1])

# Plot results
if plot:
    fig, axes = plt.subplots(figsize=(7, 4))
    axes.plot(100*np.array(l2_error))
    axes.set_xlabel("Number of interpolation steps")
    axes.set_ylabel(r"$\mathcal L_2$ error (\%)")
    axes.grid(True)
    savefig("l2_error_interpolation", "plots", ["pdf"])
    fig, axes = plt.subplots(figsize=(7, 4))
    axes.plot(100*np.array(mass_error))
    axes.set_xlabel("Number of interpolation steps")
    axes.set_ylabel(r"Mass error (\%)")
    axes.grid(True)
    savefig("mass_error_interpolation", "plots", ["pdf"])
    fig, axes = plt.subplots(figsize=(3.5, 3))
    cbar = fig.colorbar(tricontourf(s_init, axes=axes, levels=np.linspace(-1.0e-06, 1.05, 8)), ax=axes)
    cbar.set_ticks(np.linspace(0, 1.05, 8))
    cbar.ax.tick_params(labelsize=10)
    axes.axis(False)
    savefig("source_for_interpolation", "plots", ["jpg"])
    fig, axes = plt.subplots(figsize=(3.5, 3))
    cbar = fig.colorbar(tricontourf(s, axes=axes, levels=np.linspace(-1.0e-06, 1.05, 8)), ax=axes)
    cbar.set_ticks(np.linspace(0, 1.05, 8))
    cbar.ax.tick_params(labelsize=10)
    axes.axis(False)
    savefig("after_100_interpolations", "plots", ["jpg"])
