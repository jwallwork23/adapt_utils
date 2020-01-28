import firedrake
from thetis import create_directory

import os
import matplotlib
import numpy as np

from adapt_utils.tsunami.options import TohokuOptions


matplotlib.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
matplotlib.rc('text', usetex=True)

# Setup Tohoku domain
op = TohokuOptions(utm=False)
mesh = op.default_mesh
lon, lat, elev = op.read_bathymetry_file(km=True)
fig, axes = matplotlib.pyplot.subplots(nrows=1, ncols=2, sharex=True)

# Plot raw bathymetry data
ax1 = axes.flat[0]
cs = ax1.contourf(lon, lat, elev, 50, vmin=-9, vmax=2, cmap=matplotlib.cm.coolwarm)
ax1.set_xlabel("Degrees longitude")
ax1.set_ylabel("Degrees latitude")
ax1.set_title("Original data")
xlim = ax1.get_xlim()
ylim = ax1.get_ylim()

# Plot bathymetry data interpolated onto a uniform mesh
ax2 = axes.flat[1]
firedrake.plot(op.bathymetry, cmap=matplotlib.cm.coolwarm, axes=ax2)
ax2.set_xlabel("Degrees longitude")
ax2.set_ylabel("Degrees latitude")
ax2.set_xlim(xlim)
ax2.set_ylim(ylim)
ax2.yaxis.set_label_position("right")
ax2.yaxis.tick_right()
ax2.set_title("Uniform mesh interpolant")

# Save raw bathymetry and uniform interpolant
cb = fig.colorbar(cs, orientation='horizontal', ax=axes.ravel().tolist(), pad=0.2)
cb.set_label("Bathymetry $[\mathrm k\mathrm m]$")
di = create_directory('outputs/tohoku')
matplotlib.pyplot.savefig(os.path.join(di, 'uniform_bathymetry_{:d}.pdf'.format(mesh.num_cells())))

# Setup Tohoku domain
lon, lat, elev = op.read_surface_file()
fig, axes = matplotlib.pyplot.subplots(nrows=1, ncols=2, sharex=True)

# Plot raw surface data
ax1 = axes.flat[0]
cs = ax1.contourf(lon, lat, elev, 50, cmap=matplotlib.cm.coolwarm)
ax1.set_xlabel("Degrees longitude")
ax1.set_ylabel("Degrees latitude")
ax1.set_title("Original data")
xlim = ax1.get_xlim()
ylim = ax1.get_ylim()

# Plot surface data interpolated onto a uniform mesh
ax2 = axes.flat[1]
firedrake.plot(op.initial_value, cmap=matplotlib.cm.coolwarm, axes=ax2)
ax2.set_xlabel("Degrees longitude")
ax2.set_ylabel("Degrees latitude")
ax2.set_xlim(xlim)
ax2.set_ylim(ylim)
ax2.yaxis.set_label_position("right")
ax2.yaxis.tick_right()
ax2.set_title("Uniform mesh interpolant")

# Save raw surface and uniform interpolant
cb = fig.colorbar(cs, orientation='horizontal', ax=axes.ravel().tolist(), pad=0.2)
cb.set_label("Initial free surface $[\mathrm m]$")
di = create_directory('outputs/tohoku')
matplotlib.pyplot.savefig(os.path.join(di, 'uniform_ic_{:d}.pdf'.format(mesh.num_cells())))
