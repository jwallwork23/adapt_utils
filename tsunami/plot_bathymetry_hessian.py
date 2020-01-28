import firedrake

import matplotlib as mpl
import numpy as np

from adapt_utils.tsunami.options import TohokuOptions
from adapt_utils.adapt.metric import steady_metric


mpl.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
mpl.rc('text', usetex=True)

# Setup Tohoku domain
op = TohokuOptions(utm=False)
mesh = op.default_mesh
lon, lat, elev = op.read_bathymetry_file(km=True)
cs = mpl.pyplot.contourf(lon, lat, elev, 50, vmin=-9, vmax=2, cmap=mpl.cm.coolwarm)  # Get colorbar
fig, axes = mpl.pyplot.subplots(nrows=1, ncols=2, sharex=True)

# Adapt mesh to Hessian of bathymetry
op.target = 1e3
op.num_adapt = 4
op.adapt_to_bathymetry_hessian()
P1 = op.bathymetry.function_space()
mesh = op.default_mesh

# Plot adapted mesh
ax1 = axes.flat[0]
ax1 = firedrake.plot(firedrake.Function(P1), axes=ax1, colorbar=False, cmap=mpl.cm.binary, edgecolors='dimgray')
ax1.set_xlabel("Degrees longitude")
ax1.set_ylabel("Degrees latitude")
ax1.set_title("Adapted mesh")

# Plot bathymetry data interpolated onto adapted mesh
ax2 = axes.flat[1]
ax2 = firedrake.plot(op.bathymetry, axes=ax2, colorbar=False)
ax2.set_xlabel("Degrees longitude")
ax2.set_ylabel("Degrees latitude")
ax2.set_title("Adapted mesh interpolant")

# Save adapted mesh and interpolant
cb = fig.colorbar(cs, orientation='horizontal', ax=axes.ravel().tolist(), pad=0.2)
cb.set_label("Bathymetry $[\mathrm k\mathrm m]$")
mpl.pyplot.savefig('outputs/metric_adapt_bathymetry_{:d}.pdf'.format(mesh.num_cells()))

# New figure for initial free surface
lon, lat, elev = op.read_surface_file()
cs = mpl.pyplot.contourf(lon, lat, elev, 50, cmap=mpl.cm.coolwarm)  # Get colorbar
fig, axes = mpl.pyplot.subplots(nrows=1, ncols=2, sharex=True)

# Plot adapted mesh
ax1 = axes.flat[0]
ax1 = firedrake.plot(firedrake.Function(P1), axes=ax1, colorbar=False, cmap=mpl.cm.binary, edgecolors='dimgray')
ax1.set_xlabel("Degrees longitude")
ax1.set_ylabel("Degrees latitude")
ax1.set_title("Adapted mesh")

# Plot bathymetry data interpolated onto adapted mesh
ax2 = axes.flat[1]
ax2 = firedrake.plot(op.initial_surface, axes=ax2, colorbar=False)
ax2.set_xlabel("Degrees longitude")
ax2.set_ylabel("Degrees latitude")
ax2.set_title("Adapted mesh interpolant")

# Save adapted mesh and interpolant
cb = fig.colorbar(cs, orientation='horizontal', ax=axes.ravel().tolist(), pad=0.2)
cb.set_label("Initial free surface $[\mathrm m]$")
mpl.pyplot.savefig('outputs/metric_adapt_ic_{:d}.pdf'.format(mesh.num_cells()))
