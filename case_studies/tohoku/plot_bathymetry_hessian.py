import firedrake

import matplotlib as mpl
import matplotlib.pyplot as plt

from adapt_utils.case_studies.tohoku.options import TohokuOptions
from adapt_utils.swe.tsunami.solver import TsunamiProblem


mpl.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
mpl.rc('text', usetex=True)

# Setup Tohoku domain
op = TohokuOptions(utm=False)
lon, lat, elev = op.read_bathymetry_file(km=True)
cs = mpl.pyplot.contourf(lon, lat, elev, 50, vmin=-9, vmax=2, cmap=mpl.cm.coolwarm)  # Get colorbar
xlim = plt.gca().get_xlim()
ylim = plt.gca().get_ylim()
fig, axes = mpl.pyplot.subplots(nrows=1, ncols=2, sharex=True)

# Adapt mesh to Hessian of bathymetry
op.target = 1e3
tp = TsunamiProblem(op, levels=0)
tp.initialise_mesh(num_adapt=4, approach='hessian', adapt_field='bathymetry')

# Plot adapted mesh
ax1 = axes.flat[0]
ax1 = firedrake.plot(firedrake.Function(tp.P1), axes=ax1, colorbar=False, cmap=mpl.cm.binary, edgecolors='dimgray')
ax1.set_xlabel("Degrees longitude")
ax1.set_ylabel("Degrees latitude")
ax1.set_xlim(xlim)
ax1.set_ylim(ylim)
ax1.set_title("Adapted mesh")

# Plot bathymetry data interpolated onto adapted mesh
ax2 = axes.flat[1]
ax2 = firedrake.plot(op.bathymetry, axes=ax2, colorbar=False)
ax2.set_xlabel("Degrees longitude")
ax2.set_ylabel("Degrees latitude")
ax2.set_xlim(xlim)
ax2.set_ylim(ylim)
ax2.yaxis.set_label_position("right")
ax2.yaxis.tick_right()
ax2.set_title("Adapted mesh interpolant")

# Save adapted mesh and interpolant
cb = fig.colorbar(cs, orientation='horizontal', ax=axes.ravel().tolist(), pad=0.2)
xlim = plt.gca().get_xlim()
ylim = plt.gca().get_ylim()
cb.set_label("Bathymetry $[\mathrm k\mathrm m]$")
mpl.pyplot.savefig('outputs/metric_adapt_bathymetry_{:d}.pdf'.format(tp.num_cells[-1]))

# New figure for initial free surface
lon, lat, elev = op.read_surface_file()
cs = mpl.pyplot.contourf(lon, lat, elev, 50, cmap=mpl.cm.coolwarm)  # Get colorbar
fig, axes = mpl.pyplot.subplots(nrows=1, ncols=2, sharex=True)

# Plot adapted mesh
ax1 = axes.flat[0]
ax1 = firedrake.plot(firedrake.Function(tp.P1), axes=ax1, colorbar=False, cmap=mpl.cm.binary, edgecolors='dimgray')
ax1.set_xlabel("Degrees longitude")
ax1.set_ylabel("Degrees latitude")
ax1.set_xlim(xlim)
ax1.set_ylim(ylim)
ax1.set_title("Adapted mesh")

# Plot bathymetry data interpolated onto adapted mesh
ax2 = axes.flat[1]
ax2 = firedrake.plot(op.initial_surface, axes=ax2, colorbar=False)
ax2.set_xlabel("Degrees longitude")
ax2.set_ylabel("Degrees latitude")
ax2.set_xlim(xlim)
ax2.set_ylim(ylim)
ax2.yaxis.set_label_position("right")
ax2.yaxis.tick_right()
ax2.set_title("Adapted mesh interpolant")

# Save adapted mesh and interpolant
cb = fig.colorbar(cs, orientation='horizontal', ax=axes.ravel().tolist(), pad=0.2)
cb.set_label("Initial free surface $[\mathrm m]$")
mpl.pyplot.savefig('outputs/metric_adapt_ic_{:d}.pdf'.format(tp.num_cells[-1]))
