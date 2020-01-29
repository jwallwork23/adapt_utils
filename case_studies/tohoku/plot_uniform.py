import firedrake

import matplotlib
import matplotlib.pyplot as plt

from adapt_utils.case_studies.tohoku.options import TohokuOptions


matplotlib.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
matplotlib.rc('text', usetex=True)

# Setup Tohoku domain
op = TohokuOptions(utm=False)
mesh = op.default_mesh
lon, lat, elev = op.read_bathymetry_file(km=True)
fig, axes = plt.subplots(nrows=1, ncols=2, sharex=True)

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
plt.savefig('outputs/uniform_bathymetry_{:d}.pdf'.format(mesh.num_cells()))

# Setup Tohoku domain
lon, lat, elev = op.read_surface_file()
fig, axes = plt.subplots(nrows=1, ncols=2, sharex=True)

# Plot raw surface data
ax1 = axes.flat[0]
cs = ax1.contourf(lon, lat, elev, 50, cmap=matplotlib.cm.coolwarm)
ax1.set_xlabel("Degrees longitude")
ax1.set_ylabel("Degrees latitude")
ax1.set_title("Original data")

# Plot surface data interpolated onto a uniform mesh
ax2 = axes.flat[1]
firedrake.plot(op.initial_surface, cmap=matplotlib.cm.coolwarm, axes=ax2)
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
plt.savefig('outputs/uniform_ic_{:d}.pdf'.format(mesh.num_cells()))

fig = plt.figure()
axes = fig.gca()
P1 = op.bathymetry.function_space()
axes, cb = firedrake.plot(op.set_coriolis(P1), axes=axes, cmap=matplotlib.cm.coolwarm, colorbar=True)
axes.set_xlabel("Degrees longitude")
axes.set_ylabel("Degrees latitude")
axes.set_xlim(xlim)
axes.set_ylim(ylim)
cb.set_label(r"Coriolis parameter ($\times10^{-5}$)")
cb.set_ticks([0.000075, 0.000080, 0.000085, 0.000090, 0.000095, 0.000100])
cb.ax.set_yticklabels(['{:.1f}'.format(t*1.0e+5) for t in cb.get_ticks()])
plt.savefig('outputs/uniform_coriolis_{:d}.pdf'.format(mesh.num_cells()))
