import firedrake

import matplotlib
import matplotlib.pyplot as plt

from adapt_utils.case_studies.tohoku.options import TohokuOptions


matplotlib.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
matplotlib.rc('text', usetex=True)

# Setup Tohoku domain
op = TohokuOptions(utm=False, offset=0, n=40)
mesh = op.default_mesh
lon, lat, elev = op.read_bathymetry_file(km=True)
fig = plt.figure()
ax = fig.add_subplot(111)

# Plot raw bathymetry data
cs = ax.contourf(lon, lat, elev, 50, vmin=-9, vmax=2, cmap=matplotlib.cm.coolwarm)
ax.contour(lon, lat, elev, vmin=-0.01, vmax=0.01, levels=0, colors='k')
ax.set_xlabel("Degrees longitude")
ax.set_ylabel("Degrees latitude")
# ax.set_title("Original data")
op.annotate_plot(ax, gauges=True)
xlim = ax.get_xlim()
ylim = ax.get_ylim()

# Save raw bathymetry and uniform interpolant
cb = fig.colorbar(cs)
cb.set_label(r"Bathymetry $[\mathrm{km}]$")
fname = 'outputs/bathymetry'
plt.savefig(fname + '.png')
plt.savefig(fname + '.pdf')

# Setup Tohoku domain
lon1, lat1, elev1 = op.read_surface_file()
fig = plt.figure()
ax = fig.add_subplot(111)

# Plot raw surface data
cs = ax.contourf(lon1, lat1, elev1, 50, cmap=matplotlib.cm.coolwarm)
ax.contour(lon, lat, elev, vmin=-0.01, vmax=0.01, levels=0, colors='k')
ax.set_xlabel("Degrees longitude")
ax.set_ylabel("Degrees latitude")
ax.set_xlim(xlim)
ax.set_ylim(ylim)
# op.annotate_plot(ax, gauges=False)
# ax.set_title("Original data")

# Save raw surface and uniform interpolant
cb = fig.colorbar(cs)
cb.set_label("Initial free surface $[\mathrm m]$")
fname = 'outputs/ic'
plt.savefig(fname + '.png')
plt.savefig(fname + '.pdf')
