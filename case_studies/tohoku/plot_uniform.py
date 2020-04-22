import firedrake
from thetis import print_output

import matplotlib
import matplotlib.pyplot as plt

from adapt_utils.case_studies.tohoku.options import TohokuOptions
from adapt_utils.swe.tsunami.conversion import lonlat_to_utm


# matplotlib.rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})  # FIXME
# matplotlib.rc('text', usetex=True)  # FIXME

# Setup Tohoku domain
op = TohokuOptions(level=0)
mesh = op.default_mesh
lon1, lat1, elev1 = op.read_surface_file()
lon, lat, elev = op.read_bathymetry_file()
x, y = lonlat_to_utm(lon, lat, op.force_zone_number)

# Plot raw bathymetry data  FIXME
print_output("Plotting raw bathymetry...")
fig = plt.figure(figsize=(5, 5))
ax = plt.gca()
cs = ax.contourf(lon, lat, elev, 50, cmap=matplotlib.cm.coolwarm)
ax.contour(lon, lat, elev, vmin=-0.01, vmax=0.01, levels=0, colors='k')
ax.set_xlabel("Degrees longitude")
ax.set_ylabel("Degrees latitude")
ax.set_title("Original data")
op.annotate_plot(ax, gauges=False, coords="lonlat")
cb = fig.colorbar(cs)
cb.set_label("Bathymetry [m]")
ax.set_xlim([136, 150])
ax.set_ylim([30, 42])
fname = 'outputs/bathymetry_data'
plt.savefig(fname + '.png')
# plt.savefig(fname + '.pdf')

# Plot bathymetry data interpolated onto a uniform mesh
print_output("Plotting interpolated bathymetry...")
fig = plt.figure(figsize=(5, 5))
ax = plt.gca()
op.bathymetry *= -1
cs = firedrake.tricontourf(op.bathymetry, 50, cmap=matplotlib.cm.coolwarm, axes=ax)
ax.contour(x, y, elev, vmin=-0.01, vmax=0.01, levels=0, colors='k')
ax.set_xlabel("UTM x-coordinate [m]")
ax.set_ylabel("UTM y-coordinate [m]")
# op.annotate_plot(ax, gauges=True, coords="utm")
ax.set_title("Interpolant")
cb = fig.colorbar(cs)
cb.set_label("Bathymetry [m]")
fname = 'outputs/bathymetry_{:d}'.format(mesh.num_cells())
plt.savefig(fname + '.png')
# plt.savefig(fname + '.pdf')

# Plot raw surface data
fig = plt.figure(figsize=(5, 5))
ax = plt.gca()
print_output("Plotting raw initial surface...")
cs = ax.contourf(lon1, lat1, elev1, 50, cmap=matplotlib.cm.coolwarm)
ax.contour(lon, lat, elev, vmin=-0.01, vmax=0.01, levels=0, colors='k')
ax.set_xlabel("Degrees longitude")
ax.set_ylabel("Degrees latitude")
ax.set_xlim([136, 150])
ax.set_ylim([30, 42])
# op.annotate_plot(ax, gauges=False)
ax.set_title("Original data")
cb = fig.colorbar(cs)
cb.set_label("Initial free surface [m]")
fname = 'outputs/ic_data'
plt.savefig(fname + '.png')
# plt.savefig(fname + '.pdf')

# Plot surface data interpolated onto a uniform mesh
print_output("Plotting interpolated initial surface...")
fig = plt.figure(figsize=(5, 5))
ax = plt.gca()
cs = firedrake.tricontourf(op.initial_surface, 50, cmap=matplotlib.cm.coolwarm, axes=ax)
ax.contour(x, y, elev, vmin=-0.01, vmax=0.01, levels=0, colors='k')
ax.set_xlabel("UTM x-coordinate [m]")
ax.set_ylabel("UTM y-coordinate [m]")
# op.annotate_plot(ax, gauges=True, coords="utm")
ax.set_title("Interpolant")
cb = fig.colorbar(cs)
cb.set_label("Initial free surface [m]")
fname = 'outputs/ic_{:d}'.format(mesh.num_cells())
plt.savefig(fname + '.png')
# plt.savefig(fname + '.pdf')

print_output("Plotting Coriolis parameter...")
fig = plt.figure(figsize=(5, 5))
ax = plt.gca()
P1 = op.bathymetry.function_space()
cs = firedrake.tricontourf(op.set_coriolis(P1), 50, cmap=matplotlib.cm.coolwarm, axes=ax)
ax.contour(x, y, elev, vmin=-0.01, vmax=0.01, levels=0, colors='k')
ax.set_xlabel("UTM x-coordinate [m]")
ax.set_ylabel("UTM y-coordinate [m]")
cb = fig.colorbar(cs)
cb.set_label("Coriolis parameter")
fname = 'outputs/coriolis_{:d}'.format(mesh.num_cells())
plt.savefig(fname + '.png')
# plt.savefig(fname + '.pdf')
print_output("Done!")
