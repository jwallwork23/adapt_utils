from thetis import *
from firedrake.plot import two_dimension_plot

import os
import scipy.interpolate as si
from scipy.io.netcdf import NetCDFFile
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import rc
import numpy as np

rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)


nc = NetCDFFile('tohoku.nc', mmap=False)
lon = nc.variables['lon'][:]
lat = nc.variables['lat'][:-1]
elev = nc.variables['elevation'][:-1,:]/1000

fig, axes = plt.subplots(nrows=1, ncols=2, sharex=True)
ax1 = axes.flat[0]
cs = ax1.contourf(lon, lat, elev, 50, vmin=-9, vmax=2, cmap=matplotlib.cm.coolwarm)
ax1.set_xlabel("Degrees longitude")
ax1.set_ylabel("Degrees latitude")
xlim = ax1.get_xlim()
ylim = ax1.get_ylim()
# ax1.set(aspect='equal')
ax1.set_title("Original data")

lon_min = np.min(lon)
lon_max = np.max(lon)
lat_min = np.min(lat)
lat_max = np.max(lat)

mesh = RectangleMesh(40, 40, lon_max-lon_min, lat_max-lat_min)
x, y = SpatialCoordinate(mesh)
mesh.coordinates.interpolate(as_vector([x+lon_min, y+lat_min]))

P1 = FunctionSpace(mesh, "CG", 1)
bathy_interp = si.RectBivariateSpline(lat, lon, elev)
b = Function(P1, name="Bathymetry")
for i in range(mesh.num_vertices()):
    xy = mesh.coordinates.dat.data[i] 
    b.dat.data[i] = bathy_interp(xy[1], xy[0])

ax2 = axes.flat[1]
ax2 = two_dimension_plot(b, num_sample_points=10, vmin=-9, vmax=2, axes=ax2, cmap=matplotlib.cm.coolwarm)
ax2.set_xlabel("Degrees longitude")
ax2.set_ylabel("Degrees latitude")
ax2.set_xlim(xlim)
ax2.set_ylim(ylim)
ax2.yaxis.set_label_position("right")
ax2.yaxis.tick_right()
# ax2.set(aspect='equal')
ax2.set_title("Interpolated data")

cb = fig.colorbar(cs, orientation='horizontal', ax=axes.ravel().tolist(), pad=0.2)
cb.set_label("Bathymetry $[\mathrm k\mathrm m]$")
di = create_directory('outputs/tohoku')
plt.savefig(os.path.join(di, 'bathymetry.pdf'))
plt.show()
