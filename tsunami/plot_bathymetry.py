from thetis import *

import os
import scipy.interpolate as si
from scipy.io.netcdf import NetCDFFile
import matplotlib.pyplot as plt
import numpy as np

nc = NetCDFFile('tohoku.nc', mmap=False)
lon = nc.variables['lon'][:]
lat = nc.variables['lat'][:-1]
elev = nc.variables['elevation'][:-1,:]

lon_min = np.min(lon)
lon_max = np.max(lon)
num_lon = len(lon)
lat_min = np.min(lat)
lat_max = np.max(lat)
num_lat = len(lat)

# mesh = RectangleMesh(num_lon, num_lat, lon_max-lon_min, lat_max-lat_min)
mesh = RectangleMesh(40, 40, lon_max-lon_min, lat_max-lat_min)
x, y = SpatialCoordinate(mesh)
mesh.coordinates.interpolate(as_vector([x+lon_min, y+lat_min]))

P1 = FunctionSpace(mesh, "CG", 1)
bathy_interp = si.RectBivariateSpline(lat, lon, elev)
b = Function(P1, name="Bathymetry")
for i in range(mesh.num_vertices()):
    xy = mesh.coordinates.dat.data[i] 
    b.dat.data[i] = bathy_interp(xy[1], xy[0])

fig = plt.figure(1)
ax = plt.subplot(121)
ax.contourf(lon, lat, elev)
plt.title("Original data")
ax = plt.subplot(122)
plot(b, axes=ax)
plt.title("Interpolated data")
di = create_directory('outputs/tohoku')
plt.savefig(os.path.join(di, 'bathymetry.pdf'))
plt.show()
