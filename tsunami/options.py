from thetis import *

import scipy.interpolate as si
from scipy.io.netcdf import NetCDFFile
import numpy as np

from adapt_utils.swe.options import ShallowWaterOptions
from adapt_utils.tsunami.conversion import lonlat_to_utm, to_latlon


__all__ = ["TohokuOptions"]


class TohokuOptions(ShallowWaterOptions):
    # TODO: doc
    def __init__(self, approach='fixed_mesh', utm=True):
        super(TohokuOptions, self).__init__(approach=approach)
        lon, lat, elev = self.read_bathymetry_file()
        lon_min = np.min(lon)
        lon_max = np.max(lon)
        lat_min = np.min(lat)
        lat_max = np.max(lat)
        self.default_mesh = RectangleMesh(40, 40, lon_max-lon_min, lat_max-lat_min)
        x, y = SpatialCoordinate(self.default_mesh)
        self.default_mesh.coordinates.interpolate(as_vector([x+lon_min, y+lat_min]))
        if utm:
            self.default_mesh.coordinates.interpolate(as_vector(lonlat_to_utm(y, x, force_zone_number=54)))
        P1 = FunctionSpace(self.default_mesh, "CG", 1)
        self.set_bathymetry(P1, dat=(lat, lon, elev), utm=utm)

    def read_bathymetry_file(self, km=False):
        nc = NetCDFFile('tohoku.nc', mmap=False)
        lon = nc.variables['lon'][:]
        lat = nc.variables['lat'][:-1]
        elev = nc.variables['elevation'][:-1,:]/1000 if km else nc.variables['elevation'][:-1,:]
        nc.close()
        return lon, lat, elev

    def set_bathymetry(self, fs, dat=None, utm=True):
        y0, x0, elev = dat or self.read_bathymetry_file()
        if utm:
            x0, y0 = lonlat_to_utm(y0, x0, force_zone_number=54)
        bathy_interp = si.RectBivariateSpline(y0, x0, elev)
        self.bathymetry = Function(fs, name="Bathymetry")
        for i in range(fs.mesh().num_vertices()):
            xy = fs.mesh().coordinates.dat.data[i] 
            self.bathymetry.dat.data[i] = bathy_interp(xy[1], xy[0])
        return self.bathymetry
