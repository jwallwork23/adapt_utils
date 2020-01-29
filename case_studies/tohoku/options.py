from thetis import *
from thetis.configuration import *

from scipy.io.netcdf import NetCDFFile

from adapt_utils.tsunami.options import TsunamiOptions


__all__ = ["TohokuOptions"]


class TohokuOptions(TsunamiOptions):
    # TODO: doc

    def __init__(self, **kwargs):
        self.force_zone_number = 54
        super(TohokuOptions, self).__init__(**kwargs)

        # TODO: gauges

    def read_bathymetry_file(self, km=False):
        """Initial bathymetry data courtesy of GEBCO."""
        nc = NetCDFFile('resources/tohoku.nc', mmap=False)
        lon = nc.variables['lon'][:]
        lat = nc.variables['lat'][:-1]
        elev = nc.variables['elevation'][:-1,:]/1000 if km else nc.variables['elevation'][:-1,:]
        nc.close()
        return lon, lat, elev

    def read_surface_file(self):
        """Initial suface data courtesy of Saito."""
        nc = NetCDFFile('resources/surf_zeroed.nc', mmap=False)
        lon = nc.variables['lon'][:]
        lat = nc.variables['lat'][:]
        elev = nc.variables['z'][:,:]
        nc.close()
        return lon, lat, elev

    def set_boundary_conditions(self, fs):
        self.boundary_conditions = {}
        return self.boundary_conditions

    def set_qoi_kernel(self, solver_obj):
        pass
        # raise NotImplementedError  # TODO
