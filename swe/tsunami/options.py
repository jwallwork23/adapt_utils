from thetis import *
from thetis.configuration import *

import scipy.interpolate as si
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from adapt_utils.swe.options import ShallowWaterOptions
from adapt_utils.swe.tsunami.conversion import lonlat_to_utm, to_latlon, radians
from adapt_utils.adapt.metric import steady_metric


__all__ = ["TsunamiOptions"]


matplotlib.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
matplotlib.rc('text', usetex=True)


class TsunamiOptions(ShallowWaterOptions):
    """
    Parameter class for general tsunami propagation problems.
    """
    Omega = PositiveFloat(7.291e-5, help="Planetary rotation rate").tag(config=True)

    def __init__(self, utm=True, n=40, **kwargs):
        super(TsunamiOptions, self).__init__(**kwargs)
        self.utm = utm
        if not hasattr(self, 'force_zone_number'):
            self.force_zone_number = False

        # Setup default domain
        lon, lat, elev = self.read_bathymetry_file()
        lon_min = np.min(lon)
        lon_max = np.max(lon)
        lat_min = np.min(lat)
        lat_max = np.max(lat)
        self.default_mesh = RectangleMesh(n, n, lon_max-lon_min, lat_max-lat_min)
        x, y = SpatialCoordinate(self.default_mesh)
        self.default_mesh.coordinates.interpolate(as_vector([x+lon_min, y+lat_min]))
        if utm:
            self.default_mesh.coordinates.interpolate(as_vector(lonlat_to_utm(y, x, force_zone_number=self.force_zone_number)))

        # Set fields
        P1 = FunctionSpace(self.default_mesh, "CG", 1)
        self.set_bathymetry(P1, dat=(lon, lat, elev))
        self.set_initial_surface(P1)
        self.base_viscosity = 1.0e-3

        # Wetting and drying
        self.wetting_and_drying = True
        self.wetting_and_drying_alpha = Constant(0.43)

        # Timestepping
        self.timestepper = 'CrankNicolson'
        self.dt = 5.0
        self.dt_per_export = 10
        self.dt_per_remesh = 10
        self.end_time = 1800.0

        self.gauges = {}
        self.locations_of_interest = {}

    def set_bathymetry(self, fs, dat=None):
        assert fs.ufl_element().degree() == 1 and fs.ufl_element().family() == 'Lagrange'
        x0, y0, elev = dat or self.read_bathymetry_file()
        if self.utm:
            x0, y0 = lonlat_to_utm(y0, x0, force_zone_number=self.force_zone_number)
        bathy_interp = si.RectBivariateSpline(y0, x0, elev)
        self.bathymetry = Function(fs, name="Bathymetry")
        self.print_debug("Interpolating bathymetry...")
        msg = "Coordinates ({:.1f}, {:.1f}) Bathymetry {:.3f} km"
        for i in range(fs.mesh().num_vertices()):
            xy = fs.mesh().coordinates.dat.data[i] 
            self.bathymetry.dat.data[i] = bathy_interp(xy[1], xy[0])
            self.print_debug(msg.format(xy[0], xy[1], self.bathymetry.dat.data[i]/1000))
        self.print_debug("Done!")
        return self.bathymetry

    def set_initial_surface(self, fs):
        assert fs.ufl_element().degree() == 1 and fs.ufl_element().family() == 'Lagrange'
        x0, y0, elev = self.read_surface_file()
        if self.utm:
            x0, y0 = lonlat_to_utm(y0, x0, force_zone_number=self.force_zone_number)
        surf_interp = si.RectBivariateSpline(y0, x0, elev)
        self.initial_surface = Function(fs, name="Initial free surface")
        self.print_debug("Interpolating initial surface...")
        msg = "Coordinates ({:.1f}, {:.1f}) Surface {:.3f} m"
        for i in range(fs.mesh().num_vertices()):
            xy = fs.mesh().coordinates.dat.data[i] 
            self.initial_surface.dat.data[i] = surf_interp(xy[1], xy[0])
            self.print_debug(msg.format(xy[0], xy[1], self.initial_surface.dat.data[i]))
        self.print_debug("Done!")
        return self.initial_surface

    def set_initial_condition(self, fs):
        self.initial_value = Function(fs)
        u, eta = self.initial_value.split()

        # (Naively) assume zero initial velocity
        u.assign(0.0)

        # Interpolate free surface from inversion data
        self.set_initial_surface(FunctionSpace(fs.mesh(), "CG", 1))
        eta.interpolate(self.initial_surface)

        return self.initial_value

    def set_coriolis(self, fs):
        self.coriolis = Function(fs)
        x, y = SpatialCoordinate(fs.mesh())
        lat = to_latlon(x, y, self.force_zone_number, northern=True, force_longitude=True)[0] if self.utm else y
        self.coriolis.interpolate(2*self.Omega*sin(radians(lat)))
        return self.coriolis

    def get_export_func(self, solver_obj):
        eta = solver_obj.fields.elev_2d

        def export_func():
            # for loc in (self.gauges, self.locations_of_interest):
            for loc in (self.gauges,):
                for l in loc:
                    xy = loc[l]["utm" if self.utm else "lonlat"]
                    loc[l]["timeseries"].append(float(eta.at(xy)))

        return export_func

    def plot_coastline(self, axes):
        """
        Plot the coastline according to `bathymetry` on `axes`.
        """
        plot(self.bathymetry, vmin=-0.01, vmax=0.01, levels=0, axes=axes, cmap=None, colors='k', contour=True)

    def plot_timeseries(self):
        """
        Plot gauge timeseries data.
        """
        N = len(self.gauges["P02"]["timeseries"])
        assert N > 0
        t = np.linspace(0, self.end_time/60.0, N)
        for g in self.gauges:
            y = np.array(self.gauges[g]["timeseries"])-self.gauges[g]["timeseries"][0]
            plt.plot(t, y, label=g, linestyle='dashed', marker='*')
        plt.xlabel(r"Time $[\mathrm{min}]$")
        plt.ylabel("Free surface displacement $[\mathrm m]$")
        plt.legend()
        plt.savefig(os.path.join(self.di, "gauge_timeseries.pdf"))
