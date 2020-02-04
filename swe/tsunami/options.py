from thetis import *
from thetis.configuration import *

import scipy.interpolate as si
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import h5py

from adapt_utils.swe.options import ShallowWaterOptions
from adapt_utils.swe.tsunami.conversion import *
from adapt_utils.adapt.metric import steady_metric
from adapt_utils.norms import total_variation, lp_norm
from adapt_utils.misc import find


__all__ = ["TsunamiOptions"]


matplotlib.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
matplotlib.rc('text', usetex=True)


class TsunamiOptions(ShallowWaterOptions):
    """
    Parameter class for general tsunami propagation problems.
    """
    Omega = PositiveFloat(7.291e-5, help="Planetary rotation rate").tag(config=True)

    def __init__(self, utm=True, n=30, **kwargs):
        self.utm = utm
        super(TsunamiOptions, self).__init__(**kwargs)
        if not hasattr(self, 'force_zone_number'):
            self.force_zone_number = False

        # Setup longitude-latitude domain
        b_lon, b_lat, b = self.read_bathymetry_file()
        lon_min = np.min(b_lon)
        lon_diff = np.max(b_lon) - lon_min
        lat_min = np.min(b_lat)
        lat_diff = np.max(b_lat) - lat_min
        self.lonlat_mesh = RectangleMesh(n, n*int(np.round(lon_diff/lat_diff)), lon_diff, lat_diff)
        lon, lat = SpatialCoordinate(self.lonlat_mesh)
        self.lonlat_mesh.coordinates.interpolate(as_vector([lon + lon_min, lat + lat_min]))

        # Setup problem domain
        self.default_mesh = Mesh(Function(self.lonlat_mesh.coordinates))
        if self.utm:
            self.get_utm_mesh()

        # Set fields
        self.set_bathymetry(dat=(b_lon, b_lat, b), adapted=False)
        self.set_initial_surface()
        self.base_viscosity = 1.0e-3

        # Wetting and drying
        self.wetting_and_drying = True
        # self.wetting_and_drying_alpha = Constant(0.43)
        self.wetting_and_drying_alpha = Constant(1.5)

        # Timestepping
        self.timestepper = 'CrankNicolson'
        self.dt = 5.0
        self.dt_per_export = 12
        self.dt_per_remesh = 12
        self.end_time = 1500.0

        self.gauges = {}
        self.locations_of_interest = {}

        # Outputs
        P1DG = FunctionSpace(self.default_mesh, "DG", 1)
        self.eta_tilde_file = File(os.path.join(self.di, 'eta_tilde.pvd'))
        self.eta_tilde = Function(P1DG, name='Modified elevation')

    def get_utm_mesh(self):
        zone = self.force_zone_number
        self.default_mesh = Mesh(Function(self.lonlat_mesh.coordinates))
        lon, lat = SpatialCoordinate(self.default_mesh)
        self.default_mesh.coordinates.interpolate(as_vector(lonlat_to_utm(lon, lat, zone)))

    def get_lonlat_mesh(self, northern=True):
        zone = self.force_zone_number
        self.lonlat_mesh = Mesh(Function(self.default_mesh.coordinates))
        if self.utm:
            x, y = SpatialCoordinate(self.lonlat_mesh)
            self.lonlat_mesh.coordinates.interpolate(as_vector(utm_to_lonlat(x, y, zone, northern=northern, force_longitude=True)))

    def set_bathymetry(self, fs=None, dat=None, adapted=False):
        if fs is not None:
            self.default_mesh = fs.mesh()
        P1 = FunctionSpace(self.default_mesh, "CG", 1)
        self.bathymetry = Function(P1, name="Bathymetry")
        if adapted:
            self.get_lonlat_mesh()

        # Interpolate bathymetry data *in lonlat space*
        lon, lat, elev = dat or self.read_bathymetry_file()
        bath_interp = si.RectBivariateSpline(lat, lon, elev)

        # Insert interpolated data onto nodes of *problem domain space*
        self.print_debug("Interpolating bathymetry...")
        msg = "Coordinates ({:.1f}, {:.1f}) Bathymetry {:.3f} km"
        for i in range(self.lonlat_mesh.num_vertices()):
            lonlat = self.lonlat_mesh.coordinates.dat.data[i] 
            self.bathymetry.dat.data[i] = -bath_interp(lonlat[1], lonlat[0])
            self.print_debug(msg.format(lonlat[0], lonlat[1], self.bathymetry.dat.data[i]/1000))
        self.print_debug("Done!")
        return self.bathymetry

    def set_initial_surface(self, fs=None):
        P1 = fs or FunctionSpace(self.default_mesh, "CG", 1)
        self.initial_surface = Function(P1, name="Initial free surface")

        # Interpolate bathymetry data *in lonlat space*
        x0, y0, elev = self.read_surface_file()
        surf_interp = si.RectBivariateSpline(y0, x0, elev)

        # Insert interpolated data onto nodes of *problem domain space*
        self.print_debug("Interpolating initial surface...")
        msg = "Coordinates ({:.1f}, {:.1f}) Surface {:.3f} m"
        for i in range(self.lonlat_mesh.num_vertices()):
            xy = self.lonlat_mesh.coordinates.dat.data[i] 
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

    def plot_coastline(self, axes):
        """
        Plot the coastline according to `bathymetry` on `axes`.
        """
        plot(self.bathymetry, vmin=-0.01, vmax=0.01, levels=0, axes=axes, cmap=None, colors='k', contour=True)

    def get_eta_tilde(self, solver_obj):
        bathymetry_displacement = solver_obj.eq_sw.bathymetry_displacement_mass_term.wd_bathymetry_displacement
        eta = solver_obj.fields.elev_2d
        self.eta_tilde.project(eta + bathymetry_displacement(eta))

    def get_export_func(self, solver_obj):
        def export_func():
            self.get_eta_tilde(solver_obj)
            self.eta_tilde_file.write(self.eta_tilde)
        return export_func

    def plot_timeseries(self, gauge, extension=None):  # TODO: Plot multiple mesh approaches
        """
        Plot timeseries for `gauge` under all stored mesh resolutions.
        """
        try:
            assert gauge in self.gauges
        except AssertionError:
            raise ValueError("Gauge '{:s}' is not valid. Choose from {:}.".format(gauge, self.gauges.keys()))

        fig = plt.figure()
        ax = plt.gca()

        print_output("#### TODO: Get gauge data in higher precision")
        y_data = np.array(self.gauges[gauge]["data"])  # TODO: Store in a HDF5 file
        N = int(self.end_time/self.dt/self.dt_per_export)
        t = np.linspace(0, self.end_time/60.0, N+1)  # TODO: Read from 'time' in HDF5 file
        ax.plot(t, y_data, label='Data', linestyle='solid')

        # Dictionary for norms and errors of timeseries
        errors = {'tv': {'data': total_variation(y_data)}}
        for p in ('l1', 'l2', 'linf'):
            errors[p] = {'data': lp_norm(y_data, p=p)}
        for key in errors:
            errors[key]['abs'] = []
            errors[key]['rel'] = []

        # Find all relevant HDF5 files and sort by ascending mesh resolution
        approach = 'uniform' if self.approach == 'fixed_mesh' else self.approach
        fnames = find('diagnostic_gauges_*.hdf5', self.di)
        resolutions = [int(fname.split('_')[-1][:-5]) for fname in fnames]
        resolutions.sort()

        # Loop over all available mesh resolutions
        for res in resolutions:
            f = h5py.File(os.path.join(self.di, 'diagnostic_gauges_{:d}.hdf5'.format(res)), 'r')
            y = f[gauge][()].reshape(N+1,)
            y -= y[0]

            # Plot timeseries for current mesh resolution
            label = ' '.join([approach.replace('_', ' '), "({:d} cells)".format(res)]).title()
            ax.plot(t, y, label=label, linestyle='dashed', marker='x')
            f.close()

            # Compute absolute and relative errors
            error = np.array(y) - np.array(y_data)
            for p in ('l1', 'l2', 'linf'):
                errors[p]['abs'].append(lp_norm(error, p=p))
            errors['tv']['abs'].append(total_variation(error))
            for key in errors:
                errors[key]['rel'].append(errors[key]['abs'][-1]/errors[key]['data'])
        plt.xlabel(r"Time $[\mathrm{min}]$")
        plt.ylabel("Free surface displacement $[\mathrm m]$")
        plt.ylim([-2, 5])
        plt.legend()
        fname = "gauge_timeseries_{:s}".format(gauge)
        if extension is not None:
            fname = '_'.join([fname, str(extension)])
        fig.savefig(os.path.join(self.di, '.'.join([fname, 'png'])))
        fig.savefig(os.path.join(self.di, '.'.join([fname, 'pdf'])))

        # Plot total variation
        fig = plt.figure()
        ax = plt.gca()
        ax.semilogx(resolutions, 100.0*np.array(errors['tv']['rel']), marker='o')
        plt.xlabel("Number of elements")
        plt.ylabel("Relative total variation error (\%)")
        fname = "gauge_tv_error_{:s}".format(gauge)
        if extension is not None:
            fname = '_'.join([fname, str(extension)])
        fig.savefig(os.path.join(self.di, '.'.join([fname, 'png'])))
        fig.savefig(os.path.join(self.di, '.'.join([fname, 'pdf'])))

        # TODO: Optionally do something with lp errors
