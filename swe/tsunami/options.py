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
from adapt_utils.misc import find, heaviside_approx


__all__ = ["TsunamiOptions"]


# matplotlib.rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})  # FIXME
# matplotlib.rc('text', usetex=True)  # FIXME


class TsunamiOptions(ShallowWaterOptions):
    """
    Parameter class for general tsunami propagation problems.
    """
    Omega = PositiveFloat(7.291e-5, help="Planetary rotation rate").tag(config=True)

    def __init__(self, **kwargs):
        super(TsunamiOptions, self).__init__(**kwargs)
        if not hasattr(self, 'force_zone_number'):
            self.force_zone_number = False
        self.base_viscosity = 1.0e-3
        self.gauges = {}
        self.locations_of_interest = {}

    def get_utm_mesh(self):
        zone = self.force_zone_number
        self.default_mesh = Mesh(Function(self.lonlat_mesh.coordinates))
        lon, lat = SpatialCoordinate(self.default_mesh)
        self.default_mesh.coordinates.interpolate(as_vector(lonlat_to_utm(lon, lat, zone)))

    def get_lonlat_mesh(self, northern=True):
        zone = self.force_zone_number
        self.lonlat_mesh = Mesh(Function(self.default_mesh.coordinates))
        x, y = SpatialCoordinate(self.lonlat_mesh)
        self.lonlat_mesh.coordinates.interpolate(as_vector(utm_to_lonlat(x, y, zone, northern=northern, force_longitude=True)))

    def set_bathymetry(self, dat=None, cap=30.0):
        assert hasattr(self, 'initial_surface')
        if cap is not None:
            assert cap > 0.0
        P1 = FunctionSpace(self.default_mesh, "CG", 1)
        self.bathymetry = Function(P1, name="Bathymetry")

        # Interpolate bathymetry data *in lonlat space*
        lon, lat, elev = dat or self.read_bathymetry_file()
        self.print_debug("Transforming bathymetry to UTM coordinates...")
        x, y = lonlat_to_utm(lon, lat, self.force_zone_number)
        self.print_debug("Done!")
        self.print_debug("Creating bathymetry interpolator...")
        bath_interp = si.RectBivariateSpline(y, x, elev)
        self.print_debug("Done!")

        # Insert interpolated data onto nodes of *problem domain space*
        self.print_debug("Interpolating bathymetry...")
        # msg = "Coordinates ({:.1f}, {:.1f}) Bathymetry {:.3f} km"
        depth = self.bathymetry.dat.data 
        for i, xy in enumerate(self.default_mesh.coordinates.dat.data):
            depth[i] -= self.initial_surface.dat.data[i]
            depth[i] -= bath_interp(xy[1], xy[0])
            # self.print_debug(msg.format(xy[0], xy[1], depth[i]/1000))
        if cap is not None:
            self.bathymetry.interpolate(max_value(cap, self.bathymetry))
        self.print_debug("Done!")
        return self.bathymetry

    def set_initial_surface(self, fs=None):
        P1 = fs or FunctionSpace(self.default_mesh, "CG", 1)
        self.initial_surface = Function(P1, name="Initial free surface")

        # Interpolate bathymetry data *in lonlat space*
        lon, lat, elev = self.read_surface_file()
        x, y = lonlat_to_utm(lon, lat, self.force_zone_number)
        surf_interp = si.RectBivariateSpline(y, x, elev)

        # Insert interpolated data onto nodes of *problem domain space*
        self.print_debug("Interpolating initial surface...")
        # msg = "Coordinates ({:.1f}, {:.1f}) Surface {:.3f} m"
        for i, xy in enumerate(self.default_mesh.coordinates.dat.data):
            self.initial_surface.dat.data[i] = surf_interp(xy[1], xy[0])
            # self.print_debug(msg.format(xy[0], xy[1], self.initial_surface.dat.data[i]))
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
        lat, lon = to_latlon(x, y, self.force_zone_number, northern=True, force_longitude=True)
        self.coriolis.interpolate(2*self.Omega*sin(radians(lat)))
        return self.coriolis

    # TODO: Plot multiple mesh approaches
    def plot_timeseries(self, gauge, extension=None, plot_lp=False, cutoff=25):
        """
        Plot timeseries for `gauge` under all stored mesh resolutions.
        """
        try:
            assert gauge in self.gauges
        except AssertionError:
            raise ValueError("Gauge '{:s}' is not valid. Choose from {:}.".format(gauge, self.gauges.keys()))

        fig = plt.figure(figsize=[10.0, 5.0])
        ax = fig.add_subplot(111)

        # Plot measurements
        print_output("#### TODO: Get gauge data in higher precision")  # TODO: And update below
        N = int(self.end_time/self.dt/self.dt_per_export)
        if 'data' in self.gauges[gauge]:
            y_data = np.array(self.gauges[gauge]["data"])  # TODO: Store in a HDF5 file
        else:
            y_data = np.array([])
        t = np.linspace(0, float(len(y_data)-1), len(y_data))  # TODO: Read from 'time' in HDF5 file
        ax.plot(t, y_data, label='Data', linestyle='solid')

        # Dictionary for norms and errors of timeseries
        if 'data' in self.gauges[gauge]:
            errors = {'tv': {'data': total_variation(y_data), 'name': 'total variation'}}
            if plot_lp:
                errors['l1'] = {'data': lp_norm(y_data, p=1), 'name': '$\ell_1$ error'}
                errors['l2'] = {'data': lp_norm(y_data, p=2), 'name': '$\ell_2$ error'}
                errors['linf'] = {'data': lp_norm(y_data, p='inf'), 'name': '$\ell_\infty$ error'}
            for key in errors:
                errors[key]['abs'] = []
                errors[key]['rel'] = []

        # Find all relevant HDF5 files and sort by ascending mesh resolution
        fnames = find('diagnostic_gauges_*.hdf5', self.di)
        resolutions = []
        for fname in fnames:
            s = fname.split('_')
            res = int(s[-1][:-5])
            if len(s) == 4:  # TODO: Temporary: only plot fixed_mesh
                resolutions.append(res)
        resolutions.sort()

        # Loop over all available mesh resolutions
        for res in resolutions:
            fname = os.path.join(self.di, 'diagnostic_gauges')
            if extension is not None:
                fname = '_'.join([fname, extension])
            fname = '_'.join([fname, '{:d}.hdf5'.format(res)])
            assert os.path.exists(fname)
            f = h5py.File(fname, 'r')
            y = f[gauge][()]
            y = y.reshape(len(y),)[:cutoff+1]
            y -= y[0]
            y = np.round(y, 2)  # Ensure consistent precision  # TODO: Update according to above
            t = f["time"][()]
            t = t.reshape(len(t),)[:cutoff+1]/60.0

            # Plot timeseries for current mesh resolution
            label = ' '.join([self.approach.replace('_', ' '), "({:d} cells)".format(res)]).title()
            ax.plot(t, y, label=label, linestyle='dashed', marker='x')
            f.close()

            # Compute absolute and relative errors
            if 'data' in self.gauges[gauge]:
                y_cutoff = np.array(y[:len(y_data)])
                error = y_cutoff - np.array(y_data)
                if plot_lp:
                    for p in ('l1', 'l2', 'linf'):
                        errors[p]['abs'].append(lp_norm(error, p=p))
                errors['tv']['abs'].append(total_variation(error))
                for key in errors:
                    errors[key]['rel'].append(errors[key]['abs'][-1]/errors[key]['data'])
        # plt.xlabel(r"Time $[\mathrm{min}]$")
        plt.xlabel("Time [min]")
        # plt.ylabel(r"Free surface displacement $[\mathrm m]$")
        plt.ylabel("Free surface displacement [m]")
        plt.xlim([0, cutoff])
        plt.ylim([-2, 5])
        plt.grid(True)

        # Legend to one side
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width*0.8, box.height])
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

        fname = "gauge_timeseries_{:s}".format(gauge)
        if extension is not None:
            fname = '_'.join([fname, str(extension)])
        fig.savefig(os.path.join(self.di, '.'.join([fname, 'png'])))
        fig.savefig(os.path.join(self.di, '.'.join([fname, 'pdf'])))

        # Plot relative errors
        if not 'data' in self.gauges[gauge]:
            return
        for key in errors:
            fig = plt.figure(figsize=[3.2, 4.8])
            ax = fig.add_subplot(111)
            ax.semilogx(resolutions, 100.0*np.array(errors[key]['rel']), marker='o')
            plt.xlabel("Number of elements")
            plt.ylabel(r"Relative {:s} (\%)".format(errors[key]['name']))
            plt.grid(True)
            fname = "gauge_{:s}_error_{:s}".format(key, gauge)
            if extension is not None:
                fname = '_'.join([fname, str(extension)])
            fig.savefig(os.path.join(self.di, '.'.join([fname, 'png'])))
            fig.savefig(os.path.join(self.di, '.'.join([fname, 'pdf'])))

    def set_qoi_kernel(self, solver_obj):
        pass  # TODO

    def plot_qoi(self):  # FIXME
        """Timeseries plot of instantaneous QoI."""
        print_output("#### TODO: Update plotting to use callback")
        plt.figure()
        T = self.trange/3600
        qois = [q/1.0e9 for q in self.qois]
        qoi = self.evaluate_qoi()/1.0e9
        plt.plot(T, qois, linestyle='dashed', color='b', marker='x')
        plt.fill_between(T, np.zeros_like(qois), qois)
        plt.xlabel("Time [$\mathrm h$]")
        plt.ylabel("Instantaneous QoI [$\mathrm{km}^3$]")
        plt.title("Time integrated QoI: ${:.1f}\,\mathrm k\mathrm m^3\,\mathrm h$".format(qoi))
        plt.savefig(os.path.join(self.di, "qoi_timeseries_{:s}.pdf".format(self.qoi_mode)))
