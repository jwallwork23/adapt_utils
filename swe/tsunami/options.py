from thetis import *
from thetis.configuration import *

import scipy.interpolate as si
import numpy as np
import h5py
import matplotlib.pyplot as plt

from adapt_utils.swe.options import ShallowWaterOptions
from adapt_utils.swe.tsunami.conversion import *
from adapt_utils.norms import total_variation, lp_norm
# from adapt_utils.misc import find


__all__ = ["TsunamiOptions"]


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

    def set_bathymetry(self, fs=None, dat=None, cap=30.0):
        assert hasattr(self, 'initial_surface')
        if cap is not None:
            assert cap > 0.0
        fs = fs or FunctionSpace(self.default_mesh, "CG", 1)
        self.bathymetry = Function(fs, name="Bathymetry")

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
        # if self.initial_surface.function_space().mesh() != fs.mesh():
        #     self.set_initial_surface(fs)
        for i, xy in enumerate(fs.mesh().coordinates.dat.data):
            # depth[i] -= self.initial_surface.dat.data[i]
            depth[i] -= bath_interp(xy[1], xy[0])
            # self.print_debug(msg.format(xy[0], xy[1], depth[i]/1000))
        if cap is not None:
            self.bathymetry.interpolate(max_value(cap, self.bathymetry))
        self.print_debug("Done!")
        return self.bathymetry

    def set_initial_surface(self, fs=None):
        fs = fs or FunctionSpace(self.default_mesh, "CG", 1)
        self.initial_surface = Function(fs, name="Initial free surface")

        # Interpolate bathymetry data *in lonlat space*
        lon, lat, elev = self.read_surface_file()
        x, y = lonlat_to_utm(lon, lat, self.force_zone_number)
        surf_interp = si.RectBivariateSpline(y, x, elev)

        # Insert interpolated data onto nodes of *problem domain space*
        self.print_debug("Interpolating initial surface...")
        # msg = "Coordinates ({:.1f}, {:.1f}) Surface {:.3f} m"
        for i, xy in enumerate(fs.mesh().coordinates.dat.data):
            self.initial_surface.dat.data[i] = surf_interp(xy[1], xy[0])
            # self.print_debug(msg.format(xy[0], xy[1], self.initial_surface.dat.data[i]))
        self.print_debug("Done!")
        return self.initial_surface

    def set_initial_condition(self, fs):
        P1 = FunctionSpace(fs.mesh(), "CG", 1)
        self.initial_value = Function(fs)
        u, eta = self.initial_value.split()

        # (Naively) assume zero initial velocity
        u.assign(0.0)

        # Interpolate free surface from inversion data
        self.set_initial_surface(P1)
        eta.interpolate(self.initial_surface)

        return self.initial_value

    def set_coriolis(self, fs):
        self.coriolis = Function(fs)
        x, y = SpatialCoordinate(fs.mesh())
        lat, lon = to_latlon(
            x, y, self.force_zone_number,
            northern=True, coords=fs.mesh().coordinates, force_longitude=True
        )
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
            msg = "Gauge '{:s}' is not valid. Choose from {:}."
            raise ValueError(msg.format(gauge, self.gauges.keys()))

        fig = plt.figure(figsize=[10.0, 5.0])
        ax = fig.add_subplot(111)

        # Plot measurements
        print_output("#### TODO: Get gauge data in higher precision")  # TODO: And update below
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
                errors['l1'] = {'data': lp_norm(y_data, p=1), 'name': r'$\ell_1$ error'}
                errors['l2'] = {'data': lp_norm(y_data, p=2), 'name': r'$\ell_2$ error'}
                errors['linf'] = {'data': lp_norm(y_data, p='inf'), 'name': r'$\ell_\infty$ error'}
            for key in errors:
                errors[key]['abs'] = []
                errors[key]['rel'] = []

        # Global time and profile arrays
        T = np.array([])
        Y = np.array([])

        # TODO: Plot multiple runs on single plot
        # Loop over mesh iterations
        for i in range(self.num_meshes):
            fname = os.path.join(self.di, 'diagnostic_gauges')
            if extension is not None:
                fname = '_'.join([fname, extension])
            fname = '_'.join([fname, '{:d}.hdf5'.format(i)])
            assert os.path.exists(fname)
            f = h5py.File(fname, 'r')
            y = f[gauge][()]
            y = y.reshape(len(y),)[:cutoff+1]

            if i == 0:
                y0 = y[0]
            y -= y0
            y = np.round(y, 2)  # Ensure consistent precision  # TODO: Update according to above
            t = f["time"][()]
            t = t.reshape(len(t),)[:cutoff+1]/60.0

            # Put arrays from individual meshes into global arrays
            T = np.concatenate((T, t))
            Y = np.concatenate((Y, y))

        # Plot timeseries for current mesh
        label = self.approach.replace('_', ' ').title()
        ax.plot(T, Y, label=label, linestyle='dashed', marker='x')
        f.close()

        # TODO
        # # Compute absolute and relative errors
        # if 'data' in self.gauges[gauge]:
        #     Y_cutoff = np.array(Y[:len(y_data)])
        #     error = Y_cutoff - np.array(y_data)
        #     if plot_lp:
        #         for p in ('l1', 'l2', 'linf'):
        #             errors[p]['abs'].append(lp_norm(error, p=p))
        #     errors['tv']['abs'].append(total_variation(error))
        #     for key in errors:
        #         errors[key]['rel'].append(errors[key]['abs'][-1]/errors[key]['data'])

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

        # TODO
        # # Plot relative errors
        # if 'data' not in self.gauges[gauge]:
        #     raise ValueError("Data not found.")
        # for key in errors:
        #     fig = plt.figure(figsize=[3.2, 4.8])
        #     ax = fig.add_subplot(111)
        #     ax.semilogx(resolutions, 100.0*np.array(errors[key]['rel']), marker='o')
        #     plt.xlabel("Number of elements")
        #     plt.ylabel(r"Relative {:s} (\%)".format(errors[key]['name']))
        #     plt.grid(True)
        #     fname = "gauge_{:s}_error_{:s}".format(key, gauge)
        #     if extension is not None:
        #         fname = '_'.join([fname, str(extension)])
        #     fig.savefig(os.path.join(self.di, '.'.join([fname, 'png'])))
        #     fig.savefig(os.path.join(self.di, '.'.join([fname, 'pdf'])))

    def set_qoi_kernel(self, solver_obj):
        # V = solver_obj.function_spaces.U_2d*solver_obj.function_spaces.P0_2d  # (Arbitrary)
        V = solver_obj.function_spaces.V_2d
        b = self.ball(V, source=False)

        # TODO: Normalise by area computed on fine reference mesh
        # area = assemble(b*dx)
        # area_fine_mesh = ...
        # rescaling = 1.0 if np.allclose(area, 0.0) else area_fine_mesh/area
        rescaling = 1.0

        self.kernel = Function(V, name="QoI kernel")
        kernel_u, kernel_eta = self.kernel.split()
        kernel_u.rename("QoI kernel (uv component)")
        kernel_eta.rename("QoI kernel (elev component)")
        kernel_eta.interpolate(rescaling*b)
        return self.kernel

    def plot_qoi(self):  # FIXME
        """Timeseries plot of instantaneous QoI."""
        print_output("#### TODO: Update plotting to use callback")
        plt.figure()
        T = self.trange/3600
        qois = [q/1.0e9 for q in self.qois]
        qoi = self.evaluate_qoi()/1.0e9
        plt.plot(T, qois, linestyle='dashed', color='b', marker='x')
        plt.fill_between(T, np.zeros_like(qois), qois)
        plt.xlabel(r"Time [$\mathrm h$]")
        plt.ylabel(r"Instantaneous QoI [$\mathrm{km}^3$]")
        plt.title(r"Time integrated QoI: ${:.1f}\,\mathrm k\mathrm m^3\,\mathrm h$".format(qoi))
        plt.savefig(os.path.join(self.di, "qoi_timeseries_{:s}.pdf".format(self.qoi_mode)))
