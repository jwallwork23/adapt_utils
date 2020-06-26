from thetis import *
from thetis.configuration import *

import scipy.interpolate as si
import numpy as np
import h5py
import matplotlib.pyplot as plt

from adapt_utils.unsteady.swe.options import CoupledOptions
from adapt_utils.unsteady.swe.tsunami.conversion import *
from adapt_utils.norms import timeseries_error


__all__ = ["TsunamiOptions"]


class TsunamiOptions(CoupledOptions):
    """
    Parameter class for general tsunami propagation problems.
    """
    Omega = PositiveFloat(7.291e-5, help="Planetary rotation rate").tag(config=True)
    bathymetry_cap = NonNegativeFloat(30.0, allow_none=True, help="Minimum depth").tag(config=True)

    def __init__(self, **kwargs):
        super(TsunamiOptions, self).__init__(**kwargs)
        self.solve_swe = True
        self.solve_tracer = False
        if not hasattr(self, 'force_zone_number'):
            self.force_zone_number = False
        self.gauges = {}
        self.locations_of_interest = {}

        # Stabilisation
        # =============
        # In some cases qmesh generated meshes can have tiny elements with sharp angles
        # near the coast. To account for this, we set a large SIPG parameter value. (If
        # we use the automatic SIPG functionality then it would return an enormous value.)
        self.base_viscosity = 1.0e-03
        self.sipg_parameter = Constant(100.0)

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

    def set_bathymetry(self, fs=None, dat=None):
        if self.bathymetry_cap is not None:
            assert self.bathymetry_cap >= 0.0
        fs = fs or FunctionSpace(self.default_mesh, "CG", 1)
        bathymetry = Function(fs, name="Bathymetry")

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
        depth = bathymetry.dat.data
        # initial_surface = self.set_initial_surface(fs)  # DUPLICATED
        for i, xy in enumerate(fs.mesh().coordinates.dat.data):
            # depth[i] -= initial_surface.dat.data[i]  # TODO?
            depth[i] -= bath_interp(xy[1], xy[0])
            # self.print_debug(msg.format(xy[0], xy[1], depth[i]/1000))
        if self.bathymetry_cap is not None:
            bathymetry.interpolate(max_value(self.bathymetry_cap, bathymetry))
        self.print_debug("Done!")
        return bathymetry

    def set_initial_surface(self, fs=None):
        fs = fs or FunctionSpace(self.default_mesh, "CG", 1)
        initial_surface = Function(fs, name="Initial free surface")

        # Interpolate bathymetry data *in lonlat space*
        lon, lat, elev = self.read_surface_file()
        x, y = lonlat_to_utm(lon, lat, self.force_zone_number)
        surf_interp = si.RectBivariateSpline(y, x, elev)

        # Insert interpolated data onto nodes of *problem domain space*
        self.print_debug("Interpolating initial surface...")
        # msg = "Coordinates ({:.1f}, {:.1f}) Surface {:.3f} m"
        for i, xy in enumerate(fs.mesh().coordinates.dat.data):
            initial_surface.dat.data[i] = surf_interp(xy[1], xy[0])
            # self.print_debug(msg.format(xy[0], xy[1], initial_surface.dat.data[i]))
        self.print_debug("Done!")
        return initial_surface

    def set_initial_condition(self, prob):
        u, eta = prob.fwd_solutions[0].split()

        # (Naively) assume zero initial velocity
        u.assign(0.0)

        # Interpolate free surface from inversion data
        eta.interpolate(self.set_initial_surface(prob.P1[0]))

    def set_coriolis(self, fs):
        x, y = SpatialCoordinate(fs.mesh())
        lat, lon = to_latlon(
            x, y, self.force_zone_number,
            northern=True, coords=fs.mesh().coordinates, force_longitude=True
        )
        return interpolate(2*self.Omega*sin(radians(lat)), fs)

    def set_qoi_kernel(self, prob, i):
        fs = prob.V[i]

        # b = self.ball(fs.mesh(), source=False)
        # b = self.circular_bump(fs.mesh(), source=False)
        b = self.gaussian(fs.mesh(), source=False)

        # TODO: Normalise by area computed on fine reference mesh
        # area = assemble(b*dx)
        # area_fine_mesh = ...
        # rescaling = 1.0 if np.allclose(area, 0.0) else area_fine_mesh/area
        rescaling = 1.0

        prob.kernels[i] = Function(fs, name="QoI kernel")
        kernel_u, kernel_eta = prob.kernels[i].split()
        kernel_u.rename("QoI kernel (uv component)")
        kernel_eta.rename("QoI kernel (elev component)")
        kernel_eta.interpolate(rescaling*b)

    def set_final_condition(self, prob):
        prob.adj_solutions[-1].assign(self.set_qoi_kernel(prob.V[-1]))

    def get_gauge_data(self, gauge, **kwargs):
        raise NotImplementedError("Implement in derived class")

    # TODO: Plot multiple mesh approaches
    # TODO: UPDATE
    def plot_timeseries(self, gauge, axes=None, **kwargs):
        """
        Plot timeseries for `gauge` under all stored mesh resolutions.

        :arg gauge: tag for gauge to be plotted.
        :kwarg cutoff: time cutoff level.
        """
        print_output("Plotting timeseries for gauge {:s}...".format(gauge))
        cutoff = kwargs.get('cutoff', 24)
        sample = kwargs.get('sample', 60)
        if gauge not in self.gauges:
            msg = "Gauge '{:s}' is not valid. Choose from {:}."
            raise ValueError(msg.format(gauge, self.gauges.keys()))
        fexts = []
        if kwargs.get('plot_pdf', False):
            fexts.append('pdf')
        if kwargs.get('plot_png', axes is None):
            fexts.append('png')
        plot_errors = axes is None
        plot_nonlinear = axes is None
        linearities = ('linear', )
        if plot_nonlinear:
            linearities += ('nonlinear', )

        # Get data
        if 'data' not in self.gauges[gauge]:
            self.get_gauge_data(gauge, sample=sample)
        data, time = self.gauges[gauge]['data'], self.gauges[gauge]['time']
        time = np.array([t/60.0 for t in time if t/60 <= cutoff + 1])
        data = np.array([d for d, t in zip(data, time)])
        data -= data[0]
        time -= time[0]
        assert len(time) == len(data)

        # Dictionary for norms and errors of timeseries
        if 'data' in self.gauges[gauge]:
            errors = {
                'tv': {'name': 'total variation'},
                # 'l1': {'name': r'$\ell_1$ error'},
                'l2': {'name': r'$\ell_2$ error'},
                # 'linf': {'name': r'$\ell_\infty$ error'},
            }
            for norm_type in errors:
                errors[norm_type]['data'] = timeseries_error(data, norm_type=norm_type)
                for linearity in ('linear', 'nonlinear'):
                    errors[norm_type][linearity] = {'abs': [], 'rel': []}

        # Consider cases of both linear and nonlinear shallow water equations
        num_cells = {}
        for linearity in linearities:
            num_cells[linearity] = []

            # Plot observations
            if axes is None:
                fig, axes = plt.subplots(figsize=(10.0, 5.0))
            axes.plot(time, data, label='Data', linestyle='solid')

            # Loop over runs
            for level in range(5):
                tag = '{:s}_level{:d}'.format(linearity, level)
                fname = os.path.join(self.di, '_'.join(['diagnostic_gauges', tag, '0.hdf5']))
                if not os.path.exists(fname):
                    continue
                fname = os.path.join(self.di, '_'.join(['meshdata', tag, '0.hdf5']))
                if not os.path.exists(fname):
                    continue
                with h5py.File(fname, 'r') as f:
                    num_cells[linearity].append(f['num_cells'][()])

                # Global time and profile arrays
                T = np.array([])
                Y = np.array([])

                # Loop over mesh iterations  # FIXME: Only currently works for fixed mesh
                for i in range(self.num_meshes):
                    fname = os.path.join(self.di, '_'.join(['diagnostic_gauges', tag, str(i)+'.hdf5']))
                    assert os.path.exists(fname)
                    with h5py.File(fname, 'r') as f:
                        gauge_time = f["time"][()]
                        gauge_time = gauge_time.reshape(len(gauge_time),)
                        gauge_time = np.array([t/60.0 for t in gauge_time if t <= 60*cutoff])
                        gauge_data = f[gauge][()]
                        gauge_data = gauge_data.reshape(len(gauge_data),)

                        # Set first value as reference point
                        if i == 0:
                            gauge_data0 = gauge_data[0]
                        gauge_data -= gauge_data0

                    # Put arrays from individual meshes into global arrays
                    Y = np.concatenate((Y, gauge_data))
                    T = np.concatenate((T, gauge_time))

                # Plot timeseries for current mesh
                label = '{:s} ({:d} elements)'.format(self.approach, num_cells[linearity][-1])
                label = label.replace('_', ' ').title()
                axes.plot(T, Y, label=label, linestyle='dashed', marker='x')

                r = 1
                if len(data) % len(Y) == 0:
                    r = len(data)//len(Y)
                else:
                    msg = "Gauge data and observations have incompatible lengths ({:d} vs {:d})"
                    raise ValueError(msg.format(len(gauge_data), len(data)))

                # Compute absolute and relative errors
                if 'data' in self.gauges[gauge]:
                    for norm_type in errors:
                        err = timeseries_error(data[::r], Y[:cutoff+1], norm_type=norm_type)
                        errors[norm_type][linearity]['abs'].append(err)
                        errors[norm_type][linearity]['rel'].append(err/errors[norm_type]['data'])

            # Plot labels etc.
            axes.set_title("{:s} timeseries ({:s})".format(gauge, linearity))
            axes.set_xlabel("Time [min]")
            axes.set_ylabel("Free surface displacement [m]")
            axes.set_xlim([0, cutoff])
            plt.grid(True)
            box = axes.get_position()
            axes.set_position([box.x0, box.y0, box.width*0.8, box.height])
            axes.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            fname = "gauge_timeseries_{:s}_{:s}".format(gauge, linearity)
            for fext in fexts:
                fig.savefig(os.path.join(self.di, '.'.join([fname, fext])))
            axes = None
        print_output("Done!")

        if not plot_errors:
            return

        # Plot relative errors
        print_output("Plotting errors for gauge {:s}...".format(gauge))
        if 'data' not in self.gauges[gauge]:
            raise ValueError("Data not found.")
        for key in errors:
            fig, axes = plt.subplots(figsize=(3.2, 4.8))
            for linearity in linearities:
                relative_errors = 100.0*np.array(errors[key][linearity]['rel'])
                cells = num_cells[linearity][:len(relative_errors)]
                axes.semilogx(cells, relative_errors, marker='o', label=linearity.title())
            axes.set_title("{:s}".format(gauge))
            axes.set_xlabel("Number of elements")
            axes.set_ylabel("Relative {:s} (%)".format(errors[key]['name']))
            plt.grid(True)
            axes.legend()
            fname = "gauge_{:s}_error_{:s}".format(key, gauge)
            plt.tight_layout()
            for fext in fexts:
                fig.savefig(os.path.join(self.di, '.'.join([fname, fext])))
            plt.close()
        print_output("Done!")