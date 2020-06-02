from thetis import *
from thetis.configuration import *

import scipy.interpolate as si
import numpy as np
import h5py
import matplotlib
import matplotlib.pyplot as plt

from adapt_utils.swe.options import ShallowWaterOptions
from adapt_utils.swe.tsunami.conversion import *
from adapt_utils.norms import timeseries_error


__all__ = ["TsunamiOptions"]


class TsunamiOptions(ShallowWaterOptions):
    """
    Parameter class for general tsunami propagation problems.
    """
    Omega = PositiveFloat(7.291e-5, help="Planetary rotation rate").tag(config=True)
    bathymetry_cap = NonNegativeFloat(30.0, allow_none=True, help="Minimum depth").tag(config=True)

    def __init__(self, **kwargs):
        super(TsunamiOptions, self).__init__(**kwargs)
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

        # Solver
        # =====
        # The time-dependent shallow water system looks like
        #
        #                             ------------------------- -----   -----
        #       ------------- -----   |                 |     | |   |   |   |
        #       | A00 | A01 | | U |   |  T + C + V + D  |  G  | | U |   | 0 |
        # A x = ------------- ----- = |                 |     | |   | = |   |  = b,
        #       | A10 | A11 | | H |   ------------------------- -----   -----
        #       ------------- -----   |        B        |  T  | | H |   | 0 |
        #                             ------------------------- -----   -----
        #
        # where:
        #  * T - time derivative;
        #  * C - Coriolis;
        #  * V - viscosity;
        #  * D - quadratic drag;
        #  * G - gravity;
        #  * B - bathymetry.
        #
        # We apply a multiplicative fieldsplit preconditioner, i.e. block Gauss-Seidel:
        #
        #     ---------------- ------------ ----------------
        #     | I |     0    | |   I  | 0 | | A00^{-1} | 0 |
        # P = ---------------- ------------ ----------------.
        #     | 0 | A11^{-1} | | -A10 | 0 | |    0     | I |
        #     ---------------- ------------ ----------------
        self.params = {
            "snes_converged_reason": None,
            "ksp_type": "gmres",                     # default
            "ksp_converged_reason": None,
            "pc_type": "fieldsplit",                 # default
            "pc_fieldsplit_type": "multiplicative",  # default
            "fieldsplit_U_2d": {
                "ksp_type": "preonly",               # default
                "ksp_max_it": 10000,                 # default
                "ksp_rtol": 1.0e-05,                 # default
                "pc_type": "sor",                    # default
                # "ksp_view": None,
                # "ksp_converged_reason": None,
            },
            "fieldsplit_H_2d": {
                "ksp_type": "preonly",               # default
                "ksp_max_it": 10000,                 # default
                "ksp_rtol": 1.0e-05,                 # default
                # "pc_type": "sor",                  # default
                "pc_type": "jacobi",
                # "ksp_view": None,
                # "ksp_converged_reason": None,
            },
        }
        self.adjoint_params.update(self.params)

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
        assert hasattr(self, 'initial_surface')
        if self.bathymetry_cap is not None:
            assert self.bathymetry_cap >= 0.0
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
        if self.bathymetry_cap is not None:
            self.bathymetry.interpolate(max_value(self.bathymetry_cap, self.bathymetry))
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

    def set_qoi_kernel(self, fs):
        # b = self.ball(fs.mesh(), source=False)
        # b = self.circular_bump(fs.mesh(), source=False)
        b = self.gaussian(fs.mesh(), source=False)

        # TODO: Normalise by area computed on fine reference mesh
        # area = assemble(b*dx)
        # area_fine_mesh = ...
        # rescaling = 1.0 if np.allclose(area, 0.0) else area_fine_mesh/area
        rescaling = 1.0

        self.kernel = Function(fs, name="QoI kernel")
        kernel_u, kernel_eta = self.kernel.split()
        kernel_u.rename("QoI kernel (uv component)")
        kernel_eta.rename("QoI kernel (elev component)")
        kernel_eta.interpolate(rescaling*b)
        return self.kernel

    def set_final_condition(self, fs):
        self.set_qoi_kernel(fs)
        return Function(fs, name="Final time condition").assign(self.kernel)

    def get_gauge_data(self, gauge, **kwargs):
        raise NotImplementedError("Implement in derived class")

    # TODO: Plot multiple mesh approaches
    def plot_timeseries(self, gauge, **kwargs):
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
        if kwargs.get('plot_png', True):
            fexts.append('png')

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
        for linearity in ('linear', 'nonlinear'):
            num_cells[linearity] = []

            # Plot observations
            fig, ax = plt.subplots(figsize=(10.0, 5.0))
            ax.plot(time, data, label='Data', linestyle='solid')

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
                ax.plot(T, Y, label=label, linestyle='dashed', marker='x')

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
            ax.set_title("{:s} timeseries ({:s})".format(gauge, linearity))
            ax.set_xlabel("Time [min]")
            ax.set_ylabel("Free surface displacement [m]")
            ax.set_xlim([0, cutoff])
            plt.grid(True)
            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width*0.8, box.height])
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            fname = "gauge_timeseries_{:s}_{:s}".format(gauge, linearity)
            for fext in fexts:
                fig.savefig(os.path.join(self.di, '.'.join([fname, fext])))
            plt.close()

        # Plot relative errors
        print_output("Done!")
        print_output("Plotting errors for gauge {:s}...".format(gauge))
        if 'data' not in self.gauges[gauge]:
            raise ValueError("Data not found.")
        for key in errors:
            fig, ax = plt.subplots(figsize=(3.2, 4.8))
            for linearity in ('linear', 'nonlinear'):
                relative_errors = 100.0*np.array(errors[key][linearity]['rel'])
                cells = num_cells[linearity][:len(relative_errors)]
                ax.semilogx(cells, relative_errors, marker='o', label=linearity.title())
            ax.set_title("{:s}".format(gauge))
            ax.set_xlabel("Number of elements")
            ax.set_ylabel("Relative {:s} (%)".format(errors[key]['name']))
            plt.grid(True)
            ax.legend()
            fname = "gauge_{:s}_error_{:s}".format(key, gauge)
            plt.tight_layout()
            for fext in fexts:
                fig.savefig(os.path.join(self.di, '.'.join([fname, fext])))
            plt.close()
        print_output("Done!")
