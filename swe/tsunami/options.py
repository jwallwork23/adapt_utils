from thetis import *
from thetis.configuration import *

import scipy.interpolate as si
import numpy as np
import h5py
import matplotlib
import matplotlib.pyplot as plt

from adapt_utils.swe.options import ShallowWaterOptions
from adapt_utils.swe.tsunami.conversion import *
from adapt_utils.norms import total_variation, lp_norm


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
    def plot_timeseries(self, gauge, plot_lp=False, cutoff=24, plot_pdf=False, plot_png=True, nonlinear=False):
        """
        Plot timeseries for `gauge` under all stored mesh resolutions.

        :arg gauge: tag for gauge to be plotted.
        :kwarg plot_lp: toggle plotting of Lp errors.
        :kwarg cutoff: time cutoff level.
        """
        if gauge not in self.gauges:
            msg = "Gauge '{:s}' is not valid. Choose from {:}."
            raise ValueError(msg.format(gauge, self.gauges.keys()))
        fexts = []
        if plot_pdf:
            fexts.append('pdf')
        if plot_png:
            fexts.append('png')

        fig, ax = plt.subplots(figsize=(10.0, 5.0))

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

        # Loop over runs
        num_cells = []
        for level in range(4):
            tag = 'nonlinear_level{:d}'.format(level)
            if not nonlinear:
                tag = tag[3:]
            fname = os.path.join(self.di, '_'.join(['diagnostic_gauges', tag, '0.hdf5']))
            if not os.path.exists(fname):
                continue
            fname = os.path.join(self.di, '_'.join(['meshdata', tag, '0.hdf5']))
            if not os.path.exists(fname):
                continue
            with h5py.File(fname, 'r') as f:
                num_cells.append(f['num_cells'][()])

            # Global time and profile arrays
            T = np.array([])
            Y = np.array([])

            # Loop over mesh iterations  # FIXME: Only currently works for fixed mesh
            for i in range(self.num_meshes):
                fname = os.path.join(self.di, '_'.join(['diagnostic_gauges', tag, str(i)+'.hdf5']))
                assert os.path.exists(fname)
                with h5py.File(fname, 'r') as f:
                    y = f[gauge][()]
                    y = y.reshape(len(y),)[:cutoff+1]

                    if i == 0:
                        y0 = y[0]
                    y -= y0
                    y = np.round(y, 2)  # Ensure consistent precision  # TODO: Update as above
                    t = f["time"][()]
                    t = t.reshape(len(t),)[:cutoff+1]/60.0

                # Put arrays from individual meshes into global arrays
                T = np.concatenate((T, t))
                Y = np.concatenate((Y, y))

            # Plot timeseries for current mesh
            label = '{:s} ({:d} elements)'.format(self.approach, num_cells[-1])
            label = label.replace('_', ' ').title()
            ax.plot(T, Y, label=label, linestyle='dashed', marker='x')

            # Compute absolute and relative errors
            if 'data' in self.gauges[gauge]:
                error = np.array(Y[:cutoff+1]) - np.array(y_data[:cutoff+1])
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

        fname = "gauge_timeseries_{:s}_{:s}linear".format(gauge, 'non' if nonlinear else '')
        for fext in fexts:
            fig.savefig(os.path.join(self.di, '.'.join([fname, fext])))

        # Plot relative errors
        if 'data' not in self.gauges[gauge]:
            raise ValueError("Data not found.")
        for key in errors:
            fig, ax = plt.subplots(figsize=(3.2, 4.8))
            ax.semilogx(num_cells, 100.0*np.array(errors[key]['rel']), marker='o')
            plt.xlabel("Number of elements")
            # plt.ylabel(r"Relative {:s} (\%)".format(errors[key]['name']))
            plt.ylabel("Relative {:s} (%)".format(errors[key]['name']))
            plt.grid(True)
            fname = "gauge_{:s}_error_{:s}_{:s}linear".format(key, gauge, 'non' if nonlinear else '')
            for fext in fexts:
                fig.savefig(os.path.join(self.di, '.'.join([fname, fext])))

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
        ftc = Function(fs)
        if not hasattr(self, 'kernel'):
            self.set_qoi_kernel(fs)
        ftc.assign(self.kernel)
        return ftc
