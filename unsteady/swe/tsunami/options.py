from thetis import *
from thetis.configuration import *

import h5py
import matplotlib.pyplot as plt
import numpy as np
import os
import scipy.interpolate as si

from adapt_utils.unsteady.options import CoupledOptions
from adapt_utils.unsteady.swe.tsunami.conversion import *
from adapt_utils.norms import timeseries_error


__all__ = ["TsunamiOptions"]


class TsunamiOptions(CoupledOptions):
    """
    Parameter class for general tsunami propagation problems.
    """
    Omega = PositiveFloat(7.291e-5, help="Planetary rotation rate").tag(config=True)
    bathymetry_cap = NonNegativeFloat(30.0, allow_none=True, help="Minimum depth").tag(config=True)

    def __init__(self, regularisation=0.0, **kwargs):
        """
        kwarg regularisation: non-negative constant parameter used for Tikhonov regularisation
            methods for the quantity of interest.
        """
        super(TsunamiOptions, self).__init__(**kwargs)
        self.solve_swe = True
        self.solve_tracer = False
        self.force_zone_number = kwargs.get('force_zone_number', False)
        self.gauges = {}
        self.locations_of_interest = {}

        # Model
        self.rotational = True

        # Bathymetry shift
        # ================
        self.shift_bathymetry = True
        self.shift_bathymetry_flag = False

        # Stabilisation
        # =============
        # In some cases qmesh generated meshes can have tiny elements with sharp angles
        # near the coast. To account for this, we set a large SIPG parameter value. (If
        # we use the automatic SIPG functionality then it would return an enormous value.)
        self.base_viscosity = 1.0e-03
        self.sipg_parameter = Constant(100.0)

        # Quantity of interest
        self.regularisation = regularisation
        if self.regularisation < 0.0:
            raise ValueError("Regularisation parameter should be non-negative.")

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

    def set_bathymetry(self, fs=None, dat=None, northern=True, force_longitude=True, **kwargs):
        if self.bathymetry_cap is not None:
            assert self.bathymetry_cap >= 0.0
        fs = fs or FunctionSpace(self.default_mesh, "CG", 1)
        bathymetry = Function(fs, name="Bathymetry")

        # Interpolate bathymetry data *in lonlat space*
        lon, lat, elev = dat or self.read_bathymetry_file(**kwargs)
        self.print_debug("INIT: Creating bathymetry interpolator...")
        bath_interp = si.RectBivariateSpline(lat, lon, elev)
        self.print_debug("INIT: Done!")

        # Insert interpolated data onto nodes of *problem domain space*
        self.print_debug("INIT: Interpolating bathymetry...")
        msg = "    Coordinates ({:.1f}, {:.1f}) Bathymetry {:.3f} km"
        for i, xy in enumerate(fs.mesh().coordinates.dat.data):
            lon, lat = utm_to_lonlat(xy[0], xy[1], self.force_zone_number,
                                     northern=northern, force_longitude=force_longitude)
            bathymetry.dat.data[i] -= bath_interp(lat, lon)
            self.print_debug(msg.format(xy[0], xy[1], bathymetry.dat.data[i]/1000), mode='full')
        self.print_debug("INIT: Done!")

        # Cap bathymetry to enforce a minimum depth
        self.print_debug("INIT: Capping bathymetry...")  # TODO: Should we really be doing this?
        if self.bathymetry_cap is not None:
            bathymetry.interpolate(max_value(self.bathymetry_cap, bathymetry))
        self.print_debug("INIT: Done!")

        return bathymetry

    def set_initial_surface(self, fs=None, northern=True, force_longitude=True, **kwargs):
        fs = fs or FunctionSpace(self.default_mesh, "CG", 1)
        initial_surface = Function(fs, name="Initial free surface")

        # Interpolate bathymetry data *in lonlat space*
        self.print_debug("INIT: Creating surface interpolator...")
        lon, lat, elev = self.read_surface_file(**kwargs)
        surf_interp = si.RectBivariateSpline(lat, lon, elev)
        self.print_debug("INIT: Done!")

        # Insert interpolated data onto nodes of *problem domain space*
        self.print_debug("INIT: Interpolating initial surface...")
        msg = "    Coordinates ({:.1f}, {:.1f}) Surface {:.3f} m"
        for i, xy in enumerate(fs.mesh().coordinates.dat.data):
            lon, lat = utm_to_lonlat(xy[0], xy[1], self.force_zone_number,
                                     northern=northern, force_longitude=force_longitude)
            initial_surface.dat.data[i] = surf_interp(lat, lon)
            self.print_debug(msg.format(xy[0], xy[1], initial_surface.dat.data[i]), mode='full')
        self.print_debug("INIT: Done!")

        return initial_surface

    def set_initial_condition(self, prob):

        # Read initial surface data from file
        surf = self.set_initial_surface(prob.P1[0])
        
        # Set initial condition
        self.print_debug("INIT: Setting initial condition...")
        u, eta = prob.fwd_solutions[0].split()
        u.assign(0.0)  # (Naively) assume zero initial velocity
        eta.interpolate(surf)  # Initial surface from inversion data
        self.print_debug("INIT: Done!")

        # Subtract from the bathymetry field
        self.subtract_surface_from_bathymetry(prob, surf=surf)

    def subtract_surface_from_bathymetry(self, prob, surf=None):
        """
        It is common practice in the tsunami modelling literature to subtract the initial surface
        from the bathymetry field. This is in line with the fact the earthquake disturbs the bed.
        In the hydrostatic case, we can assume that this translates to an idential disturbance to
        the ocean surface.
        """
        self.print_debug("INIT: Subtracting initial surface from bathymetry field...")
        assert hasattr(prob, 'bathymetry')
        if surf is None:
            surf = interpolate(prob.fwd_solutions[0].split()[1], prob.P1[0])
        assert prob.bathymetry[0].function_space() == surf.function_space()
        prob.bathymetry[0] -= surf
        for i in range(1, self.num_meshes):
            prob.bathymetry[1].project(prob.bathymetry[0])
        self.print_debug("INIT: Done!")

    def set_coriolis(self, fs):
        if not self.rotational:
            return
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

    def set_terminal_condition(self, prob):  # TODO: For hazard case
        # # b = self.ball(prob.meshes[-1], source=False)
        # # b = self.circular_bump(prob.meshes[-1], source=False)
        # b = self.gaussian(prob.meshes[-1], source=False)

        # # TODO: Normalise by area computed on fine reference mesh
        # # area = assemble(b*dx)
        # # area_fine_mesh = ...
        # # rescaling = 1.0 if np.allclose(area, 0.0) else area_fine_mesh/area
        # rescaling = 1.0

        # z, zeta = prob.adj_solutions[-1].split()
        # zeta.interpolate(rescaling*b)
        return

    def get_gauge_data(self, gauge, **kwargs):
        raise NotImplementedError("Implement in derived class")

    def check_in_domain(self, point):
        """
        Check that a `point` lies within at least one of the UTM and longitude-latitude domains.
        """
        try:
            self.default_mesh.coordinates.at(point)
        except PointNotInDomainError:
            if not hasattr(self, 'lonlat_mesh'):
                self.get_lonlat_mesh()
            self.lonlat_mesh.coordinates.at(point)

    def extract_data(self, gauge):
        """
        Extract gauge time and elevation data from file as NumPy arrays.

        Note that this isn't *raw* data because it has been converted to appropriate units using
        `preproc.py`.
        """
        data_file = os.path.join(self.resource_dir, 'gauges', gauge + '.dat')
        if not os.path.exists(data_file):
            raise IOError("Requested timeseries for gauge '{:s}' cannot be found.".format(gauge))
        times, data = [], []
        with open(data_file, 'r') as f:
            for line in f:
                time, dat = line.split()
                times.append(float(time))
                data.append(float(dat))
        return np.array(times), np.array(data)

    def sample_timeseries(self, gauge, sample=1, detide=False, timeseries=None, **kwargs):
        """
        Interpolate from gauge data. Since the data is provided at regular intervals, we use linear
        interpolation between the data points.

        Since some of the timeseries are rather noisy, there is an optional `sample` parameter, which
        averages over the specified number of datapoints before interpolating.

        Keyword arguments are passed to `scipy.interpolate.interp1d`.
        """
        if detide:
            times, data, _ = self.detide(gauge)
        else:
            times, data = self.extract_data(gauge)
        if timeseries is not None:
            assert len(data) == len(timeseries)
            data[:] = timeseries

        # Process data
        i, time_prev = 0, 0.0
        sampled_times, sampled_data, running = [], [], []
        for time, dat in zip(times, data):
            if np.isnan(dat):
                continue
            running.append(dat)
            if i % sample == 0 and i > 0:
                sampled_times.append(0.5*(time + time_prev))
                sampled_data.append(np.mean(running))
                time_prev = time
                running = []
            i += 1

        # Construct interpolant
        kwargs.setdefault('bounds_error', False)
        kwargs.setdefault('fill_value', 'extrapolate')
        interp = si.interp1d(sampled_times, sampled_data, **kwargs)

        # Shift by initial value
        self.gauges[gauge]["interpolator"] = lambda tau: interp(tau) - interp(0.0)

    def detide(self, gauge):
        """To be implemented in subclass."""
        raise NotImplementedError
