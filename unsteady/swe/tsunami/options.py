from thetis import *
from thetis.configuration import *

import numpy as np
import os
import scipy.interpolate as si

from adapt_utils.unsteady.options import CoupledOptions
from adapt_utils.unsteady.swe.tsunami.conversion import *


__all__ = ["TsunamiOptions"]


class TsunamiOptions(CoupledOptions):
    """
    Parameter class for general tsunami propagation problems.
    """
    # TODO: more doc details
    Omega = PositiveFloat(7.291e-5, help="Planetary rotation rate").tag(config=True)
    bathymetry_cap = NonNegativeFloat(30.0, allow_none=True, help="Minimum depth").tag(config=True)

    def __init__(self, regularisation=0.0, coordinate_system='utm', **kwargs):
        """
        kwarg regularisation: non-negative constant parameter used for Tikhonov regularisation
            methods for the quantity of interest.
        """
        super(TsunamiOptions, self).__init__(**kwargs)
        self.solve_swe = True
        self.solve_tracer = False
        self.force_zone_number = kwargs.get('force_zone_number', False)
        if not coordinate_system in ('utm', 'lonlat'):
            raise NotImplementedError("Coordinate system '{:s}' not recognised.".format(coordinate_system))
        self.coordinate_system = coordinate_system
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
        """
        Given a mesh in longitude-latitude coordinates, establish a corresponding mesh in UTM
        coordinates using the conversion code in `adapt_utils.unsteady.swe.tsunami.conversion`.
        """
        zone = self.force_zone_number
        self._utm_mesh = Mesh(Function(self.lonlat_mesh.coordinates))
        lon, lat = SpatialCoordinate(self._utm_mesh)
        x, y = lonlat_to_utm(lon, lat, zone)
        self._utm_mesh.coordinates.interpolate(as_vector([x, y]))

    @property
    def utm_mesh(self):
        if self.coordinate_system == 'utm':
            return self.default_mesh
        elif not hasattr(self, '_utm_mesh'):
            self.get_utm_mesh()
        return self._utm_mesh

    def get_lonlat_mesh(self, northern=True):
        """
        Given a mesh in UTM coordinates, establish a corresponding mesh in longitude-latitude
        coordinates using the conversion code in `adapt_utils.unsteady.swe.tsunami.conversion`.

        :kwarg northern: tell the UTM coordinate transformation which hemisphere we are in.
        """
        zone = self.force_zone_number
        self._lonlat_mesh = Mesh(Function(self.utm_mesh.coordinates))
        x, y = SpatialCoordinate(self._lonlat_mesh)
        lon, lat = utm_to_lonlat(x, y, zone, northern=northern, force_longitude=True)
        self._lonlat_mesh.coordinates.interpolate(as_vector([lon, lat]))

    @property
    def lonlat_mesh(self):
        if self.coordinate_system == 'lonlat':
            return self.default_mesh
        # elif not hasattr(self, '_lonlat_mesh'):
        #     self.get_lonlat_mesh()
        self.get_lonlat_mesh()
        return self._lonlat_mesh

    def set_bathymetry(self, fs=None, northern=True, force_longitude=True, **kwargs):
        """
        Derived classes should implement :attr:`read_bathymetry_file` such that it returns a 3-tuple
        of longitude, latitude and elevation data over a rectangular grid.

        If a minimum water depth has been provided via :attr:`bathymetry_cap` the this is enforced
        at this stage.

        Note on interpolation
        =====================
        We should always be cautious of accessing `f.dat.data` for a Firedrake :class:`Function` `f`,
        because it isn't parallel safe in some cases. However, as described in the documentation
        (https://firedrakeproject.org/interpolation.html#interpolation-from-external-data), what we
        do is okay in the case of interpolating external data. The interpolation is just done on all
        processors and Firedrake manages halo updates.

        :kwarg fs: :class:`FunctionSpace` for the bathymetry to live in. By default, P1 space is used
        :kwarg northern: tell the UTM coordinate transformation which hemisphere we are in.
        :kwarg force_longitude: toggle checking validity of the UTM zone.

        All other kwargs are passed to the :attr:`read_bathymetry_file` method.
        """
        if self.bathymetry_cap is not None:
            assert self.bathymetry_cap >= 0.0
        fs = fs or FunctionSpace(self.default_mesh, "CG", 1)
        bathymetry = Function(fs, name="Bathymetry")

        # Interpolate bathymetry data *in lonlat space*
        if not hasattr(self, 'bathymetry_interpolator'):
            lon, lat, elev = self.read_bathymetry_file(**kwargs)
            self.print_debug("INIT: Creating bathymetry interpolator...")
            self.bathymetry_interpolator = si.RectBivariateSpline(lat, lon, elev)

        # Insert interpolated data onto nodes of *problem domain space*
        self.print_debug("INIT: Interpolating bathymetry...")
        conversion_kwargs = {'northern': northern, 'force_longitude': force_longitude}
        for i, xy in enumerate(fs.mesh().coordinates.dat.data_ro):  # TODO: Use lonlat_mesh?
            lon, lat = utm_to_lonlat(xy[0], xy[1], self.force_zone_number, **conversion_kwargs)
            bathymetry.dat.data[i] -= self.bathymetry_interpolator(lat, lon)

        # Cap bathymetry to enforce a minimum depth
        self.print_debug("INIT: Capping bathymetry...")  # TODO: Should we really be doing this?
        if self.bathymetry_cap is not None:
            bathymetry.interpolate(max_value(self.bathymetry_cap, bathymetry))

        return bathymetry

    def set_initial_surface(self, fs=None, northern=True, force_longitude=True, **kwargs):
        """
        Derived classes should implement :attr:`read_surface_file` such that it returns a 3-tuple
        of longitude, latitude and elevation data over a rectangular grid.

        Note on interpolation
        =====================
        We should always be cautious of accessing `f.dat.data` for a Firedrake :class:`Function` `f`,
        because it isn't parallel safe in some cases. However, as described in the documentation
        (https://firedrakeproject.org/interpolation.html#interpolation-from-external-data), what we
        do is okay in the case of interpolating external data. The interpolation is just done on all
        processors and Firedrake manages halo updates.

        :kwarg fs: :class:`FunctionSpace` for the bathymetry to live in. By default, P1 space is used
        :kwarg northern: tell the UTM coordinate transformation which hemisphere we are in.
        :kwarg force_longitude: toggle checking validity of the UTM zone.

        All other kwargs are passed to the :attr:`read_surface_file` method.
        """
        fs = fs or FunctionSpace(self.default_mesh, "CG", 1)
        initial_surface = Function(fs, name="Initial free surface")

        # Interpolate bathymetry data *in lonlat space*
        if not hasattr(self, 'surface_interpolator'):
            self.print_debug("INIT: Creating surface interpolator...")
            lon, lat, elev = self.read_surface_file(**kwargs)
            self.surface_interpolator = si.RectBivariateSpline(lat, lon, elev)

        # Insert interpolated data onto nodes of *problem domain space*
        self.print_debug("INIT: Interpolating initial surface...")
        conversion_kwargs = {'northern': northern, 'force_longitude': force_longitude}
        for i, xy in enumerate(fs.mesh().coordinates.dat.data_ro):  # TODO: use lonlat_mesh?
            lon, lat = utm_to_lonlat(xy[0], xy[1], self.force_zone_number, **conversion_kwargs)
            initial_surface.dat.data[i] = self.surface_interpolator(lat, lon)

        return initial_surface

    def set_initial_condition(self, prob):
        """
        Initialise the hydrodynamics in :class:`AdaptiveProblem` `prob` using the
        :attr:`set_initial_surface` method.

        We follow the standard practice in the tsunami modelling literature by assuming zero initial
        velocity.
        """

        # Read initial surface data from file
        surf = self.set_initial_surface(prob.P1[0])

        # Set initial condition
        self.print_debug("INIT: Setting initial condition...")
        u, eta = prob.fwd_solutions[0].split()
        u.assign(0.0)  # (Naively) assume zero initial velocity
        eta.interpolate(surf)  # Initial surface from inversion data

        # Subtract from the bathymetry field
        self.subtract_surface_from_bathymetry(prob, surf=surf)
        return surf

    def subtract_surface_from_bathymetry(self, prob, surf=None):
        """
        It is common practice in the tsunami modelling literature to subtract the initial surface
        from the bathymetry field. This is in line with the fact the earthquake disturbs the bed.
        In the hydrostatic case, we can assume that this translates to an idential disturbance to
        the ocean surface.

        Note that we reset the bathymetry first, in case this method has already been called.
        """
        self.print_debug("INIT: Subtracting initial surface from bathymetry field...")

        # Reset bathymetry
        fs = prob.bathymetry[0].function_space()
        prob.bathymetry[0] = self.set_bathymetry(fs)

        # Project bathymetry into P1 space
        P1 = prob.P1[0]
        b = project(prob.bathymetry[0], P1)

        # Interpolate free surface into P1 space
        if surf is None:
            surf = interpolate(prob.fwd_solutions[0].split()[1], P1)

        # Subtract surface from bathymetry
        b -= surf

        # Project updated bathymetry onto each mesh
        for i in range(self.num_meshes):
            prob.bathymetry[i].project(b)

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
        raise NotImplementedError("Should be implemented in derived class.")

    def set_terminal_condition(self, prob):  # TODO: For hazard case
        prob.adj_solutions[-1].assign(0.0)

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

        If the sampling frequency is set to unity then the raw data is used.

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
        time_prev = 0.0
        sampled_times, sampled_data, running = [], [], []
        for i, (time, dat) in enumerate(zip(times, data)):
            if np.isnan(dat):
                continue
            if sample == 1:
                sampled_times.append(time)
                sampled_data.append(dat)
            else:
                running.append(dat)
                if i % sample == 0 and i > 0:
                    sampled_times.append(0.5*(time + time_prev))
                    sampled_data.append(np.mean(running))
                    time_prev = time
                    running = [dat]

        # Construct interpolant
        kwargs.setdefault('bounds_error', False)
        kwargs.setdefault('fill_value', 'extrapolate')
        interp = si.interp1d(sampled_times, sampled_data, **kwargs)

        # Shift by initial value
        arrival_time = self.gauges[gauge]["arrival_time"]
        self.gauges[gauge]["interpolator"] = lambda tau: interp(tau) - interp(arrival_time)

    def detide(self, gauge):
        """To be implemented in subclass."""
        raise NotImplementedError

    def check_cfl_criterion(self, prob, i, error_factor=None):
        r"""
        Check whether the CFL criterion is met under the current discretisation, using the minimum
        mesh element size and the timestep. Fluid speed is taken to be the celerity,
        :math:`\sqrt{gb_{\max}}`, where :math:`b_{\max}` is the maximum water depth.

        :arg prob: the :class:`AdaptiveTsunamiProblem` object.
        :arg i: mesh number from sequence.
        :kwarg error_factor: optionally raise an error if the CFL criterion is not met.
        """
        if error_factor is None and not self.debug:
            return
        self.print_debug("INIT: Computing CFL number on mesh {:d}...".format(i))
        b = prob.bathymetry[i].vector().gather().max()
        g = self.g.values()[0]
        celerity = np.sqrt(g*b)
        dx = interpolate(CellDiameter(prob.meshes[i]), prob.P0[i]).vector().gather().min()
        cfl = celerity*self.dt/dx
        msg = "INIT:   dx = {:.4e}  dt = {:.4e}  CFL number = {:.4e} {:1s} 1"
        self.print_debug(msg.format(dx, self.dt, cfl, '<' if cfl < 1 else '>'))
        if error_factor is not None and cfl >= error_factor:
            if np.isclose(error_factor, 1.0):
                raise ValueError("CFL criterion not met! (CFL number {:.4e})".format(cfl))
            else:
                msg = "Relaxed CFL criterion not met! (CFL number {:.4e} > {:.4e})"
                raise ValueError(msg.format(cfl, error_factor))
