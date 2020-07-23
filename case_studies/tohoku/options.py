from thetis import *
from thetis.configuration import *

import os
import netCDF4

from adapt_utils.unsteady.swe.tsunami.options import TsunamiOptions
from adapt_utils.unsteady.swe.tsunami.conversion import from_latlon


__all__ = ["TohokuOptions", "TohokuBoxBasisOptions", "TohokuGaussianBasisOptions", "TohokuOkadaOptions"]


class TohokuOptions(TsunamiOptions):
    """
    Setup for model of the Tohoku tsunami which struck the east coast of Japan in 2011, leading to
    the meltdown of Daiichi nuclear power plant, Fukushima.


    Data sources:
      * Bathymetry data extracted from GEBCO (https://www.gebco.net/).
      * Initial free surface elevation field generated by inversion on tide gauge data by
        [Saito et al.].
      * Timeseries for gauges P02 and P06 obtained via personal communication with T. Saito.
      * Timeseries for gauges 801-806 obtained from the Japanese Port and Airport Research
          Institute (PARI).
      * Timeseries for gauges KPG1 and KPG2 obtained from the Japanese Agency for Marine-Earth
          Science and Technology (JAMSTEC) via http://www.jamstec.go.jp/scdc/top_e.html.
      * Timeseries for gauges 21401, 21413, 21418 and 21419 obtained from the US National Oceanic
        and Atmospheric Administration (NOAA) via https://www.ndbc.noaa.gov.


    [Saito et al.] T. Saito, Y. Ito, D. Inazu, R. Hino, "Tsunami source of the 2011 Tohoku‐Oki
                   earthquake, Japan: Inversion analysis based on dispersive tsunami simulations",
                   Geophysical Research Letters (2011), 38(7).
    """
    def __init__(self, mesh=None, level=0, save_timeseries=False, synthetic=False, **kwargs):
        """
        :kwarg mesh: optionally use a custom mesh.
        :kwarg level: mesh resolution level, to be used if no mesh is provided.
        :kwarg save_timeseries: :type:`bool` toggling the extraction and storage of timeseries data.
        :kwarg synthetic: :type:`bool` toggling whether real or synthetic timeseries data are used.
        :kwarg qoi_scaling: custom scaling for quantity of interest (defaults to unity).
        :kwarg base_viscosity: :type:`float` value to be assigned to constant viscosity field.
        :kwarg postproc: :type:`bool` value toggling whether to use an initial mesh which has been
            postprocessed using Pragmatic (see `resources/meshes/postproc.py`.)
        :kwarg locations: a list of strings indicating locations of interest.
        :kwarg radii: a list of distances indicating radii around the locations of interest, thereby
            determining regions of interest for use in hazard assessment QoIs.
        """
        self.force_zone_number = 54
        super(TohokuOptions, self).__init__(**kwargs)
        self.save_timeseries = save_timeseries
        self.synthetic = synthetic
        self.qoi_scaling = kwargs.get('qoi_scaling', 1.0)

        # Stabilisation
        self.use_automatic_sipg_parameter = False
        self.sipg_parameter = None
        self.base_viscosity = kwargs.get('base_viscosity', 0.0)

        # Mesh
        self.print_debug("Loading mesh...")
        self.resource_dir = os.path.join(os.path.dirname(__file__), 'resources')
        self.level = level
        self.meshfile = os.path.join(self.resource_dir, 'meshes', 'Tohoku{:d}'.format(self.level))
        postproc = kwargs.get('postproc', True)
        if mesh is None:
            if postproc:
                from firedrake.petsc import PETSc

                newplex = PETSc.DMPlex().create()
                newplex.createFromFile(self.meshfile + '.h5')
                self.default_mesh = Mesh(newplex)
            else:
                self.default_mesh = Mesh(self.meshfile + '.msh')
        else:
            self.default_mesh = mesh
        self.print_debug("Done!")

        # Fields
        self.friction = 'manning'
        self.friction_coeff = 0.025

        # Timestepping
        # ============
        #
        # NOTES:
        #  * Export once per minute for 24 minutes.
        #  * There is a trade-off between having an unneccesarily small timestep and being able to
        #    represent the gauge timeseries profiles.
        self.timestepper = 'CrankNicolson'
        # self.dt = 5.0
        self.dt = 60.0*0.5**level
        self.dt_per_export = int(60.0/self.dt)
        # self.start_time = 15*60.0
        self.start_time = 0.0
        self.end_time = 24*60.0
        # self.end_time = 60*60.0

        # Compute CFL number
        if self.debug:
            P0 = FunctionSpace(self.default_mesh, "DG", 0)
            P1 = FunctionSpace(self.default_mesh, "CG", 1)
            b = self.set_bathymetry(P1).vector().gather().max()
            g = self.g.values()[0]
            celerity = sqrt(g*b)
            dx = interpolate(CellDiameter(self.default_mesh), P0).vector().gather().max()
            cfl = celerity*self.dt/dx
            msg = "dx = {:.4e}  dt = {:.4e}  CFL number = {:.4e} {:1s} 1"
            print_output(msg.format(dx, self.dt, cfl, '<' if cfl < 1 else '>'))

        # Gauges where we have timeseries
        self.gauges = {
            "P02": {"lonlat": (142.5016, 38.5002), "operator": "Tohoku University"},  # TODO: depth
            "P06": {"lonlat": (142.5838, 38.6340), "operator": "Tohoku University"},  # TODO: depth
            "801": {"lonlat": (141.6856, 38.2325), "operator": "PARI"},  # TODO: depth
            "802": {"lonlat": (142.0969, 39.2586), "operator": "PARI"},  # TODO: depth
            "803": {"lonlat": (141.8944, 38.8578), "operator": "PARI"},  # TODO: depth
            "804": {"lonlat": (142.1867, 39.6272), "operator": "PARI"},  # TODO: depth
            "806": {"lonlat": (141.1856, 36.9714), "operator": "PARI"},  # TODO: depth
            "KPG1": {"lonlat": (144.4375, 41.7040), "depth": 2218.0, "operator": "JAMSTEC"},
            "KPG2": {"lonlat": (144.8485, 42.2365), "depth": 2210.0, "operator": "JAMSTEC"},
            "MPG1": {"lonlat": (134.4753, 32.3907), "depth": 2308.0, "operator": "JAMSTEC"},
            "MPG2": {"lonlat": (134.3712, 32.6431), "depth": 1507.0, "operator": "JAMSTEC"},
            # TODO: VCM1 (NEID)
            # TODO: VCM3 (NEID)
            "21401": {"lonlat": (152.583, 42.617), "operator": "NOAA"},
            "21413": {"lonlat": (152.132, 30.533), "depth": 5880.0, "operator": "NOAA"},
            "21418": {"lonlat": (148.655, 38.735), "depth": 5777.0, "operator": "NOAA"},
            "21419": {"lonlat": (155.717, 44.435), "depth": 5282.0, "operator": "NOAA"},
        }
        self.pressure_gauges = ("P02", "P06", "KPG1", "KPG2", "MPG1", "MPG2", "21418")
        # self.pressure_gauges += ("VCM1", "VCM3")
        self.gps_gauges = ("801", "802", "803", "804", "806")

        # Possible coastal locations of interest, including major cities and nuclear power plants
        locations = kwargs.get('locations', ["Fukushima Daiichi", ])
        radii = kwargs.get('radii', [50.0e+03, ])
        locations_of_interest = {
            "Fukushima Daiichi": {"lonlat": (141.0281, 37.4213)},
            "Onagawa": {"lonlat": (141.5008, 38.3995)},
            "Fukushima Daini": {"lonlat": (141.0249, 37.3166)},
            "Tokai": {"lonlat": (140.6067, 36.4664)},
            "Hamaoka": {"lonlat": (138.1433, 34.6229)},
            "Tohoku": {"lonlat": (141.3903, 41.1800)},
            "Tokyo": {"lonlat": (139.6917, 35.6895)},
        }
        self.locations_of_interest = {loc: locations_of_interest[loc] for loc in locations}
        radii = {locations[i]: r for i, r in enumerate(radii)}

        # Convert coordinates to UTM and create timeseries array
        for loc in (self.gauges, self.locations_of_interest):
            for l in loc:
                loc[l]["timeseries"] = []
                lon, lat = loc[l]["lonlat"]
                loc[l]["utm"] = from_latlon(lat, lon, force_zone_number=54)
                loc[l]["coords"] = loc[l]["utm"]

        # Check validity of gauge coordinates
        gauges = list(self.gauges.keys())
        for gauge in gauges:
            try:
                self.default_mesh.coordinates.at(self.gauges[gauge]['coords'])
            except PointNotInDomainError:
                self.print_debug("NOTE: Gauge {:5s} is not in the domain; removing now".format(gauge))
                self.gauges.pop(gauge)  # Some gauges aren't within the domain

        # Regions of interest
        loi = self.locations_of_interest
        self.region_of_interest = [loi[loc]["coords"] + (radii[loc], ) for loc in loi]

    def read_bathymetry_file(self, source='etopo1'):
        self.print_debug("Reading bathymetry file...")
        if source == 'gebco':
            nc = netCDF4.Dataset(os.path.join(self.resource_dir, 'bathymetry', 'gebco.nc'), 'r')
            lon = nc.variables['lon'][:]
            lat = nc.variables['lat'][:-1]
            elev = nc.variables['elevation'][:-1, :]
        elif source == 'etopo1':
            nc = netCDF4.Dataset(os.path.join(self.resource_dir, 'bathymetry', 'etopo1.nc'), 'r')
            lon = nc.variables['lon'][:]
            lat = nc.variables['lat'][:]
            elev = nc.variables['Band1'][:, :]
        else:
            raise ValueError("Bathymetry data source {:s} not recognised.".format(source))
        nc.close()
        self.print_debug("Done!")
        return lon, lat, elev

    def read_surface_file(self, zeroed=True):
        self.print_debug("Reading initial surface file...")
        fname = 'surf'
        if zeroed:
            fname = '_'.join([fname, 'zeroed'])
        nc = netCDF4.Dataset(os.path.join(self.resource_dir, 'surf', fname + '.nc'), 'r')
        lon = nc.variables['lon' if zeroed else 'x'][:]
        lat = nc.variables['lat' if zeroed else 'y'][:]
        elev = nc.variables['z'][:, :]
        nc.close()
        self.print_debug("Done!")
        return lon, lat, elev

    def _get_update_forcings_forward(self, prob, i):
        from adapt_utils.misc import ellipse
        from adapt_utils.case_studies.tohoku.resources.gauges.sample import sample_timeseries

        self.J = self.get_regularisation_term(prob)
        scaling = Constant(0.5*self.qoi_scaling)
        weight = Constant(1.0)  # Quadrature weight for time integration scheme

        # These will be updated by the checkpointing routine
        u, eta = prob.fwd_solutions[i].split()

        # Account for timeseries shift
        # ============================
        #   This can be troublesome business. With synthetic data, we can actually get away with not
        #   shifting, provided we are solving the linearised equations. However, in the nonlinear case
        #   and when using real data, we should make sure that the timeseries are comparable by always
        #   shifting them by the initial elevation at each gauge. This isn't really a problem for the
        #   continuous adjoint method. However, for discrete adjoint we need to annotate the initial
        #   gauge evaluation. Until point evaluation is annotated in Firedrake, the best thing is to
        #   just use the initial surface *field*. This does modify the QoI, but it shouldn't be too much
        #   of a problem if the mesh is sufficiently fine (and hence the indicator regions are
        #   sufficiently small.
        if self.synthetic:
            self.eta_init = Constant(0.0)
        else:
            # TODO: Use point evaluation once it is annotated
            self.eta_init = Function(eta.function_space()).assign(eta)

        mesh = eta.function_space().mesh()
        radius = 20.0e+03*pow(0.5, self.level)  # The finer the mesh, the smaller the region
        self.times = []
        for gauge in self.gauges:
            gauge_dat = self.gauges[gauge]
            gauge_dat["obs"] = Constant(0.0)  # Constant associated with free surface observations

            # Setup interpolator
            sample = 1 if gauge[0] == '8' else 60
            gauge_dat["interpolator"] = sample_timeseries(gauge, sample=sample)

            # Assemble an area-normalised indicator function
            x, y = gauge_dat["coords"]
            disc = ellipse([(x, y, radius,), ], mesh)
            area = assemble(disc*dx, annotate=False)
            gauge_dat["indicator"] = interpolate(disc/area, prob.P0[i])
            I = gauge_dat["indicator"]

            # Get initial pointwise and area averaged values
            gauge_dat["init"] = eta.at(gauge_dat["coords"])
            gauge_dat["init_smooth"] = assemble(I*eta*dx, annotate=False)

            # Initialise arrays for storing timeseries
            if self.save_timeseries:
                gauge_dat["timeseries"] = []
                gauge_dat["timeseries_smooth"] = []
                gauge_dat["diff"] = []
                gauge_dat["diff_smooth"] = []
                if not self.synthetic or "data" not in self.gauges[gauge]:
                    gauge_dat["data"] = []

        def update_forcings(t):
            """
            Evaluate free surface elevation at gauges, compute the contribution to the quantity of
            interest from the current timestep and store data in :attr:`self.gauges`.

            NOTE: `update_forcings` is called one timestep along so we shift time back.
            """
            dt = self.dt
            t = t - dt
            weight.assign(0.5*dt if t < 0.5*dt or t >= self.end_time - 0.5*dt else dt)
            self.times.append(t)
            for gauge in self.gauges:
                gauge_dat = self.gauges[gauge]
                I = gauge_dat["indicator"]

                # Point evaluation and average value at gauges
                if self.save_timeseries:
                    eta_discrete = eta.at(gauge_dat["coords"]) - gauge_dat["init"]
                    gauge_dat["timeseries"].append(eta_discrete)
                    eta_smoothed = assemble(I*eta*dx, annotate=False) - gauge_dat["init_smooth"]
                    gauge_dat["timeseries_smooth"].append(eta_smoothed)
                if self.synthetic and gauge_dat["data"] == []:
                    continue

                # Read data
                interpolator = gauge_dat["interpolator"]
                obs = gauge_dat["data"][prob.iteration] if self.synthetic else float(interpolator(t))
                gauge_dat["obs"].assign(obs)
                if self.save_timeseries:
                    if not self.synthetic:
                        gauge_dat["data"].append(obs)

                    # Discrete form of error
                    gauge_dat["diff"].append(0.5*(eta_discrete - gauge_dat["obs"].dat.data[0])**2)

                # Continuous form of error
                #   NOTES:
                #     * The initial free surface *field* is subtracted in some cases.
                #     * Factor of half is included in `scaling`
                #     * Quadrature weights and timestep included in `weight`
                diff = I*(eta - self.eta_init - gauge_dat["obs"])**2
                self.J += assemble(scaling*weight*diff*dx)
                if self.save_timeseries:
                    gauge_dat["diff_smooth"].append(assemble(diff*dx, annotate=False))

        return update_forcings

    def _get_update_forcings_adjoint(self, prob, i):
        expr = 0
        # scaling = Constant(self.qoi_scaling)
        scaling = Constant(2*self.qoi_scaling)  # TODO: TEMPORARY FACTOR OF 2
        msg = "CHECKPOINT LOAD:  u norm: {:.8e}  eta norm: {:.8e} (iteration {:d})"

        # Gauge data (to be loaded from checkpoint)
        u_saved, eta_saved = prob.fwd_solutions[i].split()

        # Construct an expression for the RHS of the adjoint continuity equation
        #   NOTE: The initial free surface *field* is subtracted in some cases.
        for gauge in self.gauges:
            gauge_dat = self.gauges[gauge]
            expr += scaling*gauge_dat["indicator"]*(eta_saved - self.eta_init - gauge_dat["obs"])
        k_u, k_eta = prob.kernels[i].split()

        def update_forcings(t):
            """
            Evaluate RHS for adjoint equations using forward solution data retreived from checkpoint.

            NOTE: `update_forcings` is called one timestep along so we shift time.
            """
            t = t + self.dt
            if self.debug:
                print_output(msg.format(norm(u_saved), norm(eta_saved), prob.iteration))

            # Get timeseries data, implicitly modifying the expression
            for gauge in self.gauges:
                gauge_dat = self.gauges[gauge]
                obs = gauge_dat["data"][prob.iteration-1] if self.synthetic else float(interpolator(t))
                gauge_dat["obs"].assign(obs)

            # Interpolate expression onto RHS
            k_eta.interpolate(expr)
            if self.debug and prob.iteration % self.dt_per_export == 0:
                prob.kernel_file.write(k_eta)

        return update_forcings

    def get_update_forcings(self, prob, i, adjoint=False):
        if adjoint:
            return self._get_update_forcings_adjoint(prob, i)
        else:
            return self._get_update_forcings_forward(prob, i)

    def get_regularisation_term(self, prob):
        """
        Tikhonov regularisation term that enforces spatial smoothness in the initial surface.

        NOTES:
          * Assumes the forward model has been initialised but not taken any iterations.
          * Assumes a linear relationship between control parameters and source.
        """
        from adapt_utils.adapt.recovery import construct_gradient

        # Set regularisation parameter
        if np.isclose(self.regularisation, 0.0):
            return 0
        alpha = Constant(self.regularisation)

        # Recover gradient of initial surface using L2 projection
        u0, eta0 = prob.fwd_solutions[0].split()
        deta0dx = construct_gradient(eta0, op=self)

        # Compute regularisation term
        R = assemble(0.5*alpha*inner(deta0dx, deta0dx)*dx)
        print_output("Regularisation term = {:.4e}".format(R))
        self.regularisation_term = R

        # Compute components of gradient of regularisation term
        self.regularisation_term_gradients = []
        for i in range(len(self.basis_functions)):
            dphidx = construct_gradient(self.basis_functions[i].split()[1], op=self)
            dRdm = assemble(alpha*inner(dphidx, deta0dx)*dx)
            if len(self.basis_functions) == 1:
                print_output("Gradient of regularisation term = {:.4e}".format(dRdm))
            else:
                print_output("(Gradient of regularisation term)[{:d}] = {:.4e}".format(i, dRdm))
            self.regularisation_term_gradients.append(dRdm)

        return R

    def set_boundary_conditions(self, prob, i):
        ocean_tag = 100
        coast_tag = 200
        fukushima_tag = 300
        boundary_conditions = {
            'shallow_water': {
                coast_tag: {'un': Constant(0.0)},
                fukushima_tag: {'un': Constant(0.0)},
                ocean_tag: {'un': Constant(0.0), 'elev': Constant(0.0)},  # Weakly reflective
            }
        }
        # TODO: Sponge at ocean boundary?
        #        - Could potentially do this by defining a gradation to the ocean boundary with a
        #          different PhysID.
        return boundary_conditions

    def annotate_plot(self, axes, coords="utm", gauges=False, fontsize=12):
        """
        Annotate `axes` in coordinate system `coords` with all gauges or locations of interest, as
        determined by the Boolean kwarg `gauges`.
        """
        try:
            assert coords in ("lonlat", "utm")
        except AssertionError:
            raise ValueError("Coordinate system {:s} not recognised.".format(coords))
        dat = self.gauges if gauges else self.locations_of_interest
        offset = 40.0e+03  # Offset by an extra 40 km
        for loc in dat:
            x, y = dat[loc][coords]
            xytext = (x + offset, y)
            color = "indigo"
            ha = "right"
            va = "center"
            if loc == "Fukushima Daini":
                continue
            elif loc == "Fukushima Daiichi":
                loc = "Fukushima"
            elif "80" in loc:
                color = "C3"
                xytext = (x - offset, y)
                ha = "right"
            elif gauges:
                color = "navy"
                xytext = (x + offset, y)
                ha = "left"
                if loc == "P02":
                    xytext = (x + offset, y - offset)
                    va = "bottom"
                elif loc == "P06":
                    xytext = (x + offset, y + offset)
                    va = "top"
            axes.plot(x, y, 'x', color=color)
            axes.annotate(
                loc, xy=(x, y), xycoords='data', xytext=xytext,
                fontsize=fontsize, color=color, ha=ha, va=va
            )


class TohokuBoxBasisOptions(TohokuOptions):
    """
    Initialise the free surface with an initial condition consisting of an array of rectangular
    indicator functions, each scaled by a control parameter. Setups of this type have been used by
    numerous authors in the tsunami modelling literature.

    The source region centre is predefined. In the 1D case the basis function is centred at the same
    point. In the case of multiple basis functions they are distributed linearly both perpendicular and
    parallel to the fault axis. Note that the support does not overlap, unlike with radial basis
    functions.

    The 1D case is useful for inversion experiments because the control parameter space is one
    dimensional, meaning it can be easily plotted.
    """
    def __init__(self, **kwargs):
        """
        :kwarg control_parameters: a list of values to use for the basis function coefficients.
        :kwarg centre_x: x-coordinate of centre of source region in UTM coordinates [m].
        :kwarg centre_y: y-coordinate of centre of source region in UTM coordinates [m].
        :kwarg nx: number of basis functions along strike direction (i.e. along the fault).
        :kwarg ny: number of basis functions perpendicular to the strike direction.
        :kwarg radius_x: radius of basis function along strike direction [m].
        :kwarg radius_y: radius of basis function perpendicular to the strike direction [m].
        :kwarg angle: angle of fault to north [radians].
        """
        super(TohokuBoxBasisOptions, self).__init__(**kwargs)
        self.nx = kwargs.get('nx', 1)
        self.ny = kwargs.get('ny', 1)
        N_b = self.nx*self.ny
        control_parameters = kwargs.get('control_parameters', [0.0 for i in range(N_b)])
        N_c = len(control_parameters)
        if N_c != N_b:
            raise ValueError("{:d} controls inconsistent with {:d} basis functions".format(N_c, N_b))

        # Parametrisation of source region
        self.centre_x = kwargs.get('centre_x', 0.7e+06)
        self.centre_y = kwargs.get('centre_y', 4.2e+06)
        self.radius_x = kwargs.get('radius_x', 96e+03 if self.nx == 1 else 48.0e+03)
        self.radius_y = kwargs.get('radius_y', 48e+03 if self.ny == 1 else 24.0e+03)

        # Parametrisation of source basis
        R = FunctionSpace(self.default_mesh, "R", 0)
        self.control_parameters = []
        for i in range(N_c):
            self.control_parameters.append(Function(R, name="Control parameter {:d}".format(i)))
            self.control_parameters[i].assign(control_parameters[i])
        self.angle = kwargs.get('angle', 7*pi/12)

    def set_initial_condition(self, prob):
        from adapt_utils.misc import box, rotation_matrix

        # Gather parameters
        x0, y0 = self.centre_x, self.centre_y  # Centre of basis region
        nx, ny = self.nx, self.ny              # Number of basis functions in each component direction
        N = nx*ny                              # Total number of basis functions
        rx, ry = self.radius_x, self.radius_y  # Radius of each basis function in each direction
        angle = self.angle                     # Angle by which to rotate basis array

        # Setup array coordinates
        X = np.linspace((1 - nx)*rx, (nx - 1)*rx, nx)
        Y = np.linspace((1 - ny)*ry, (ny - 1)*ry, ny)

        # Assemble an array of Gaussian basis functions, rotated by specified angle
        self.basis_functions = [Function(prob.V[0]) for i in range(N)]
        R = rotation_matrix(-angle)
        for j, y in enumerate(Y):
            for i, x in enumerate(X):
                psi, phi = self.basis_functions[i + j*nx].split()
                x_rot, y_rot = tuple(np.array([x0, y0]) + np.dot(R, np.array([x, y])))
                phi.interpolate(box([(x_rot, y_rot, rx, ry), ], prob.meshes[0], rotation=angle))

        # Assemble initial surface
        #   NOTE: The calculation is split up for large arrays in order to avoid the UFL recursion limit
        prob.fwd_solutions[0].assign(0.0)
        l = 100
        for n in range(0, N, l):
            expr = sum(m*g for m, g in zip(self.control_parameters[n:n+l], self.basis_functions[n:n+l]))
            prob.fwd_solutions[0].assign(prob.fwd_solutions[0] + project(expr, prob.V[0]))


class TohokuGaussianBasisOptions(TohokuOptions):
    """
    Initialise the free surface with an initial condition consisting of an array of Gaussian basis
    functions, each scaled by a control parameter. The setup with a 10 x 13 array was presented in
    [Saito et al. 2011].

    The source region centre is predefined. In the 1D case the basis function is centred at the same
    point. In the case of multiple basis functions they are distributed linearly both perpendicular and
    parallel to the fault axis. Note that support of basis functions is overlapping, unlike the case
    where indicator functions are used.

    The 1D case is useful for inversion experiments because the control parameter space is one
    dimensional, meaning it can be easily plotted.


    [Saito et al.] T. Saito, Y. Ito, D. Inazu, R. Hino, "Tsunami source of the 2011 Tohoku‐Oki
                   earthquake, Japan: Inversion analysis based on dispersive tsunami simulations",
                   Geophysical Research Letters (2011), 38(7).
    """
    def __init__(self, **kwargs):
        """
        :kwarg control_parameters: a list of values to use for the basis function coefficients.
        :kwarg centre_x: x-coordinate of centre of source region in UTM coordinates [m].
        :kwarg centre_y: y-coordinate of centre of source region in UTM coordinates [m].
        :kwarg extent_x: extent of source region along the strike direction (i.e. along the fault) [m].
        :kwarg extent_y: extent of source region perpendicular to the strike direction [m].
        :kwarg nx: number of basis functions along strike direction.
        :kwarg ny: number of basis functions perpendicular to the strike direction.
        :kwarg radius_x: radius of basis function along strike direction [m].
        :kwarg radius_y: radius of basis function perpendicular to the strike direction [m].
        :kwarg angle: angle of fault to north [radians].
        """
        super(TohokuGaussianBasisOptions, self).__init__(**kwargs)
        self.nx = kwargs.get('nx', 1)
        self.ny = kwargs.get('ny', 1)
        N_b = self.nx*self.ny
        control_parameters = kwargs.get('control_parameters', [0.0 for i in range(N_b)])
        N_c = len(control_parameters)
        if N_c != N_b:
            raise ValueError("{:d} controls inconsistent with {:d} basis functions".format(N_c, N_b))

        # Parametrisation of source region
        self.centre_x = kwargs.get('centre_x', 0.7e+06)
        self.centre_y = kwargs.get('centre_y', 4.2e+06)
        self.extent_x = kwargs.get('extent_x', 560.0e+03)
        self.extent_y = kwargs.get('extent_y', 240.0e+03)

        # Parametrisation of source basis
        self.radius_x = kwargs.get('radius_x', 96e+03 if self.nx == 1 else 48.0e+03)
        self.radius_y = kwargs.get('radius_y', 48e+03 if self.ny == 1 else 24.0e+03)
        R = FunctionSpace(self.default_mesh, "R", 0)
        self.control_parameters = []
        for i in range(N_c):
            self.control_parameters.append(Function(R, name="Control parameter {:d}".format(i)))
            self.control_parameters[i].assign(control_parameters[i])
        self.angle = kwargs.get('angle', 7*pi/12)

    def set_initial_condition(self, prob):
        from adapt_utils.misc import gaussian, rotation_matrix

        # Gather parameters
        x0, y0 = self.centre_x, self.centre_y  # Centre of basis region
        nx, ny = self.nx, self.ny              # Number of basis functions in each component direction
        N = nx*ny                              # Total number of basis functions
        rx, ry = self.radius_x, self.radius_y  # Radius of each basis function in each direction
        dx, dy = self.extent_x, self.extent_y  # Extent of basis region in each component direction
        angle = self.angle                     # Angle by which to rotate basis array

        # Setup array coordinates
        X = np.array([0.0, ]) if nx == 1 else np.linspace(-0.5*dx, 0.5*dx, nx)
        Y = np.array([0.0, ]) if ny == 1 else np.linspace(-0.5*dy, 0.5*dy, ny)

        # Assemble an array of Gaussian basis functions, rotated by specified angle
        self.basis_functions = [Function(prob.V[0]) for i in range(N)]
        R = rotation_matrix(-angle)
        for j, y in enumerate(Y):
            for i, x in enumerate(X):
                psi, phi = self.basis_functions[i + j*nx].split()
                x_rot, y_rot = tuple(np.array([x0, y0]) + np.dot(R, np.array([x, y])))
                phi.interpolate(gaussian([(x_rot, y_rot, rx, ry), ], prob.meshes[0], rotation=angle))

        # Assemble initial surface
        #   NOTE: The calculation is split up for large arrays in order to avoid the UFL recursion limit
        prob.fwd_solutions[0].assign(0.0)
        l = 100
        for n in range(0, N, 100):
            expr = sum(m*g for m, g in zip(self.control_parameters[n:n+l], self.basis_functions[n:n+l]))
            prob.fwd_solutions[0].assign(prob.fwd_solutions[0] + project(expr, prob.V[0]))


class TohokuOkadaOptions(TohokuOptions):
    """
    Initialise the free surface with an initial condition generated using Okada functions.

    Note that, unlike in the basis comprised of an array of indicator functions or Gaussians, the
    relationship between the control parameters and the initial surface is nonlinear. In addition,
    zero is not a feasible initial guess for the Okada parameters, meaning some physical intuition is
    required in order to set up the problem.

    Control parameters comprise of the following list:
      * Focal depth - depth of the top of the fault plane [m].
      * Fault length - length of the fault plane [m].
      * Fault width - width of the fault plane [m].
      * Slip - average displacement [m].
      * Strike angle - angle from North of fault [radians].
      * Dip angle - angle from horizontal [radians].
      * Rake angle - slip of one fault block compared to another [radians].
      * Focus x-coordinate - in UTM coordinates [m].
      * Focus y-coordinate - in UTM coordinates [m].
    """
    def __init__(self, **kwargs):
        """
        :kwarg control_parameters: a list of values to use for the basis function coefficients.
        :kwarg centre_x: x-coordinate of centre of source region in UTM coordinates.
        :kwarg centre_y: y-coordinate of centre of source region in UTM coordinates.
        :kwarg extent_x: extent of source region along the strike direction (i.e. along the fault).
        :kwarg extent_y: extent of source region perpendicular to the strike direction.
        :kwarg nx: number of sub-faults along strike direction.
        :kwarg ny: number of sub-faults perpendicular to the strike direction.
        :kwarg fault_type: choose fault type from 'sinusoidal', 'average', 'circular'.
        :kwarg fault_asymmetry: asymmetry of fault in the sinusoidal case. 0.5 corresponds to symmetric,
            whilst 0 and 1 correspond to fully asymmetric.
        """
        super(TohokuOkadaOptions, self).__init__(**kwargs)
        self.control_parameters = kwargs.get('control_parameters')

        # Extract Okada parameters
        assert len(self.control_parameters) == 9
        self.focal_depth, self.fault_length, self.fault_width, self.slip, \
            self.strike_angle, self.dip_angle, self.rake_angle, \
            self.centre_x, self.centre_y = self.control_parameters
        # TODO: Check validity of controls

        # Pre-compute trigonometric functions
        self.sin_strike = np.sin(self.strike_angle)
        self.cos_strike = np.cos(self.strike_angle)
        self.sin_dip = np.sin(self.dip_angle)
        self.cos_dip = np.cos(self.dip_angle)
        self.sin_rake = np.sin(self.rake_angle)
        self.cos_rake = np.cos(self.rake_angle)

        # Parametrisation of source region
        self.extent_x = kwargs.get('extent_x', 560.0e+03)
        self.extent_y = kwargs.get('extent_y', 240.0e+03)
        self.fault_type = kwargs.get('fault_type', 'average')
        assert self.fault_type in ('average', 'sinusoidal', 'circular')
        self.fault_asymmetry = kwargs.get('fault_asymmetry', 0.35)
        if self.fault_type == 'average':
            self.fault_asymmetry = None

        # Numbers of sub-faults in each direction
        self.nx = kwargs.get('nx', 20)
        self.ny = kwargs.get('ny', 20)

        # Other parameters
        self.poisson_ratio = 0.25

    def get_fault_length(x, y):
        """
        :arg x: distance along strike direction
        :arg y: distance perpendicular to strike direction.
        """
        if self.fault_type == 'average':
            return self.slip
        else:
            raise NotImplementedError  # TODO

    def _strike_slip(self, y1, y2, q):
        # TODO: doc; copied from `geoclaw/src/python/geoclaw/dtopotools.py`
        sn = self.sin_dip
        cs = self.cos_dip

        dbar = y2*sn - q*cs
        r = np.sqrt(y1**2 + y2**2 + q**2)
        a4 = 2.0*self.poisson_ratio/cs*(np.log(r + dbar) - sn*np.log(r + y2))
        return -0.5*(dbar*q/(r*(r + y2)) + q*sn/(r + y2) + a4*sn)/pi

    def _dip_slip(self, y1, y2, q):
        # TODO: doc; copied from `geoclaw/src/python/geoclaw/dtopotools.py`
        sn = self.sin_dip
        cs = self.cos_dip

        dbar = y2*sn - q*cs
        r = np.sqrt(y1**2 + y2**2 + q**2)
        xx = np.sqrt(y1**2 + q**2)
        a5 = 4.0*self.poisson_ratio*np.arctan((y2*(xx + q*cs) + xx*(r + xx)*sn)/(y1*(r + xx)*cs))/cs
        return -0.5*(dbar*q/(r*(r + y1)) + sn*np.arctan(y1*y2/(q*r)) - a5*sn*cs)/pi

    def set_initial_condition(self, prob):
        # TODO: doc; copied from `geoclaw/src/python/geoclaw/dtopotools.py`
        from scipy.interpolate import interp2d

        # Get fault parameters
        length = self.fault_length
        width = self.fault_width
        w = width/self.ny
        depth = self.focal_depth
        halfL = 0.5*length/self.nx

        # Get fault dislocation grid  # TODO: Currently assumed constant
        x = np.linspace(self.centre_x - 0.5*length, self.centre_x + 0.5*length, self.nx)
        y = np.linspace(self.centre_y - 0.5*width, self.centre_y + 0.5*width, self.ny)
        # slip = np.array([[self.get_fault_length(xi, yj) for yj in y] for xi in x])
        slip = self.slip
        X, Y = np.meshgrid(x, y)

        # Convert to distance along strike (x1) and dip (x2)  # TODO: use rotate function
        x1 = X*self.sin_strike + Y*self.cos_strike
        x2 = X*self.cos_strike - Y*self.sin_strike
        x2 = -x2  # Account for Okada's notation (x2 is distance up fault plane, rather than down dip)
        p = x2*self.cos_dip + self.focal_depth*self.sin_dip
        q = x2*self.sin_dip - self.focal_depth*self.cos_dip

        # Sum components for strike
        f1 = self._strike_slip(x1 + halfL, p, q)
        f2 = self._strike_slip(x1 + halfL, p - w, q)
        f3 = self._strike_slip(x1 - halfL, p, q)
        f4 = self._strike_slip(x1 - halfL, p - w, q)
        us = (f1 - f2 - f3 + f4)*slip*self.cos_rake

        # Sum components for dip
        g1 = self._dip_slip(x1 + halfL, p, q)
        g2 = self._dip_slip(x1 + halfL, p - w, q)
        g3 = self._dip_slip(x1 - halfL, p, q)
        g4 = self._dip_slip(x1 - halfL, p - w, q)
        ud = (g1 - g2 - g3 + g4)*slip*self.sin_rake

        # Interpolate  # TODO: Would be better to just evaluate all the above in UFL
        surf_interp = interp2d(X, Y, us + ud)
        initial_surface = Function(prob.P1[0])
        self.print_debug("Interpolating initial surface...")
        for i, xy in enumerate(prob.meshes[0].coordinates.dat.data):
            initial_surface.dat.data[i] = surf_interp(xy[0], xy[1])
        self.print_debug("Done!")

        # Set initial condition
        u, eta = prob.fwd_solutions[0].split()
        eta.interpolate(initial_surface)

        # TODO: TEMP
        self.X = X
        self.Y = Y
        self.us = us
        self.ud = ud

    def get_regularisation_term(self, prob):
        raise NotImplementedError  # TODO
