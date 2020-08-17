from thetis import *

import os
import netCDF4

from adapt_utils.unsteady.swe.tsunami.options import TsunamiOptions
from adapt_utils.unsteady.swe.tsunami.conversion import from_latlon


__all__ = ["TohokuOptions"]


class TohokuOptions(TsunamiOptions):
    """
    Setup for model of the Tohoku tsunami which struck the east coast of Japan in 2011, leading to
    the meltdown of Daiichi nuclear power plant, Fukushima.

    Data sources:

      * Bathymetry data extracted from both GEBCO (https://www.gebco.net/) and ETOPO1
        (https://www.ngdc.noaa.gov/mgg/global/).

      * Initial free surface elevation field generated by inversion on tide gauge data by
        [Saito et al.]. Obtained via personal communication with T. Saito.

      * Timeseries for gauges P02 and P06 obtained via personal communication with T. Saito.

      * Timeseries for gauges 801-807 obtained from the Japanese Port and Airport Research
          Institute (PARI) via https://nowphas.mlit.go.jp/pastdata#contents3. Gauge locations and
          water depths are shown in https://nowphas.mlit.go.jp/pastdatars/PDF/list/dai_2017p.pdf.
          See also https://nowphas.mlit.go.jp/eng/ for further information.

      * Timeseries for gauges KPG1 and KPG2 obtained from the Japanese Agency for Marine-Earth
          Science and Technology (JAMSTEC) via http://www.jamstec.go.jp/scdc/top_e.html.

      * Timeseries for gauges 21401, 21413, 21418 and 21419 obtained from the US National Oceanic
        and Atmospheric Administration (NOAA) via https://www.ndbc.noaa.gov.


    [Saito et al.] T. Saito, Y. Ito, D. Inazu, R. Hino, "Tsunami source of the 2011 Tohoku‐Oki
                   earthquake, Japan: Inversion analysis based on dispersive tsunami simulations",
                   Geophysical Research Letters (2011), 38(7).
    """
    def __init__(self, mesh=None, level=0, force_zone_number=54, **kwargs):
        """
        :kwarg mesh: optionally use a custom mesh.
        :kwarg level: mesh resolution level, to be used if no mesh is provided.
        :kwarg force_zone_number: allow to use a specific UTM zone even if some mesh coordinates lie
            outside of it.
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
        super(TohokuOptions, self).__init__(force_zone_number=force_zone_number, **kwargs)
        self.save_timeseries = kwargs.get('save_timeseries', False)
        self.synthetic = kwargs.get('synthetic', False)
        self.qoi_scaling = kwargs.get('qoi_scaling', 1.0)

        # Mesh
        self.print_debug("INIT: Loading mesh...")
        self.resource_dir = os.path.join(os.path.dirname(__file__), '..', 'resources')
        self.level = level
        self.meshfile = os.path.join(self.resource_dir, 'meshes', 'Tohoku{:d}'.format(self.level))
        postproc = kwargs.get('postproc', False)
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
        self.print_debug("INIT: Done!")

        # Physics
        self.friction = 'manning'
        self.friction_coeff = 0.025
        self.base_viscosity = kwargs.get('base_viscosity', 0.0)

        # Stabilisation
        self.use_automatic_sipg_parameter = not np.isclose(self.base_viscosity, 0.0)
        self.sipg_parameter = kwargs.get('sipg_paramter', None)

        # Timestepping
        # ============
        #  * We export once per minute.
        #  * There is a trade-off between having an unneccesarily small timestep and being able to
        #    represent the gauge timeseries profiles.
        self.timestepper = 'CrankNicolson'
        self.dt = kwargs.get('dt', 60.0*0.5**level)
        self.dt_per_export = int(60.0/self.dt)
        self.start_time = kwargs.get('start_time', 0.0)
        # self.start_time = kwargs.get('start_time', 15*60.0)
        # self.end_time = kwargs.get('end_time', 24*60.0)
        self.end_time = kwargs.get('end_time', 60*60.0)
        self.num_timesteps = int(self.end_time/self.dt + 1)
        self.times = [i*self.dt for i in range(self.num_timesteps)]

        # Compute CFL number
        if self.debug:
            self.print_debug("INIT: Computing CFL number...")
            P0 = FunctionSpace(self.default_mesh, "DG", 0)
            P1 = FunctionSpace(self.default_mesh, "CG", 1)
            b = self.set_bathymetry(P1).vector().gather().max()
            g = self.g.values()[0]
            celerity = np.sqrt(g*b)
            dx = interpolate(CellDiameter(self.default_mesh), P0).vector().gather().max()
            cfl = celerity*self.dt/dx
            msg = "dx = {:.4e}  dt = {:.4e}  CFL number = {:.4e} {:1s} 1"
            print_output(msg.format(dx, self.dt, cfl, '<' if cfl < 1 else '>'))
            self.print_debug("INIT: Done!")

        # Gauge classifications
        self.near_field_pressure_gauges = {
            "gauges": ( "P02", "P06"),
            "arrival_time": 0.0,
            "weight": Constant(1.0),
        }
        self.mid_field_pressure_gauges = {
            "gauges": ("KPG1", "KPG2", "21418"),
            # "gauges": ("KPG1", "KPG2", "MPG1", "MPG2", "21418"),
            "arrival_time": 10*60.0,
            "weight": Constant(1.0),
        }
        self.far_field_pressure_gauges = {
            "gauges": ("21401", "21413", "21419"),
            "arrival_time": 50*60.0,
            "weight": Constant(1.0),
        }
        self.near_field_gps_gauges = {
            "gauges": ("801", "802", "803", "804", "806", "807"),
            "arrival_time": 5*60.0,
            "weight": Constant(1.0),
        }
        self.far_field_gps_gauges = {
            "gauges": ("811", "812", "813", "815"),
            "arrival_time": 10*60.0,
            "weight": Constant(1.0),
        }
        self.gauge_classifications_to_consider = (
            "near_field_pressure",
            "mid_field_pressure",
            "far_field_pressure",
            "near_field_gps",
            # "far_field_gps",
        )

        # Get gauges and locations of interest
        self.get_gauges()
        self.get_locations_of_interest(**kwargs)

    def read_bathymetry_file(self, source='etopo1'):
        self.print_debug("INIT: Reading bathymetry file...")
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
        self.print_debug("INIT: Done!")
        return lon, lat, elev

    def read_surface_file(self, zeroed=True):
        self.print_debug("INIT: Reading initial surface file...")
        fname = 'surf'
        if zeroed:
            fname = '_'.join([fname, 'zeroed'])
        nc = netCDF4.Dataset(os.path.join(self.resource_dir, 'surf', fname + '.nc'), 'r')
        lon = nc.variables['lon' if zeroed else 'x'][:]
        lat = nc.variables['lat' if zeroed else 'y'][:]
        elev = nc.variables['z'][:, :]
        nc.close()
        self.print_debug("INIT: Done!")
        return lon, lat, elev

    def get_gauges(self):
        """
        Collect gauge data, categorise and check coordinates lie within the domain.

        The gauge categories are as follows:
          * Near field GPS gauges: 801, 802, 803, 804, 806, 807;
          * Far field GPS gauges: 811, 812, 813, 815;
          * Near field pressure gauges: P02, P06;
          * Far field pressure gauges: KPG1, KPG2, 21401, 21413, 21418, 21419.

        These categorisations allow us to consider different:
          (a) start times for the period where we aim to fit the data;
          (b) weightings for the importance of the timeseries.

        Conservative start times of 5 minutes, 10 minutes, 0 minutes and 10 minutes are set by default.
        All gauges are equally weighted by unity by default.
        """
        self.gauges = {
            "801": {
                "lonlat": (141.6856, 38.2325),
                "operator": "PARI",
                "location": "Miyagi-Chubu-Oki (Central Miyagi)",
                "depth": 144.0,
            },
            "802": {
                "lonlat": (142.0969, 39.2586),
                "operator": "PARI",
                "location": "Iwate-Nanbu-Oki (South Iwate)",
                "depth": 204.0,
            },
            "803": {
                "lonlat": (141.8944, 38.8578),
                "operator": "PARI",
                "location": "Miyagi-Hokubu-Oki (North Miyagi)",
                "depth": 160.0,
            },
            "804": {
                "lonlat": (142.1867, 39.6272),
                "operator": "PARI",
                "location": "Iwate-Chubu-Oki (Central Iwate)",
                "depth": 200.0,
            },
            "806": {
                "lonlat": (141.1856, 36.9714),
                "operator": "PARI",
                "location": "Fukushimaken-Oki (Fukushima)",
                "depth": 137.0,
            },
            "807": {
                "lonlat": (142.0667, 40.1167),
                "operator": "PARI",
                "location": "Iwate-Hokubu-Oki (North Iwate)",
                "depth": 125.0,
            },
            "811": {
                "lonlat": (136.2594, 33.9022),
                "operator": "PARI",
                "location": "Mie-Owase-Oki (Owase, Mie)",
                "depth": 210.0,
            },
            "812": {
                "lonlat": (138.2750, 34.4033),
                "operator": "PARI",
                "location": "Shizuoka-Omaezaki-Oki (Omaezaki, Shizuoka)",
                "depth": 120.0,
            },
            "813": {
                "lonlat": (135.1567, 33.6422),
                "operator": "PARI",
                "location": "Wakayama-Nansei-Oki (Southwest Wakayama)",
                "depth": 201.0,
            },
            "815": {
                "lonlat": (134.4967, 33.4606),
                "operator": "PARI",
                "location": "Tokushima-Kaiyo-Oki (Kaiyo, Tokushima)",
                "depth": 430.0,
            },

            "P02": {"lonlat": (142.5016, 38.5002), "operator": "Tohoku University"},  # TODO: depth
            "P06": {"lonlat": (142.5838, 38.6340), "operator": "Tohoku University"},  # TODO: depth

            "KPG1": {"lonlat": (144.4375, 41.7040), "depth": 2218.0, "operator": "JAMSTEC"},
            "KPG2": {"lonlat": (144.8485, 42.2365), "depth": 2210.0, "operator": "JAMSTEC"},

            "MPG1": {"lonlat": (134.4753, 32.3907), "depth": 2308.0, "operator": "JAMSTEC"},
            "MPG2": {"lonlat": (134.3712, 32.6431), "depth": 1507.0, "operator": "JAMSTEC"},

            "21401": {"lonlat": (152.583, 42.617), "operator": "NOAA"}, # TODO: depth not on webpage
            "21413": {"lonlat": (152.132, 30.533), "depth": 5880.0, "operator": "NOAA"},
            "21418": {"lonlat": (148.655, 38.735), "depth": 5777.0, "operator": "NOAA"},
            "21419": {"lonlat": (155.717, 44.435), "depth": 5282.0, "operator": "NOAA"},
        }

        # Record the class containing each gauge and copy over parameters
        self.pressure_gauges = ()
        self.gps_gauges = ()
        gauge_classifications_to_consider = list(self.gauge_classifications_to_consider)
        for gauge_class in self.gauge_classifications_to_consider
            gauge_class_obj = self.__getattribute__("_".join([gauge_class, "gauges"]))
            arrival_time = gauge_class_obj["arrival_time"]
            if arrival_time >= self.end_time:
                gauge_classifications_to_consider.remove(gauge_class)
                msg = "WARNING: Removing gauge class {:s} due to late arrival time."
                self.print_debug(msg.format(gauge_class))
                continue
            gauges = gauge_class_obj["gauges"]
            for gauge in gauges:
                self.gauges[gauge]["class"] = gauge_class

                # Arrival time of tsunami weight
                self.gauges[gauge]["arrival_time"] = arrival_time
                self.gauges[gauge]["times"] = [t for t in self.times if t >= arrival_time]

                # Optional weighting of gauge classes
                self.gauges[gauge]["weight"] = gauge_class_obj["weight"]

            # Note gauges to consider
            if "pressure" in gauge_class:
                self.pressure_gauges += gauges
            elif "gps" in gauge_class:
                self.gps_gauges += gauges
        self.gauge_classifications_to_consider = tuple(self.gauge_classifications_to_consider)
        gauges_to_consider = self.pressure_gauges + self.gps_gauges
        for gauge in list(self.gauges.keys()):

            # Remove unused gauges (e.g. we don't use MPG1 or MPG2)
            if gauge not in gauges_to_consider:
                self.gauges.pop(gauge)
                continue

        # Convert coordinates to UTM and create timeseries array
        for gauge in gauges_to_consider:
            self.gauges[gauge]["data"] = []
            self.gauges[gauge]["timeseries"] = []
            lon, lat = self.gauges[gauge]["lonlat"]
            self.gauges[gauge]["utm"] = from_latlon(lat, lon, force_zone_number=54)
            self.gauges[gauge]["coords"] = self.gauges[gauge]["utm"]

        # Check validity of gauge coordinates
        self.print_debug("INIT: Checking validity of gauge coordinates...")
        for gauge in gauges_to_consider:
            try:
                self.default_mesh.coordinates.at(self.gauges[gauge]['coords'])
            except PointNotInDomainError:
                self.print_debug("NOTE: Gauge {:5s} is not in the domain; removing it".format(gauge))
                self.gauges.pop(gauge)
        self.print_debug("INIT: Done!")

    def get_locations_of_interest(self, **kwargs):
        """
        Read in locations of interest, determine their coordinates and check these coordinate lie
        within the domain.

        The possible coastal locations of interest include major cities and nuclear power plants:

        * Cities:
          - Onagawa;
          - Tokai;
          - Hamaoka;
          - Tohoku;
          - Tokyo.

        * Nuclear power plants:
          - Fukushima Daiichi;
          - Fukushima Daini.
        """
        locations = kwargs.get('locations', ["Fukushima Daiichi", ])
        radii = kwargs.get('radii', [50.0e+03, ])
        locations_of_interest = {
            "Onagawa": {"lonlat": (141.5008, 38.3995)},
            "Tokai": {"lonlat": (140.6067, 36.4664)},
            "Hamaoka": {"lonlat": (138.1433, 34.6229)},
            "Tohoku": {"lonlat": (141.3903, 41.1800)},
            "Tokyo": {"lonlat": (139.6917, 35.6895)},
            "Fukushima Daiichi": {"lonlat": (141.0281, 37.4213)},
            "Fukushima Daini": {"lonlat": (141.0249, 37.3166)},
        }
        self.locations_of_interest = {loc: locations_of_interest[loc] for loc in locations}
        radii = {locations[i]: r for i, r in enumerate(radii)}

        # Convert coordinates to UTM and create timeseries array
        for loc in self.locations_of_interest:
            self.locations_of_interest[loc]["data"] = []
            self.locations_of_interest[loc]["timeseries"] = []
            lon, lat = self.locations_of_interest[loc]["lonlat"]
            self.locations_of_interest[loc]["utm"] = from_latlon(lat, lon, force_zone_number=54)
            self.locations_of_interest[loc]["coords"] = self.locations_of_interest[loc]["utm"]

        # Check validity of gauge coordinates
        for loc in list(self.locations_of_interest.keys()):
            try:
                self.default_mesh.coordinates.at(self.locations_of_interest[loc]['coords'])
            except PointNotInDomainError:
                self.print_debug("NOTE: Location {:s} is not in the domain; removing it".format(loc))
                self.locations_of_interest.pop(loc)

        # Regions of interest
        loi = self.locations_of_interest
        self.region_of_interest = [loi[loc]["coords"] + (radii[loc], ) for loc in loi]

    def _get_update_forcings_forward(self, prob, i):
        from adapt_utils.misc import ellipse

        if np.isclose(self.regularisation, 0.0):
            self.J = 0
        else:
            self.J = self.get_regularisation_term(prob)
        scaling = Constant(0.5*self.qoi_scaling)

        # These will be updated by the checkpointing routine
        u, eta = prob.fwd_solutions[i].split()

        # Account for timeseries shift
        # ============================
        #   This can be troublesome business. With synthetic data, we can actually get away with not
        #   shifting, provided we are solving the linearised equations. However, in the nonlinear
        #   case and when using real data, we should make sure that the timeseries are comparable
        #   by always shifting them by the initial elevation at each gauge. This isn't really a
        #   problem for the continuous adjoint method. However, for discrete adjoint we need to
        #   annotate the initial gauge evaluation. Until point evaluation is annotated in Firedrake,
        #   the best thing is to just use the initial surface *field*. This does modify the QoI, but
        #   it shouldn't be too much of a problem if the mesh is sufficiently fine (and hence the
        #   indicator regions are sufficiently small.
        if self.synthetic:
            self.eta_init = Constant(0.0)
        else:
            # TODO: Use point evaluation once it is annotated
            self.eta_init = Function(eta.function_space()).assign(eta)

        mesh = eta.function_space().mesh()
        radius = 20.0e+03*pow(0.5, self.level)  # The finer the mesh, the smaller the region
        for gauge in self.gauges:
            gauge_dat = self.gauges[gauge]
            gauge_dat["obs"] = Constant(0.0)     # Constant associated with free surface observations

            # Setup interpolator
            self.sample_timeseries(gauge, sample=1 if gauge[0] == '8' else 60)

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
            quadrature_weight = Constant(0.5*dt if t < 0.5*dt or t >= self.end_time - 0.5*dt else dt)
            for gauge in self.gauges:
                gauge_dat = self.gauges[gauge]
                I = gauge_dat["indicator"]

                # Weightings
                if t < gauge_dat["arrival_time"]:  # We don't want to fit before the tsunami arrives
                    continue

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
                idx = len(gauge_dat["data"]) - self.num_timesteps + prob.iteration
                obs = gauge_dat["data"][idx] if self.synthetic else float(interpolator(t))
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
                self.J += assemble(scaling*quadrature_weight*gauge_dat["weight"]*diff*dx)
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
        weights = {}
        for gauge in self.gauges:
            I = scaling*self.gauges[gauge]["indicator"]
            weight = scaling*self.gauges[gauge]["weight"]
            eta_obs = self.gauges[gauge]["obs"]
            expr += I*weight*(eta_saved - self.eta_init - eta_obs)
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
                if t < gauge_data["arrival_time"]:  # TODO: Do we need to add/subtract a timestep?
                    weights[gauge] = gauge_dat["weight"].dat.data[0]
                    gauge_dat["weight"].assign(0.0)
                obs = gauge_dat["data"][prob.iteration-1] if self.synthetic else float(interpolator(t))
                gauge_dat["obs"].assign(obs)

            # Interpolate expression onto RHS
            k_eta.interpolate(expr)

            # Plot kernel
            if self.debug and prob.iteration % self.dt_per_export == 0:
                prob.kernel_file.write(k_eta)

            # Reset weights
            for gauge in list(weights.keys()):
                self.gauge[gauge]["weight"].assign(weights[gauge])
                weights.pop(gauge)

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
        if coords not in ("lonlat", "utm"):
            raise ValueError("Coordinate system {:s} not recognised.".format(coords))
        dat = self.gauges if gauges else self.locations_of_interest
        offset = 40.0e+03
        for loc in dat:
            x, y = np.copy(dat[loc][coords])
            kwargs = {
                "xy": dat[loc][coords],
                "color": "indigo",
                "ha": "right",
                "va": "center",
                "fontsize": fontsize,
            }
            if not gauges:
                if loc == "Fukushima Daini":
                    continue
                elif loc == "Fukushima Daiichi":
                    loc = "Fukushima"
                x += offset
            elif loc[0] == "8":
                kwargs["color"] = "C3"

                # Horizontal alignment
                if loc in ("811", "812"):
                    kwargs["ha"] = "left"
                elif loc == "813":
                    kwargs["ha"] = "center"
                if loc in ("802", "803", "804", "806", "811", "812", "815"):
                    x -= offset
                elif loc == "813":
                    x -= 0.5**offset

                # Vertical alignment
                if loc in ("801", "803", "806", "811", "812", "813", "815"):
                    kwargs["va"] = "bottom"
                elif loc in ("804", "807"):
                    kwargs["va"] = "top"
                if loc in ("804", "807"):
                    y += 2*offset
                elif loc in ("801", "803", "806"):
                    y -= 2*offset
            else:
                kwargs["color"] = "navy"
                x += offset
                kwargs["ha"] = "left"
                if loc == "P02":
                    y -= 2*offset
                    kwargs["va"] = "bottom"
                elif loc == "P06":
                    y += 2*offset
                    kwargs["va"] = "top"
                elif loc == "MPG1":
                    y -= offset
            kwargs["xytext"] = (x, y)
            axes.plot(*dat[loc][coords], 'x', color=kwargs["color"])
            axes.annotate(loc, **kwargs)

    def detide(self, gauge):
        """
        Remove tidal constituents from the observed timeseries at a given gauge. This is done using
        the Python re-implementation of the Matlab package UTide, available at
        https://github.com/wesleybowman/UTide.

        :arg gauge: string denoting the gauge of interest.
        :returns: a 3-tuple of arrays corresponding to the observation times, the de-tided timeseries
            and the original timeseries, respectively.
        """
        from pandas import date_range
        import utide
        from matplotlib.dates import date2num

        # Read data from file
        time, elev = self.extract_data(gauge)

        # Start date and time of observations (in the GMT timezone)
        start = '2011-03-11 05:46:00'

        # Observation frequency
        if gauge[0] == "8":
            freq = "5S"
        elif gauge[0] == "P" or "PG" in gauge:
            freq = "S"
        elif gauge[0] == "2":
            freq = "60S"
        else:
            raise ValueError("Gauge {:s} not recognised.".format(gauge))
        time_str = date2num(date_range(start=start, periods=len(time), freq=freq).to_pydatetime())

        # Interpolate away any NaNs
        if np.any(np.isnan(elev)):
            self.sample_timeseries(gauge)
            elev = np.array([self.gauges[gauge]["interpolator"](t) for t in time])

        # Shift to zero
        elev[:] -= elev[0]

        # Get anomaly
        anomaly = elev - elev.mean()
        assert not np.any(np.isnan(anomaly))

        # Apply de-tiding algorithm to anomaly
        kwargs = {
            'method': 'ols',     # ordinary least squares
            'conf_int': 'none',  # linearised confidence intervals
            'lat': np.array([self.gauges[gauge]["lonlat"][1], ]),
        }
        sol = utide.solve(time_str, anomaly, **kwargs)
        tide = utide.reconstruct(time_str, sol)

        # Subtract de-tided component
        detided = anomaly - np.array(tide.h).reshape(anomaly.shape)
        diff = detided - elev

        return time, detided, elev
