from thetis import *
from thetis.configuration import *
from firedrake.petsc import PETSc

import os
import netCDF4
import matplotlib.pyplot as plt

from adapt_utils.unsteady.swe.tsunami.options import TsunamiOptions
from adapt_utils.unsteady.swe.tsunami.conversion import from_latlon
from adapt_utils.case_studies.tohoku.resources.gauges.sample import sample_timeseries
from adapt_utils.misc import gaussian, ellipse


__all__ = ["TohokuOptions", "TohokuGaussianBasisOptions"]


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
    def __init__(self, mesh=None, level=0, postproc=True, save_timeseries=False, artificial=False, locations=["Fukushima Daiichi", ], radii=[50.0e+03, ], qoi_scaling=1.0e-10, **kwargs):
        self.force_zone_number = 54
        super(TohokuOptions, self).__init__(**kwargs)
        self.save_timeseries = save_timeseries
        self.artificial = artificial
        self.qoi_scaling = qoi_scaling

        # Stabilisation
        self.use_automatic_sipg_parameter = False
        self.sipg_parameter = None
        self.base_viscosity = 0.0
        # self.base_viscosity = 1.0e-03

        # Mesh
        self.print_debug("Loading mesh...")
        self.resource_dir = os.path.join(os.path.dirname(__file__), 'resources')
        self.level = level
        self.meshfile = os.path.join(self.resource_dir, 'meshes', 'Tohoku{:d}'.format(self.level))
        if mesh is None:
            if postproc:
                newplex = PETSc.DMPlex().create()
                newplex.createFromFile(self.meshfile + '.h5')
                self.default_mesh = Mesh(newplex)
            else:
                self.default_mesh = Mesh(self.meshfile + '.msh')
        else:
            self.default_mesh = mesh
        self.print_debug("Done!")

        # Fields
        self.friction = None
        # self.friction = 'manning'
        # self.friction_coeff = 0.025

        # Timestepping: export once per minute for 24 minutes
        self.timestepper = 'CrankNicolson'
        self.dt = 5.0
        self.dt_per_export = int(60.0/self.dt)
        self.start_time = 15*60.0
        self.end_time = 24*60.0
        # self.end_time = 60*60.0

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
        self.J = 0
        scaling = Constant(self.qoi_scaling)
        weight = Constant(1.0)
        eta_obs = Constant(0.0)
        eta = prob.fwd_solutions[i].split()[1]
        mesh = eta.function_space().mesh()
        self.times = []
        radius = 20.0e+03*pow(0.5, self.level)  # The finer the mesh, the smaller the region
        for gauge in self.gauges:
            if self.save_timeseries:
                if not self.artificial or "data" not in self.gauges[gauge]:
                    self.gauges[gauge]["data"] = []
                self.gauges[gauge]["timeseries"] = []
                self.gauges[gauge]["timeseries_smooth"] = []
                self.gauges[gauge]["diff"] = []
                self.gauges[gauge]["diff_smooth"] = []
                self.gauges[gauge]["init"] = eta.at(self.gauges[gauge]["coords"])
            sample = 1 if gauge[0] == '8' else 60
            self.gauges[gauge]["interpolator"] = sample_timeseries(gauge, sample=sample)
            loc = self.gauges[gauge]["coords"]
            self.gauges[gauge]["indicator"] = interpolate(ellipse([loc + (radius,), ], mesh), prob.P0[i])
            self.gauges[gauge]["area"] = assemble(self.gauges[gauge]["indicator"]*dx, annotate=False)

        def update_forcings(t):
            weight.assign(0.5 if t < 0.5*self.dt or t >= self.end_time - 0.5*self.dt else 1.0)
            dtc = Constant(self.dt)
            # iteration = len(self.times)
            iteration = prob.iteration
            for gauge in self.gauges:

                # Point evaluation at gauges
                if self.save_timeseries:
                    eta_discrete = eta.at(self.gauges[gauge]["coords"]) - self.gauges[gauge]["init"]
                    self.gauges[gauge]["timeseries"].append(eta_discrete)

                # Interpolate observations
                if self.gauges[gauge]["data"] != []:
                    if self.artificial:
                        obs = self.gauges[gauge]["data"][iteration]
                    else:
                        obs = float(self.gauges[gauge]["interpolator"](t))
                    eta_obs.assign(obs)

                    if self.save_timeseries:
                        if not self.artificial:
                            self.gauges[gauge]["data"].append(obs)

                        # Discrete form of error
                        diff = 0.5*(eta_discrete - eta_obs.dat.data[0])**2
                        self.gauges[gauge]["diff"].append(diff)

                    # Continuous form of error
                    I = self.gauges[gauge]["indicator"]
                    A = self.gauges[gauge]["area"]
                    diff = 0.5*I*(eta - eta_obs)**2
                    self.J += assemble(scaling*weight*dtc*diff*dx)
                    self.gauges[gauge]["diff_smooth"].append(assemble(diff*dx, annotate=False)/A)
                    self.gauges[gauge]["timeseries_smooth"].append(assemble(I*eta_obs*dx, annotate=False)/A)
            self.times.append(t)
            return

        return update_forcings

    def _get_update_forcings_adjoint(self, prob, i):
        eta_obs = Constant(0.0)
        scaling = Constant(self.qoi_scaling)

        def update_forcings(t):

            # Get gauge data (loaded from checkpoint)
            u_saved, eta_saved = prob.fwd_solutions[i].split()

            # Sum differences between checkpoint and data
            expr = 0
            for gauge in self.gauges:
                if self.artificial:
                    obs = self.gauges[gauge]["data"][prob.iteration-1]
                else:
                    obs = float(self.gauges[gauge]["interpolator"](t))
                eta_obs.assign(obs)
                expr += self.gauges[gauge]["indicator"]*(eta_saved - eta_obs)

            # Interpolate onto RHS
            k_u, k_eta = prob.kernels[i].split()
            k_eta.interpolate(scaling*expr)

            return

        return update_forcings

    def get_update_forcings(self, prob, i, adjoint=False):
        if adjoint:
            return self._get_update_forcings_adjoint(prob, i)
        else:
            return self._get_update_forcings_forward(prob, i)

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


class TohokuGaussianBasisOptions(TohokuOptions):
    """
    Initialise the free surface with initial condition consisting of a single Gaussian basis function
    scaled by a control parameter.

    This is useful for inversion experiments because the control parameter space is one dimensional,
    meaning it can be easily plotted.
    """
    def __init__(self, control_parameter=10.0, **kwargs):
        super(TohokuGaussianBasisOptions, self).__init__(**kwargs)
        R = FunctionSpace(self.default_mesh, "R", 0)
        self.control_parameter = Function(R, name="Control parameter")
        self.control_parameter.assign(control_parameter)

    def set_initial_condition(self, prob):
        self.basis_function = Function(prob.V[0])
        psi, phi = self.basis_function.split()
        loc = (0.7e+06, 4.2e+06)
        radii = (48e+03, 96e+03)
        angle = pi/12
        phi.interpolate(gaussian([loc + radii, ], prob.meshes[0], rotation=angle))
        prob.fwd_solutions[0].project(self.control_parameter*self.basis_function)
