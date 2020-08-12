from thetis import *
from thetis.configuration import *

import os

from adapt_utils.unsteady.swe.turbine.options import TurbineOptions


__all__ = ["SpaceshipOptions"]


class SpaceshipOptions(TurbineOptions):
    """Parameters for the 'spaceship' idealised tidal lagoon test case."""

    # Turbine parameters
    turbine_diameter = PositiveFloat(18.0).tag(config=False)
    num_turbines = PositiveInteger(2).tag(config=False)
    turbine_tags = [2, 3]
    thrust_coefficient = NonNegativeFloat(8.0).tag(config=True)

    # Domain specification
    mesh_file = os.path.join(os.path.dirname(__file__), 'spaceship.msh')
    narrows_width = PositiveFloat(1000.0).tag(config=False)
    maximum_upstream_width = PositiveFloat(5000.0).tag(config=False)
    domain_length = PositiveFloat(61500.0).tag(config=False)
    domain_width = PositiveFloat(60000.0).tag(config=False)

    # Resources
    resource_dir = os.path.join(os.path.dirname(__file__), 'resources')

    def __init__(self, **kwargs):
        super(SpaceshipOptions, self).__init__(**kwargs)

        # Domain and mesh
        if os.path.exists(self.mesh_file):
            self.default_mesh = Mesh(self.mesh_file)
        else:
            raise IOError("Need to make mesh before initialising SpaceshipOptions object.")

        # Physics
        self.base_viscosity = 5.0  # TODO: Sponge condition?
        self.friction_coeff = 0.0025
        self.max_depth = 25.5

        # Boundary forcing
        self.interpolate_tidal_forcing()
        self.elev_in = [None for i in range(self.num_meshes)]

        # Timestepping
        self.timestepper = 'CrankNicolson'
        # self.timestepper = 'PressureProjectionPicard'
        # self.implicitness_theta = 1.0
        self.dt = 10.0
        # self.end_time = self.tidal_forcing_end_time
        self.end_time = 24*3600.0
        self.dt_per_export = 30

        # Tidal farm
        D = self.turbine_diameter
        self.region_of_interest = [(6050, 0, D, D), (6450, 0, D, D)]

        # Solver parameters and discretisation
        self.stabilisation = 'lax_friedrichs'
        # self.stabilisation = None
        self.grad_div_viscosity = False
        # self.grad_depth_viscosity = True
        self.grad_depth_viscosity = False
        self.family = 'dg-cg'

    def set_bathymetry(self, fs):
        x, y = SpatialCoordinate(fs.mesh())
        bathymetry = Function(fs)
        x1, x2 = 20000, 31500
        y1, y2 = 25.5, 4.5
        bathymetry.interpolate(min_value(((x - x1)*(y2 - y1)/(x2 - x1) + y1), y1))
        return bathymetry

    def set_boundary_conditions(self, prob, i):
        # self.elev_in[i] = Function(prob.V[i].sub(1))
        self.elev_in[i] = Constant(0.0)
        inflow_tag = 2
        boundary_conditions = {
            'shallow_water': {
                inflow_tag: {'elev': self.elev_in[i]},
            }
        }
        return boundary_conditions

    def get_update_forcings(self, prob, i, **kwargs):
        interp = self.tidal_forcing_interpolator

        def update_forcings(t):
            self.elev_in[i].assign(self.tidal_forcing_interpolator(t))
            self.print_debug("DEBUG: t = {:.0f}".format(t))

        return update_forcings

    def set_initial_condition(self, prob):
        u, eta = prob.fwd_solutions[0].split()
        x, y = SpatialCoordinate(prob.meshes[0])

        # Small velocity to avoid zero initial condition
        u.interpolate(as_vector([1e-8, 0.0]))
