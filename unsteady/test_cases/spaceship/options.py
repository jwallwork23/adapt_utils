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

    # Domain specification
    mesh_file = os.path.join(os.path.dirname(__file__), 'spaceship.msh')
    narrows_width = PositiveFloat(1000.0).tag(config=False)
    maximum_upstream_width = PositiveFloat(5000.0).tag(config=False)

    def __init__(self, **kwargs):
        super(SpaceshipOptions, self).__init__(**kwargs)

        # Domain and mesh
        if os.path.exists(self.mesh_file):
            self.default_mesh = Mesh(self.mesh_file)
        else:
            raise IOError("Need to make mesh before initialising SpaceshipOptions object.")

        # Physics
        self.base_viscosity = 1.0
        self.friction_coeff = 0.0025

        # TODO: Timestepping
        self.dt = 3.0
        self.T_ramp = 1.0*self.T_tide
        self.end_time = self.T_ramp + 2.0*self.T_tide
        self.dt_per_export = 10

        # Tidal farm
        D = self.turbine_diameter
        self.region_of_interest = [(6050, 0, D, D), (6450, 0, D, D)]

        # Boundary forcing
        self.max_amplitude = 3.0
        self.omega = 2*pi/self.T_tide
        self.elev_in = [None for i in range(self.num_meshes)]

        # Solver parameters and discretisation
        self.stabilisation = None
        # self.stabilisation = 'lax_friedrichs'
        self.grad_div_viscosity = False
        self.grad_depth_viscosity = True
        self.family = 'dg-cg'

    def set_bathymetry(self, fs):
        x, y = SpatialCoordinate(fs.mesh())
        bathymetry = Function(fs)
        x1, x2 = 20000, 31500
        y1, y2 = 25.5, 4.5
        bathymetry.interpolate(min_value(((x - x1)*(y2 - y1)/(x2 - x1) + y1), y1))
        return bathymetry

    def set_boundary_conditions(self, prob, i):
        self.elev_in[i] = Function(prob.V[i].sub(1))
        inflow_tag = 2
        boundary_conditions = {
            'shallow_water': {
                inflow_tag: {'elev': self.elev_in[i]},
            }
        }
        return boundary_conditions

    def get_update_forcings(self, prob, i):
        tc = Constant(0.0)
        hmax = Constant(self.max_amplitude)

        def update_forcings(t):
            tc.assign(t)
            self.elev_in[i].assign(hmax*cos(self.omega*(tc - self.T_ramp)))
            self.print_debug("DEBUG: t = {:.0f}".format(t))

        return update_forcings

    def set_initial_condition(self, prob):
        u, eta = prob.fwd_solutions[0].split()
        x, y = SpatialCoordinate(prob.meshes[0])
        u.interpolate(as_vector([1e-8, 0.0]))
        eta.interpolate(-x/self.domain_length)
