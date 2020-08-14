from thetis import *
from thetis.configuration import *

from math import e
import numpy as np
import os

from adapt_utils.unsteady.swe.turbine.options import TurbineOptions


__all__ = ["SpaceshipOptions"]


class SpaceshipOptions(TurbineOptions):
    """Parameters for the 'spaceship' idealised tidal lagoon test case."""

    # Turbine parameters
    turbine_diameter = PositiveFloat(18.0).tag(config=False)
    num_turbines = PositiveInteger(2).tag(config=False)
    thrust_coefficient = NonNegativeFloat(0.8).tag(config=True)

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
        self.array_ids = np.array([3, 2])
        self.farm_ids = tuple(self.array_ids)

        # Domain and mesh
        if os.path.exists(self.mesh_file):
            self.default_mesh = Mesh(self.mesh_file)
        else:
            raise IOError("Need to make mesh before initialising SpaceshipOptions object.")

        # Physics
        self.base_viscosity = 5.0
        self.viscosity_sponge_type = 'linear'
        # self.viscosity_sponge_type = 'exponential'
        self.max_viscosity = 1000.0
        self.friction_coeff = 0.0025
        self.max_depth = 25.5

        # Boundary forcing
        self.interpolate_tidal_forcing()
        self.elev_in = [None for i in range(self.num_meshes)]

        # Timestepping
        self.timestepper = 'PressureProjectionPicard'
        self.implicitness_theta = 1.0
        self.use_semi_implicit_linearisation = True
        self.dt = 10.0
        # self.end_time = self.tidal_forcing_end_time
        self.T_ramp = 2.0*self.T_tide
        # self.end_time = self.T_ramp + 2.0*self.T_tide
        self.end_time = 3*24*3600.0
        self.dt_per_export = 30

        # Tidal farm
        D = self.turbine_diameter
        self.region_of_interest = [(6050, 0, D, D), (6450, 0, D, D)]

        # Solver parameters and discretisation
        self.stabilisation = 'lax_friedrichs'
        # self.stabilisation = None
        self.grad_div_viscosity = False
        self.grad_depth_viscosity = True
        # self.grad_depth_viscosity = False
        self.family = 'dg-cg'

    def set_bathymetry(self, fs):
        x, y = SpatialCoordinate(fs.mesh())
        bathymetry = Function(fs)
        x1, x2 = 20000, 31500
        y1, y2 = 25.5, 4.5
        bathymetry.interpolate(min_value(((x - x1)*(y2 - y1)/(x2 - x1) + y1), y1))
        return bathymetry

    def set_viscosity(self, fs):
        """
        We use a sponge condition on the forced boundary.

        The type of sponge condition is specified by :attr:`viscosity_sponge_type`, which may be
        None, or chosen from {'linear', 'exponential'}. The sponge ramps up the viscosity from
        :attr:`base_viscosity` to :attr:`max_viscosity`.
        """
        nu = Function(fs, name="Viscosity")
        x, y = SpatialCoordinate(fs.mesh())
        R = 30000.0  # Radius of semicircular part of domain
        r = sqrt(x**2 + y**2)/R
        base_viscosity = self.base_viscosity
        if self.viscosity_sponge_type is None:
            return Constant(base_viscosity)
        if self.viscosity_sponge_type == 'linear':
            sponge = base_viscosity + r*(self.max_viscosity - base_viscosity)
        elif self.viscosity_sponge_type == 'exponential':
            sponge = base_viscosity + (exp(r) - 1)/(e - 1)*(self.max_viscosity - base_viscosity)
        else:
            msg = "Viscosity sponge type {:s} not recognised."
            raise ValueError(msg.format(self.viscosity_sponge_type))
        nu.interpolate(max_value((x <= 0.0)*sponge, base_viscosity))
        return nu

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

        def update_forcings(t):
            tau = t - 0.5*self.dt
            forcing = float(self.tidal_forcing_interpolator(tau))
            self.elev_in[i].assign(forcing)
            self.print_debug("DEBUG: forcing at time {:.0f} is {:6.4}".format(tau, forcing))

        return update_forcings

    def set_initial_condition(self, prob):
        u, eta = prob.fwd_solutions[0].split()
        u.interpolate(as_vector([1.0e-08, 0.0]))  # Small velocity to avoid zero initial condition
