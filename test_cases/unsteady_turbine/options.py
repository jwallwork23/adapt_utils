from thetis import *
from thetis.configuration import *
import os

from adapt_utils.turbine.options import UnsteadyTurbineOptions


__all__ = ["Unsteady15TurbineOptions"]


class Unsteady15TurbineOptions(UnsteadyTurbineOptions):
    """Parameters for the unsteady 15 turbine array test case"""

    # Turbine parameters
    turbine_diameter = PositiveFloat(20.).tag(config=True)
    thrust_coefficient = NonNegativeFloat(7.6).tag(config=True)
    mesh_path = Unicode('channel.msh').tag(config=True)

    def __init__(self, approach='fixed_mesh'):
        self.base_viscosity = 3.0
        super(Steady2TurbineOptions, self).__init__(approach)
        self.domain_length = 3000.0
        self.domain_width = 1000.0
        if os.path.exists(self.mesh_path):
            self.default_mesh = Mesh(self.mesh_path)
        self.bathymetry.assign(50.0)

        # Timestepping
        self.dt = 3.0
        self.T_tide = 1.24*3600.0
        self.T_ramp = 1.0*self.T_tide
        self.end_time = self.T_ramp + 2.0*self.T_tide
        self.dt_per_export = 10
        self.dt_per_remesh = 10  # FIXME: solver seems to go out of sync if this != dt_per_export

        # Tidal farm
        D = self.turbine_diameter
        L = self.domain_length
        W = self.domain_width
        self.turbine_tags = range(2, 17):
        self.region_of_interest = [(L/2-8*D, W/2, D/2), (L/2+8*D, W/2, D/2)]  # FIXME
        self.thrust_coefficient_correction()

        # Boundary forcing
        self.max_depth = 50.0
        self.omega = 2*pi/self.T_tide

        # Solver parameters and discretisation
        self.lax_friedrichs = False  # TODO: temp
        self.family = 'dg-cg'

    def set_viscosity(self, fs):
        sponge = False
        self.viscosity = Function(fs)
        if sponge:
            x, y = SpatialCoordinate(fs.mesh())
            xmin = 0.0
            xmax = 1000.0
            ramp = 0.5
            eps = 20.0
            self.viscosity.interpolate(self.base_viscosity + exp(ramp*(x-xmax+eps)))
        else:
            self.viscosity.assign(self.base_viscosity)

    def set_bcs(self, fs):
        self.set_boundary_surface(fs.sub()[1])
        inflow_tag = 4
        outflow_tag = 2
        if not hasattr(self, 'boundary_conditions'):
            self.boundary_conditions = {}
        self.boundary_conditions[inflow_tag] = {'elev': self.elev_in}
        self.boundary_conditions[outflow_tag] = {'elev': self.elev_out}
        return self.boundary_conditions

    def set_initial_condition(self, fs):
        self.initial_condition = Function(fs)
        u, eta = self.initial_condition.split()
        x, y = SpatialCoordinate(fs.mesh())
        u.assign(as_vector([1e-8, 0.0]))
        eta.interpolate(-1/3000*x)
