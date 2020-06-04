from thetis import *
from thetis.configuration import *

import os

from adapt_utils.swe.turbine.options import SteadyTurbineOptions


__all__ = ["Steady2TurbineOptions"]


class Steady2TurbineOptions(SteadyTurbineOptions):
    """Parameters for the steady 2 turbine problem"""

    def __init__(self, offset=0, separation=8, **kwargs):
        self.offset = offset
        self.separation = separation
        self.mesh_path = 'xcoarse_{:d}.msh'.format(self.offset)

        # Physical
        self.base_viscosity = 0.5
        self.inflow_velocity = [5.0, 0.0]  # a typical fast flow in Pentland Firth

        super(Steady2TurbineOptions, self).__init__(**kwargs)

        # Domain
        self.domain_length = 1200.0
        self.domain_width = 500.0
        outer_res = 40.0
        inner_res = 8.0
        self.resolution = {'xcoarse': {'outer': outer_res, 'inner': inner_res}}
        for level in ('coarse', 'medium', 'fine', 'xfine'):
            self.resolution[level] = {'outer': outer_res, 'inner': inner_res}
            outer_res /= 2
            inner_res /= 2
        self.mesh_path = os.path.join(os.path.dirname(__file__), self.mesh_path)
        if os.path.exists(self.mesh_path):
            self.set_default_mesh()

        # Model
        self.family = 'dg-cg'
        self.sipg_parameter = None

        # Tidal farm
        D = self.turbine_diameter
        L = self.domain_length
        W = self.domain_width
        S = self.separation
        yloc = [W/2, W/2]
        yloc[0] -= self.offset*D
        yloc[1] += self.offset*D
        self.region_of_interest = [(L/2-S*D, yloc[0], D/2), (L/2+S*D, yloc[1], D/2)]
        self.thrust_coefficient_correction()  # TODO: Don't do this in __init__

    def set_default_mesh(self):
        self.default_mesh = Mesh(self.mesh_path)

    def set_bathymetry(self, fs):
        self.bathymetry = Function(fs).assign(40.0)
        return self.bathymetry

    def set_inflow(self, fs):
        self.inflow = Constant(as_vector(self.inflow_velocity))
        return self.inflow

    def set_viscosity(self, fs):
        sponge = False
        if sponge:
            self.viscosity = Function(fs)
            x, y = SpatialCoordinate(fs.mesh())
            xmax = 1000.0
            ramp = 0.5
            eps = 20.0
            self.viscosity.interpolate(self.base_viscosity + exp(ramp*(x-xmax+eps)))
        else:
            self.viscosity = Constant(self.base_viscosity)
        return self.viscosity

    def set_boundary_conditions(self, fs):
        left_tag = 1
        right_tag = 2
        wall_tag = 3
        if not hasattr(self, 'inflow'):
            self.set_inflow(fs.sub()[0])
        boundary_conditions = {
            'shallow_water': {
                left_tag: {'uv': self.inflow},
                right_tag: {'elev': Constant(0.0)},
                wall_tag: {'un': Constant(0.0)},
            }
        }
        return boundary_conditions

    def set_initial_condition(self, fs):
        if not hasattr(self, 'inflow'):
            self.set_inflow()
        self.initial_value = Function(fs)
        uv, elev = self.initial_value.split()
        uv.interpolate(self.inflow)
        return self.initial_value
