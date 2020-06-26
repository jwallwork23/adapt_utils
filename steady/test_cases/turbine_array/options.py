from thetis import *
from thetis.configuration import *

import os

from adapt_utils.steady.swe.turbine.options import TurbineOptions


__all__ = ["TurbineArrayOptions"]


class TurbineArrayOptions(TurbineOptions):
    """Parameters for the steady 2 turbine problem"""

    def __init__(self, offset=0, separation=8, **kwargs):
        self.offset = offset
        self.separation = separation
        self.mesh_path = 'xcoarse_{:d}.msh'.format(self.offset)

        # Physics
        self.inflow_velocity = [5.0, 0.0]  # Typical fast flow in Pentland Firth
        self.base_bathymetry = 40.0        # Typical depth in Pentland Firth
        self.base_viscosity = 0.5          # Chosen to give a moderately advection-dominated problem

        super(TurbineArrayOptions, self).__init__(**kwargs)

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
        self.use_automatic_sipg_parameter = True

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

    def set_inflow(self, fs):
        return interpolate(as_vector(self.inflow_velocity), fs)

    def set_boundary_conditions(self, prob, i):
        left_tag = 1
        right_tag = 2
        wall_tag = 3
        boundary_conditions = {
            'shallow_water': {
                left_tag: {'uv': Constant(as_vector(self.inflow_velocity))},
                right_tag: {'elev': Constant(0.0)},
                wall_tag: {'un': Constant(0.0)},
            }
        }
        return boundary_conditions

    def set_initial_condition(self, prob):
        u, eta = prob.fwd_solutions[0].split()
        u.interpolate(as_vector([self.inflow_velocity]))
