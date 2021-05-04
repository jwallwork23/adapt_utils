from thetis import *
from thetis.configuration import *

import numpy as np
import os

from adapt_utils.swe.turbine.options import SteadyTurbineOptions


__all__ = ["TurbineArrayOptions"]


class TurbineArrayOptions(SteadyTurbineOptions):
    """
    Parameters for the steady 2 turbine problem
    """

    turbine_diameter = PositiveFloat(18.0).tag(config=False)
    array_length = PositiveInteger(2).tag(config=False)
    array_width = PositiveInteger(1).tag(config=False)
    num_turbines = PositiveInteger(2).tag(config=False)

    # Domain specification
    mesh_dir = os.path.join(os.path.dirname(__file__), 'resources', 'meshes')
    domain_length = PositiveFloat(1200.0).tag(config=False)
    domain_width = PositiveFloat(500.0).tag(config=False)

    def __init__(self, level=0, offset=0, separation=8, meshgen=False, box=False, **kwargs):
        """
        :kwarg level: number of iso-P2 refinements to apply to the base mesh.
        :kwarg offset: offset of the turbines to the south and north in terms of turbine diameters.
        :kwarg separation: number of turbine diameters separating the two.
        :kwarg generate_geo: if True, the mesh is not built (used in meshgen.py).
        """
        super(TurbineArrayOptions, self).__init__(**kwargs)
        self.array_ids = np.array([2, 3])
        self.farm_ids = (2, 3)
        self.offset = offset
        self.separation = separation

        # Physics
        self.inflow_velocity = [5.0, 0.0]  # Typical fast flow in Pentland Firth
        self.base_velocity = self.inflow_velocity
        self.base_viscosity = Constant(0.5) # Chosen to give a moderately advection-dominated problem
        self.base_bathymetry = 40.0         # Typical depth in Pentland Firth
        self.friction_coeff = 0.0025

        # Tidal farm
        D = self.turbine_diameter
        L = self.domain_length
        W = self.domain_width
        S = self.separation
        yloc = [W/2, W/2]
        yloc[0] -= self.offset*D
        yloc[1] += self.offset*D
        self.region_of_interest = [(L/2-S*D, yloc[0], D/2), (L/2+S*D, yloc[1], D/2)]
        assert len(self.region_of_interest) == self.num_turbines

        # Gmsh specification
        self.base_outer_res = 40.0
        self.base_inner_res = 8.0
        if box:
            assert self.offset == 0
            self.mesh_file = 'channel_refined_{:d}.msh'.format(level)
        else:
            self.mesh_file = 'channel_{:d}_{:d}.msh'.format(level, self.offset)
        self.mesh_file = os.path.join(self.mesh_dir, self.mesh_file)
        if meshgen:
            return

        # Domain and mesh
        if os.path.isfile(self.mesh_file):
            self.default_mesh = Mesh(self.mesh_file)

        # Discretisation
        self.family = 'dg-cg'
        self.sipg_parameter = None
        self.use_automatic_sipg_parameter = True
        self.use_maximal_sipg = True
        self.stabilisation = 'lax_friedrichs'
        self.grad_div_viscosity = False
        self.grad_depth_viscosity = False

        # Mesh adaptation
        self.h_min = 1.0e-05
        self.h_max = 5.0e+02

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
        u, eta = prob.fwd_solution.split()
        u.interpolate(as_vector(self.base_velocity))
        eta.assign(0.0)
