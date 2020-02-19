from thetis import *
from thetis.configuration import *
import os

from adapt_utils.swe.turbine.options import SteadyTurbineOptions


__all__ = ["Steady2TurbineOptions", "Steady2TurbineOffsetOptions"]


class Steady2TurbineOptions(SteadyTurbineOptions):
    """Parameters for the steady 2 turbine problem"""

    mesh_path = Unicode('xcoarse.msh').tag(config=True)

    def __init__(self, **kwargs):
        super(Steady2TurbineOptions, self).__init__(**kwargs)

        # Domain
        self.domain_length = 1000.0
        self.domain_width = 300.0
        abspath = os.path.realpath(__file__)
        self.mesh_path = abspath.replace('options.py', self.mesh_path)
        self.default_mesh = Mesh(self.mesh_path)

        # Physical
        # self.base_viscosity = 1.3e-3
        self.base_viscosity = 1.0

        # Model
        self.family = 'dg-cg'

        # Tidal farm
        D = self.turbine_diameter
        L = self.domain_length
        W = self.domain_width
        self.region_of_interest = [(L/2-8*D, W/2, D/2), (L/2+8*D, W/2, D/2)]
        self.thrust_coefficient_correction()

    def set_bathymetry(self, fs):
        self.bathymetry = Constant(40.0)
        return self.bathymetry

    def set_inflow(self, fs):
        self.inflow = Constant(as_vector([3., 0.]))
        return self.inflow

    def set_viscosity(self, fs):
        sponge = False
        if sponge:
            self.viscosity = Function(fs)
            x, y = SpatialCoordinate(fs.mesh())
            xmin = 0.0
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
        boundary_conditions = {}
        if not hasattr(self, 'inflow'):
            self.set_inflow(fs.sub()[0])
        boundary_conditions[left_tag] = {'uv': self.inflow}
        boundary_conditions[right_tag] = {'elev': Constant(0.)}
        boundary_conditions[wall_tag] = {'un': Constant(0.)}
        return boundary_conditions


class Steady2TurbineOffsetOptions(Steady2TurbineOptions):
    def __init__(self, spacing=1.0, **kwargs):
        """
        :kwarg spacing: number of turbine widths to offset in each direction.
        """
        self.mesh_path = 'xcoarse_offset.msh'
        super(Steady2TurbineOffsetOptions, self).__init__(**kwargs)
        D = self.turbine_diameter
        L = self.domain_length
        W = self.domain_width
        self.region_of_interest = [(L/2-8*D, W/2-spacing*D, D/2), (L/2+8*D, W/2+spacing*D, D/2)]
