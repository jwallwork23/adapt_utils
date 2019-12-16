from thetis import *
from thetis.configuration import *

from adapt_utils.turbine.options import SteadyTurbineOptions


__all__ = ["Steady2TurbineOptions", "Steady2TurbineOffsetOptions"]


class Steady2TurbineOptions(SteadyTurbineOptions):
    """Parameters for the steady 2 turbine problem"""

    # Turbine parameters
    turbine_diameter = PositiveFloat(18.).tag(config=True)
    thrust_coefficient = NonNegativeFloat(0.8).tag(config=True)

    def __init__(self, approach='fixed_mesh'):
        # self.base_viscosity = 1.3e-3
        self.base_viscosity = 1.0
        super(Steady2TurbineOptions, self).__init__(approach)
        self.domain_length = 1000.0
        self.domain_width = 300.0
        self.default_mesh = Mesh('xcoarse_2_turbine.msh')

        # Tidal farm
        D = self.turbine_diameter
        L = self.domain_length
        W = self.domain_width
        self.region_of_interest = [(L/2-8*D, W/2, D/2), (L/2+8*D, W/2, D/2)]
        self.thrust_coefficient_correction()

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
        left_tag = 1
        right_tag = 2
        top_bottom_tag = 3
        if not hasattr(self, 'boundary_conditions'):
            self.boundary_conditions = {}
        if not hasattr(self, 'inflow'):
            self.set_inflow(fs.sub()[0])
        self.boundary_conditions[left_tag] = {'uv': self.inflow}
        self.boundary_conditions[right_tag] = {'elev': Constant(0.)}


class Steady2TurbineOffsetOptions(Steady2TurbineOptions):
    def __init__(self, approach='fixed_mesh', spacing=1.0):
        """
        :kwarg spacing: number of turbine widths to offset in each direction.
        """
        super(Steady2TurbineOffsetOptions, self).__init__(approach)
        D = self.turbine_diameter
        L = self.domain_length
        W = self.domain_width
        self.region_of_interest = [(L/2-8*D, W/2-spacing*D, D/2), (L/2+8*D, W/2+spacing*D, D/2)]
        self.default_mesh = Mesh('xcoarse_2_offset_turbine.msh')
