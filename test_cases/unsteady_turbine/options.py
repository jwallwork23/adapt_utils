from thetis import *
from thetis.configuration import *

from adapt_utils.turbine.options import UnsteadyTurbineOptions


__all__ = ["Unsteady15TurbineOptions"]


# TODO: Bring up to date
class Unsteady15TurbineOptions(UnsteadyTurbineOptions):
    """Parameters for the unsteady 15 turbine problem"""

    # Turbine parameters
    turbine_diameter = PositiveFloat(20.).tag(config=True)
    thrust_coefficient = NonNegativeFloat(7.6).tag(config=True)
    base_viscosity = NonNegativeFloat(3., help="Fluid viscosity (assumed constant).").tag(config=True)
    depth = PositiveFloat(50., help="Water depth (assumes flat bathymetry).").tag(config=True)

    def __init__(self, approach='fixed_mesh'):
        super(Unsteady15TurbineOptions, self).__init__(approach)
        self.default_mesh = RectangleMesh(150, 50, 3000., 1000.)    # FIXME: wrong ids
        self.default_mesh.coordinates.dat.data[:] -= [1500., 500.]  # FIXME: not parallel
        self.h_max = 100

        # Tidal farm
        D = self.turbine_diameter
        delta_x = 10*D
        delta_y = 7.5*D
        for i in [-2, -1, 0, 1, 2]:
            for j in [-1, 0, 1]:
                self.region_of_interest.append((i*delta_x, j*delta_y, D/2))
        self.thrust_coefficient_correction()

    def set_initial_condition(self, fs):
        x, y = SpatialCoordinate(fs.mesh())
        q_init = Function(fs)
        self.uv_init, self.elev_init = q_init.split()
        self.uv_init.interpolate(as_vector([1e-8, 0.]))
        self.elev_init.interpolate(-1/3000*x)  # linear from -1 to 1
        return q_init

    def set_bcs(self):  # TODO: standardise with other Options classes
        bottom_tag = 1
        right_tag = 2
        top_tag = 3
        left_tag = 4
        freeslip_bc = {'un': Constant(0.)}
        self.boundary_conditions = {
          left_tag: {'elev': self.elev_in},
          right_tag: {'elev': self.elev_out},
          top_tag: freeslip_bc,
          bottom_tag: freeslip_bc,
        }

