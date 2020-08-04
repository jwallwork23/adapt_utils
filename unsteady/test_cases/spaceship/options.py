from thetis import *

import os

from adapt_utils.unsteady.swe.turbine.options import TurbineOptions


__all__ = ["SpaceshipOptions"]


class SpaceshipOptions(TurbineOptions):
    def __init__(self, **kwargs):
        super(SpaceshipOptions, self).__init__(**kwargs)

        # Mesh
        self.mesh_file = os.path.join(os.path.dirname(__file__), 'spaceship.msh')
        if os.path.exists(self.mesh_file):
            self.default_mesh = Mesh(self.mesh_file)
        else:
            raise IOError("Need to make mesh before initialising SpaceshipOptions object.")

    def set_bathymetry(self, fs):
        x, y = SpatialCoordinate(fs.mesh())
        bathymetry = Function(fs)
        x1, x2 = 20000, 31500
        y1, y2 = 25.5, 4.5
        bathymetry.interpolate(min_value(((x - x1)*(y2 - y1)/(x2 - x1) + y1), y1))
        # bathymetry.interpolate(conditional(x < x1, y1, (x - x1)*(y2 - y1)/(x2 - x1) + y1))
        return bathymetry

    def set_boundary_conditions(self, prob, i):
        # TODO
        return {}
