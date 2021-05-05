r"""
Sensor functions defined in

Olivier, Géraldine. Anisotropic metric-based mesh adaptation for unsteady CFD simulations involving moving geometries. Diss. 2011.
"""
from firedrake import *


__all__ = ["bowl", "hyperbolic", "multiscale", "interweaved"]


def bowl(mesh, xy=None):
    x, y = xy or SpatialCoordinate(mesh)
    return 0.5*(x*x + y*y)


def hyperbolic(mesh, xy=None):
    x, y = xy or SpatialCoordinate(mesh)
    return conditional(ge(abs(x*y), 2*pi/50), 0.01*sin(50*x*y), sin(50*x*y))


def multiscale(mesh, xy=None):
    x, y = xy or SpatialCoordinate(mesh)
    return 0.1*sin(50*x) + atan(0.1/(sin(5*y) - 2*x))


def interweaved(mesh, xy=None):
    x, y = xy or SpatialCoordinate(mesh)
    return atan(0.1/(sin(5*y) - 2*x)) + atan(0.5/(sin(3*y) - 7*x))
