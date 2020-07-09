r"""
Sensor functions defined in

Olivier, Géraldine. Anisotropic metric-based mesh adaptation for unsteady CFD simulations involving moving geometries. Diss. 2011.
"""
from firedrake import *


__all__ = ["bowl", "hyperbolic", "multiscale", "interweaved"]


def bowl(mesh):
    x, y = SpatialCoordinate(mesh)
    return x*x + y*y


def hyperbolic(mesh):
    x, y = SpatialCoordinate(mesh)
    return conditional(ge(abs(x*y), 2*pi/50), 0.01*sin(50*x*y), sin(50*x*y))


def multiscale(mesh):
    x, y = SpatialCoordinate(mesh)
    return 0.1*sin(50*x) + atan(0.1/(sin(5*y) - 2*x))


def interweaved(mesh):
    x, y = SpatialCoordinate(mesh)
    return atan(0.1/(sin(5*y) - 2*x)) + atan(0.5/(sin(3*y) - 7*x))
