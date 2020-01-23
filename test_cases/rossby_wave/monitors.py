from firedrake import *


__all__ = ["equator_monitor"]


def equator_monitor(mesh, alpha=5.0, beta=6.0):
    """
    Monitor function consisting of a bump function around the equator.

    :kwarg alpha: controls the amplitude of the bump.
    :kwarg beta: controls the width of the bump.
    """
    x, y = SpatialCoordinate(mesh)
    return 1.0 + conditional(lt(y**2, beta**2), alpha*exp(1 - 1/(1 - y*y/beta**2)), 0)
