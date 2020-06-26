from firedrake import *


__all__ = ["equator_monitor", "soliton_monitor"]


def equator_monitor(mesh, alpha=5.0, beta=6.0):
    """
    Monitor function consisting of a bump function around the equator.

    :kwarg alpha: controls the amplitude of the bump.
    :kwarg beta: controls the width of the bump.
    """
    x, y = SpatialCoordinate(mesh)
    return 1.0 + conditional(lt(y**2, beta**2), alpha*exp(1 - 1/(1 - (y/beta)**2)), 0)


def soliton_monitor(mesh, alpha=5.0, beta=6.0, gamma=12.0):
    """
    Monitor function consisting of a bump function around the soliton.

    :kwarg alpha: controls the amplitude of the bump.
    :kwarg beta: controls the width of the bump in the y-direction.
    :kwarg gamma: controls the width of the bump in the x-direction.
    """
    x, y = SpatialCoordinate(mesh)
    return 1.0 + conditional(lt(x**2 + y**2, beta**2),
                             alpha*exp(1 - 1/(1 - (x**2 + y**2)/beta**2)), 0)
