from firedrake import *

from adapt_utils.adapt.r import *
from adapt_utils.options import *


def monitor(mesh, alpha=10.0, beta=200.0, gamma=0.15):
    """Some analytically defined monitor function."""
    x, y = SpatialCoordinate(mesh)
    return 1.0 + alpha*pow(cosh(beta*((x-0.5)*(x-0.5) + (y-0.5)*(y-0.5) - gamma)), -2)

op = Options()
op.debug = True
mesh = UnitSquareMesh(20, 20)
mm = MeshMover(mesh, monitor, op=op)
mm.adapt()
