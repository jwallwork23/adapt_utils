from firedrake import *

import pytest
import numpy as np

from adapt_utils.adapt.recovery import *


def get_mesh(dim, n):
    if dim == 2:
        return UnitSquareMesh(n, n)
    elif dim == 3:
        return UnitCubeMesh(n, n, n)
    else:
        raise ValueError("Dimension {:d} not supported".format(dim))


# ---------------------------
# standard tests for pytest
# ---------------------------

@pytest.fixture(params=[2, 3])
def dim(request):
    return request.param


def test_gradient(dim):
    mesh = get_mesh(dim, 3)
    x = SpatialCoordinate(mesh)
    P1 = FunctionSpace(mesh, "CG", 1)
    f = interpolate(x[0], P1)
    g = recover_gradient(f)
    dfdx = np.zeros(len(x))
    dfdx[0] = 1
    exact = [dfdx for i in g.dat.data]
    assert np.allclose(g.dat.data, exact)
