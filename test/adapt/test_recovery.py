from firedrake import *

import pytest
import numpy as np

from adapt_utils.adapt.recovery import *
from adapt_utils.unsteady.swe.utils import recover_vorticity


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


def test_vorticity():
    r"""
    Given a velocity field :math:`\mathbf u = (u,v)`, the curl is defined by

  ..math::
        \mathrm{curl}(\mathbf u) := \frac{\partial v}{\partial x} - \frac{\partial u}{\partial y}.

    For a simple velocity field :math:`\mathbf u = 0.5* (-y, x)` the vorticity should be unity
    everywhere.
    """
    mesh = get_mesh(2, 3)
    x, y = SpatialCoordinate(mesh)
    P1_vec = VectorFunctionSpace(mesh, "CG", 1)
    u, v = -0.5*y, 0.5*x
    uv = interpolate(as_vector([u, v]), P1_vec)
    zeta = recover_vorticity(uv)

    dudy, dvdx = -0.5, 0.5
    curl_uv = dvdx - dudy
    exact = [curl_uv for i in zeta.dat.data]
    assert np.allclose(zeta.dat.data, exact)
