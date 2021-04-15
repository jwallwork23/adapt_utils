from firedrake import *

import pytest
import numpy as np

from adapt_utils.adapt.recovery import *
from adapt_utils.swe.utils import recover_vorticity
from adapt_utils.options import CoupledOptions


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


# @pytest.fixture(params=['parts', 'dL2'])  # FIXME
@pytest.fixture(params=['dL2'])
def hessian_recovery(request):
    return request.param


def test_gradient(dim):
    r"""
    Given a simple field :math:`f = x`, we check that the recovered gradient matches the analytical
    gradient, :math:`\nabla f=(1, 0)`.
    """
    mesh = get_mesh(dim, 3)
    x = SpatialCoordinate(mesh)
    P1 = FunctionSpace(mesh, "CG", 1)

    # Define test function
    f = interpolate(x[0], P1)

    # Recover gradient using L2 projection
    g = recover_gradient(f)

    # Compare with analytical solution
    dfdx = np.zeros(dim)
    dfdx[0] = 1
    analytical = [dfdx for i in g.dat.data]
    assert np.allclose(g.dat.data, analytical)


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

    # Define test function
    u, v = -0.5*y, 0.5*x
    uv = interpolate(as_vector([u, v]), P1_vec)

    # Recover vorticity using L2 projection
    zeta = recover_vorticity(uv)

    # Compare with analytical solution
    dudy, dvdx = -0.5, 0.5
    curl_uv = dvdx - dudy
    analytical = [curl_uv for i in zeta.dat.data]
    assert np.allclose(zeta.dat.data, analytical)


# TODO: Do not consider nodes next to boundary either, then reduce atol
def test_hessian(dim, hessian_recovery):
    r"""
    Given a simple field :math:`f = 0.5 \mathbf x \cdot \mathbf x`, we check that the recovered
    Hessian matches the analytical Hessian: the identity matrix.
    """
    mesh = get_mesh(dim, 10)
    x = SpatialCoordinate(mesh)
    P1 = FunctionSpace(mesh, "CG", 1)

    # Define test function
    f = interpolate(0.5*sum(xi**2 for xi in x), P1)

    # Recover Hessian using double L2 projection
    op = CoupledOptions(hessian_recovery=hessian_recovery)
    H = recover_hessian(f, op=op)

    # Do not consider boundary nodes
    bnd_nodes = DirichletBC(P1, 0, 'on_boundary').nodes
    H_interior = [Hi for i, Hi in enumerate(H.dat.data) if i not in bnd_nodes]

    # Compare with analytical solution
    d2fdx2 = np.identity(dim)
    analytical = [d2fdx2 for i in H_interior]
    assert np.allclose(H_interior, analytical, atol=0.15)
