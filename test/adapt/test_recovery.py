from firedrake import *

import matplotlib.pyplot as plt
import numpy as np
import pytest

from adapt_utils.adapt.recovery import *
from adapt_utils.options import Options
from adapt_utils.swe.utils import recover_vorticity


def uniform_mesh(dim, n, l=1):
    if dim == 2:
        return SquareMesh(n, n, l)
    elif dim == 3:
        return CubeMesh(n, n, n, l)
    else:
        raise ValueError("Dimension {:d} not supported".format(dim))


# ---------------------------
# standard tests for pytest
# ---------------------------

@pytest.fixture(params=[2, 3])
def dim(request):
    return request.param


@pytest.fixture(params=[True, False])
def interp(request):
    return request.param


# @pytest.fixture(params=['parts', 'dL2'])
@pytest.fixture(params=['dL2'])
def hessian_recovery(request):
    return request.param


def test_gradient_linear(dim):
    r"""
    Given a simple linear field :math:`f = x`, we
    check that the recovered gradient matches the
    analytical gradient, :math:`\nabla f = (1, 0)`.
    """
    mesh = uniform_mesh(dim, 3)
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


def test_vorticity_anticlockwise():
    r"""
    Given a velocity field :math:`\mathbf u = (u,v)`,
    the curl is defined by

  ..math::
        \mathrm{curl}(\mathbf u) :=
            \frac{\partial v}{\partial x}
            - \frac{\partial u}{\partial y}.

    For a simple velocity field,

  ..math::
        \mathbf u = 0.5* (-y, x),

    the vorticity should be unity everywhere.
    """
    mesh = uniform_mesh(2, 3)
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


def test_hessian_bowl(dim, interp, hessian_recovery, plot=False):
    r"""
    Check that the recovered Hessian of the quadratic
    function

  ..math::
        u(\mathbf x) = \frac12(\mathbf x \cdot \mathbf x)

    is the identity matrix.

    :arg interp: toggle whether or not to interpolate
        into :math:`\mathbb P1` space before recovering
        the Hessian.
    """
    op = Options(hessian_recovery=hessian_recovery)
    n = 100 if dim == 2 else 40
    if dim == 3 and interp:
        pytest.xfail("We cannot expect recovery to be good here.")
    mesh = uniform_mesh(dim, n, 2)
    x = SpatialCoordinate(mesh)
    mesh.coordinates.interpolate(as_vector([xi - 1 for xi in x]))

    # Construct Hessian
    f = 0.5*dot(x, x)
    if interp:
        f = interpolate(f, FunctionSpace(mesh, "CG", 1))
    H = recover_hessian(f, mesh=mesh, op=op)

    # Construct analytical solution
    I = interpolate(Identity(dim), H.function_space())

    # Check correspondence
    tol = 1.0e-5 if dim == 3 else 0.8 if interp else 1.0e-6
    assert np.allclose(H.dat.data, I.dat.data, atol=tol)
    if not plot:
        return
    if dim != 2:
        raise ValueError("Cannot plot in {:d} dimensions".format(dim))

    # Plot errors on scatterplot
    H_arr = np.abs(H.dat.data - I.dat.data)
    ones = np.ones(len(H_arr))
    fig, axes = plt.subplots(figsize=(5, 6))
    for i in range(2):
        for j in range(2):
            axes.scatter((2*i + j + 1)*ones, H_arr[:, i, j], marker='x')
    axes.set_yscale('log')
    axes.set_xlim([0, 5])
    axes.set_xticks([1, 2, 3, 4])
    axes.set_xticklabels(["(0,0)", "(0,1)", "(1,0)", "(1,1)"])
    axes.set_xlabel("Hessian component")
    if interp:
        axes.set_yticks([1e-15, 1e-10, 1e-5, 1])
    axes.set_ylabel("Absolute error")
    savefig("hessian_errors_bowl", "outputs/hessian", extensions=["pdf"])


# ---------------------------
# plotting
# ---------------------------

if __name__ == "__main__":
    test_hessian_bowl(2, True, 'dL2', plot=True)
    test_hessian_bowl(2, False, 'dL2', plot=True)
