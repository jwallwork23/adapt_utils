from firedrake import *

import pytest

from adapt_utils.adapt.kernels import *
from adapt_utils.linalg import check_spd


def get_mesh(dim, n=2):
    if dim == 2:
        return UnitSquareMesh(n, n)
    elif dim == 3:
        return UnitCubeMesh(n, n, n)
    else:
        raise ValueError


# ---------------------------
# standard tests for pytest
# ---------------------------

@pytest.fixture(params=[2, 3])
def dim(request):
    return request.param


def test_polar_decomposition(dim):
    mesh = get_mesh(dim)

    # Compute cell Jacobian
    P0_ten = TensorFunctionSpace(mesh, "DG", 0)
    J = Function(P0_ten, name="Cell Jacobian")
    J.interpolate(Jacobian(mesh))

    # Get SPD part
    B = Function(P0_ten, name="SPD part")
    op2.par_loop(eigen_kernel(poldec_spd, dim), P0_ten.node_set, B.dat(op2.RW), J.dat(op2.READ))
    check_spd(B)

    # Get unitary part
    Z = Function(P0_ten, name="Unitary part")
    op2.par_loop(eigen_kernel(poldec_unitary, dim), P0_ten.node_set, Z.dat(op2.RW), J.dat(op2.READ))
    ZZT = interpolate(dot(Z, transpose(Z)), P0_ten)
    I = interpolate(Identity(dim), P0_ten)
    np.allclose(ZZT.dat.data, I.dat.data)

    # Check we have a polar decomposition
    BZ = interpolate(dot(B, Z), P0_ten)
    np.allclose(J.dat.data, BZ.dat.data)
