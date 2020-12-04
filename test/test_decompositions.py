from firedrake import *

import pytest

from adapt_utils.adapt.kernels import *
from adapt_utils.adapt.metric import get_density_and_quotients, steady_metric
from adapt_utils.linalg import check_spd
from adapt_utils.misc import prod


def uniform_mesh(dim, n, l=1):
    if dim == 2:
        return SquareMesh(n, n, l)
    elif dim == 3:
        return CubeMesh(n, n, n, l)
    else:
        raise ValueError("Dimension {:d} not supported.".format(dim))


# ---------------------------
# standard tests for pytest
# ---------------------------

@pytest.fixture(params=[2, 3])
def dim(request):
    return request.param


def test_eigendecomposition(dim):
    mesh = uniform_mesh(dim, 20, l=2)
    P1_vec = VectorFunctionSpace(mesh, "CG", 1)
    P1_ten = TensorFunctionSpace(mesh, "CG", 1)

    # Recover Hessian for some arbitrary sensor
    f = prod([sin(pi*xi) for xi in SpatialCoordinate(mesh)])
    M = steady_metric(f, mesh=mesh, V=P1_ten, normalise=False, enforce_constraints=False)

    # Extract the eigendecomposition
    V = Function(P1_ten, name="Eigenvectors")
    Λ = Function(P1_vec, name="Eigenvalues")
    kernel = eigen_kernel(get_eigendecomposition, dim)
    op2.par_loop(kernel, P1_ten.node_set, V.dat(op2.RW), Λ.dat(op2.RW), M.dat(op2.READ))

    # Reassemble it and check the two match
    VΛVT = Function(P1_ten, name="Reassembled matrix")
    kernel = eigen_kernel(set_eigendecomposition, dim)
    op2.par_loop(kernel, P1_ten.node_set, VΛVT.dat(op2.RW), V.dat(op2.READ), Λ.dat(op2.READ))
    # assert np.allclose(M.dat.data, VΛVT.dat.data, rtol=1.0e-5)
    assert np.allclose(M.dat.data, VΛVT.dat.data)


def test_density_quotients_decomposition(dim):
    mesh = uniform_mesh(dim, 20, l=2)
    P1_vec = VectorFunctionSpace(mesh, "CG", 1)
    P1_ten = TensorFunctionSpace(mesh, "CG", 1)

    # Recover Hessian for some arbitrary sensor
    f = prod([sin(pi*xi) for xi in SpatialCoordinate(mesh)])
    M = steady_metric(f, mesh=mesh, V=P1_ten, normalise=False, enforce_constraints=False)

    # Extract the eigendecomposition
    V = Function(P1_ten, name="Eigenvectors")
    Λ = Function(P1_vec, name="Eigenvalues")
    kernel = eigen_kernel(get_eigendecomposition, dim)
    op2.par_loop(kernel, P1_ten.node_set, V.dat(op2.RW), Λ.dat(op2.RW), M.dat(op2.READ))

    # Extract the density and anisotropy quotients
    d, Q = get_density_and_quotients(M)
    Λ.interpolate(as_vector([pow(d/Q[i], 2/dim) for i in range(dim)]))

    # Reassemble the matrix and check the two match
    dVQVT = Function(P1_ten, name="Reassembled matrix")
    kernel = eigen_kernel(set_eigendecomposition, dim)
    op2.par_loop(kernel, P1_ten.node_set, dVQVT.dat(op2.RW), V.dat(op2.READ), Λ.dat(op2.READ))
    assert np.allclose(M.dat.data, dVQVT.dat.data, rtol=1.0e-3)


def test_polar_decomposition_cell_jacobian(dim):
    mesh = uniform_mesh(dim, 2)

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


if __name__ == "__main__":
    test_density_quotients_decomposition(2)
