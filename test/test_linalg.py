from firedrake import *

import pytest

from adapt_utils.linalg import *
from adapt_utils.fem import cg2dg


def get_mesh(dim, n=4):
    if dim == 1:
        return UnitIntervalMesh(n)
    elif dim == 2:
        return UnitSquareMesh(n, n)
    elif dim == 3:
        return UnitCubeMesh(n, n, n)
    else:
        raise ValueError("Expected dimension 1, 2 or 3 but got {:d}".format(dim))


def get_function_spaces(mesh, shape):
    if shape == 'scalar':
        constructor = FunctionSpace
    elif shape == 'vector':
        constructor = VectorFunctionSpace
    elif shape == 'tensor':
        constructor = TensorFunctionSpace
    else:
        raise ValueError("Shape {:s} not recognised.".format(shape))
    P1 = constructor(mesh, "CG", 1)
    P1DG = constructor(mesh, "DG", 1)
    return P1, P1DG


# ---------------------------
# standard tests for pytest
# ---------------------------

# @pytest.fixture(params=[1, 2, 3])  # FIXME
@pytest.fixture(params=[2, 3])
def dim(request):
    return request.param


# @pytest.fixture(params=['scalar', 'vector', 'tensor'])  # FIXME
@pytest.fixture(params=['scalar', 'vector'])
def shape(request):
    return request.param


def test_is_spd(dim):
    mesh = get_mesh(dim)
    P1_ten, _ = get_function_spaces(mesh, 'tensor')
    M = interpolate(Identity(dim), P1_ten)
    assert is_spd(M)
    M.dat.data[0][0, 0] *= -1
    assert not is_spd(M)


def test_check_spd(dim):
    mesh = get_mesh(dim)
    P1_ten, _ = get_function_spaces(mesh, 'tensor')
    M = interpolate(Identity(dim), P1_ten)
    check_spd(M)


def test_rotation():
    v = np.array([1.0, 0.0])
    Rv = np.dot(rotation_matrix(pi/2), v)
    assert np.allclose(np.dot(v, Rv), 0.0)


def test_gram_schmidt_numpy(dim):
    v = [np.random.rand(dim) for i in range(dim)]
    orth = gram_schmidt(*v, normalise=True)
    for j in range(dim):

        # Check orthogonal
        for i in range(j):
            assert np.isclose(np.dot(orth[i], orth[j]), 0.0)

        # Check orthonormal
        assert np.isclose(np.dot(orth[j], orth[j]), 1.0)


def test_gram_schmidt_2d():
    dim = 2
    mesh = get_mesh(dim, n=1)
    n = FacetNormal(mesh)
    v = as_vector(np.random.rand(dim))
    n, s = gram_schmidt(n, v, normalise=True)

    P1_vec, _ = get_function_spaces(mesh, 'vector')
    for i, u in zip([1, 2, 3, 4], [[-1, 0], [1, 0], [0, -1], [0, 1]]):
        uu = interpolate(as_vector(u), P1_vec)
        assert np.isclose(assemble(dot(uu, n)*ds(i)), 1.0)  # Check normals align
        assert np.isclose(assemble(dot(uu, s)*ds(i)), 0.0)  # Check tangents align


def test_cg2dg(dim, shape):
    mesh = get_mesh(dim)
    P1, P1DG = get_function_spaces(mesh, shape)
    cg = Function(P1).assign(RandomGenerator(PCG64(seed=0)).normal(P1, 0.0, 1.0))
    dg = Function(P1DG)
    cg2dg(cg, dg)
    assert np.allclose(cg.dat.data, project(dg, P1).dat.data)
