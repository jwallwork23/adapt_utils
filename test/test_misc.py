from firedrake import *

import pytest
import numpy as np

from adapt_utils.misc import *


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


def test_cg2dg(dim, shape):
    mesh = get_mesh(dim)
    P1, P1DG = get_function_spaces(mesh, shape)
    cg = Function(P1).assign(RandomGenerator(PCG64(seed=0)).normal(P1, 0.0, 1.0))
    dg = Function(P1DG)
    cg2dg(cg, dg)
    assert np.allclose(cg.dat.data, project(dg, P1).dat.data)


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
