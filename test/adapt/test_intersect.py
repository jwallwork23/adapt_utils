from firedrake import *

import pytest
import numpy as np

from adapt_utils.adapt.metric import *


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

def test_interior(dim):
    mesh = get_mesh(dim, 3)
    P1_ten = TensorFunctionSpace(mesh, "CG", 1)
    M1 = interpolate(2*Identity(dim), P1_ten)
    M2 = interpolate(1*Identity(dim), P1_ten)
    M = metric_intersection(M1, M2)
    assert np.allclose(M.dat.data, M1.dat.data)
    M2.interpolate(2*Identity(dim))
    M = metric_intersection(M1, M2)
    assert np.allclose(M.dat.data, M1.dat.data)
    assert np.allclose(M.dat.data, M2.dat.data)
    M2.interpolate(4*Identity(dim))
    M = metric_intersection(M1, M2)
    assert np.allclose(M.dat.data, M2.dat.data)

def test_boundary(dim):
    mesh = get_mesh(dim, 3)
    P1_ten = TensorFunctionSpace(mesh, "CG", 1)
    M1 = interpolate(2*Identity(dim), P1_ten)
    M2 = interpolate(4*Identity(dim), P1_ten)
    M = metric_intersection(M1, M2, boundary_tag='on_boundary')
    boundary_nodes = DirichletBC(P1_ten, 0, 'on_boundary').nodes
    all_nodes = set(range(mesh.num_vertices()))
    interior_nodes = np.array(list(all_nodes.difference(set(boundary_nodes))))
    assert np.allclose(M.dat.data[interior_nodes], M1.dat.data[interior_nodes])
    assert np.allclose(M.dat.data[boundary_nodes], M2.dat.data[boundary_nodes])
