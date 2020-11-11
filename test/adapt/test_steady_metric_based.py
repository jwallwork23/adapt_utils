from firedrake import *

import numpy as np
import os
import pytest

from adapt_utils import *


def get_mesh(dim, n):
    if dim == 2:
        return UnitSquareMesh(n, n)
    elif dim == 3:
        return UnitCubeMesh(n, n, n)
    else:
        raise ValueError("Dimension {:d} not supported".format(dim))


def check_coordinates(mesh1, mesh2):
    """
    Verify that the vertices of `mesh1` are all vertices of `mesh2`.
    """
    for v1 in mesh1.coordinates.dat.data:
        found = False
        for v2 in mesh2.coordinates.dat.data:
            if np.allclose(v1, v2):
                found = True
                break
        if not found:
            raise AssertionError("Vertex")


# ---------------------------
# standard tests for pytest
# ---------------------------

@pytest.fixture(params=[2, 3])
def dim(request):
    return request.param


def test_indentity_metric(dim):
    """
    Verify that adapting with respect to the identity metric does not change the mesh.
    """
    if os.environ.get('FIREDRAKE_ADAPT') == '0':
        pytest.xfail("Firedrake installation does not include Pragmatic")

    mesh = get_mesh(dim, 1)
    assert mesh.num_vertices() == 2**dim

    # Adapt using a hard coded identity metric
    identity = Identity(dim)/sqrt(dim)
    P1_ten = TensorFunctionSpace(mesh, "CG", 1)
    M_hardcoded = interpolate(identity, P1_ten)
    newmesh = adapt(mesh, M_hardcoded)
    assert mesh.num_vertices() == newmesh.num_vertices()
    check_coordinates(mesh, newmesh)

    # Adapt using an identity metric created using the isotropic_metric driver
    P1 = FunctionSpace(mesh, "CG", 1)
    f = Function(P1).assign(1/np.sqrt(dim))
    M = isotropic_metric(f, normalise=False, enforce_constraints=False)
    newmesh = adapt(mesh, M)
    assert np.allclose(M_hardcoded.dat.data, M.dat.data)
    assert mesh.num_vertices() == newmesh.num_vertices()
    check_coordinates(mesh, newmesh)
