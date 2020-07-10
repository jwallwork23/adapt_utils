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


def check_entity_counts(mesh1, mesh2):
    assert mesh1.num_vertices() == mesh2.num_vertices()
    assert mesh1.num_edges() == mesh2.num_edges()
    assert mesh1.num_cells() == mesh2.num_cells()
    assert mesh1.num_facets() == mesh2.num_facets()


def check_coordinates(mesh1, mesh2):
    """Verify that the vertices of `mesh1` are all vertices of `mesh2`."""
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
    """Verify that adapting with respect to the identity metric does not change the mesh."""
    if os.environ.get('FIREDRAKE_ADAPT') == '0':
        pytest.xfail("Firedrake installation does not include Pragmatic")

    mesh = get_mesh(dim, 1)
    assert mesh.num_vertices() == 2**dim

    # Adapt using a hard coded identity metric
    identity = Identity(dim)/sqrt(dim)
    P1_ten = TensorFunctionSpace(mesh, "CG", 1)
    M_hardcoded = interpolate(identity, P1_ten)
    newmesh = pragmatic_adapt(mesh, M_hardcoded)
    check_entity_counts(mesh, newmesh)
    check_coordinates(mesh, newmesh)

    # Adapt using an identity metric created using the isotropic_metric driver
    P1 = FunctionSpace(mesh, "CG", 1)
    f = Function(P1).assign(1/np.sqrt(dim))
    M = isotropic_metric(f, normalise=False, enforce_constraints=False)
    newmesh = pragmatic_adapt(mesh, M)
    assert np.allclose(M_hardcoded.dat.data, M.dat.data)
    check_entity_counts(mesh, newmesh)
    check_coordinates(mesh, newmesh)


def test_anisotropic_stretch(dim):
    """
    Verify that adapting with respect to metrics given by stretching an identity metric in a single
    component direction works as expected.
    """
    if os.environ.get('FIREDRAKE_ADAPT') == '0':
        pytest.xfail("Firedrake installation does not include Pragmatic")

    method = 'topological'
    mesh = get_mesh(dim, 1)
    identity = Identity(dim)/sqrt(2)
    n = 2*dim  # Number of exterior faces

    # Stretch an identity metric in each component direction only
    for i in range(dim):
        amd = AnisotropicMetricDriver(AdaptiveMesh(mesh))
        assert amd.mesh.num_vertices() == 2**dim
        amd.p1metric.interpolate(identity)
        amd.component_stretch()
        amd.p1metric.assign(amd.component_stretch_metrics[i])
        amd.adapt_mesh()

        num_vertices_face_i = len(amd.P1.boundary_nodes(2*i+1, method))
        assert num_vertices_face_i < len(amd.P1.boundary_nodes((2*i-1) % n, method))

        # Check metric intersection combines these appropriately
        amd = AnisotropicMetricDriver(AdaptiveMesh(mesh))
        amd.p1metric.interpolate(identity)
        amd.component_stretch()
        amd.p1metric.assign(metric_intersection(amd.component_stretch_metrics[i],
                                                amd.component_stretch_metrics[(i+1) % dim]))
        amd.adapt_mesh()
        num_vertices_face_i = len(amd.P1.boundary_nodes(2*i+1, method))
        assert num_vertices_face_i == len(amd.P1.boundary_nodes((2*(i+1)+1) % n, method))
