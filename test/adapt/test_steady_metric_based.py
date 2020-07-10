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

# ---------------------------
# standard tests for pytest
# ---------------------------

@pytest.fixture(params=[2, 3])
def dim(request):
    return request.param


def test_indentity_metric(dim):
    if os.environ.get('FIREDRAKE_ADAPT') == '0':
        pytest.xfail("Firedrake installation does not include Pragmatic")

    op = Options()
    mesh = get_mesh(dim, 1)
    assert mesh.num_vertices = dim**2

    # Adapt using a hard coded identity metric
    identity = Identity(dim)/sqrt(2)
    P1_ten = TensorFunctionSpace(mesh, "CG", 1)
    M_hardcoded = interpolate(identity, P1_ten)
    newmesh = adapt(mesh, M_hardcoded)
    assert np.allclose(mesh.coordinates.dat.data, newmesh.coordinates.dat.data)

    # Adapt using an identity metric created using the isotropic_metric driver
    P1 = FunctionSpace(mesh, "CG", 1)
    f = Function(P1).assign(1/np.sqrt(2))
    M = isotropic(f, normalise=False)
    newmesh = adapt(mesh, M)
    assert np.allclose(M_hardcoded.dat.data, M.dat.data)
    assert np.allclose(mesh.coordinates.dat.data, newmesh.coordinates.dat.data)


def test_anisotropic_stretch(dim):
    if os.environ.get('FIREDRAKE_ADAPT') == '0':
        pytest.xfail("Firedrake installation does not include Pragmatic")

    # Create an AnisotropicMetricDriver object
    op = Options()
    method = 'topological'
    mesh = get_mesh(dim, 1)
    amd = AnisotropicMetricDriver(AdaptiveMesh(mesh))
    assert amd.mesh.num_vertices = 2**dim
    identity = Identity(dim)/sqrt(2)

    # Stretch an identity metric in the x-direction only
    amd.p1metric.interpolate(identity)
    amd.component_stretch()
    amd.p1metric.assign(amd.component_stretch_metrics[0])
    amd.adapt_mesh()
    assert amd.mesh.num_vertices() == 3*2**(dim-1)
    assert len(amd.P1.boundary_nodes(1, method)) < len(amd.P1.boundary_nodes(3, method))

    # Stretch an identity metric in the y-direction only
    amd.p1metric.interpolate(identity)
    amd.component_stretch()
    amd.p1metric.assign(amd.component_stretch_metrics[1])
    amd.adapt_mesh()
    assert amd.mesh.num_vertices() == 3*2**(dim-1)
    assert len(amd.P1.boundary_nodes(1, method)) > len(amd.P1.boundary_nodes(3, method))

    # Check metric intersection combines these appropriately
    amd.p1metric.interpolate(identity)
    amd.component_stretch()
    amd.p1metric.assign(metric_intersection(amd.component_stretch_metrics[0],
                                            amd.component_stretch_metrics[1]))
    amd.adapt_mesh()
    assert amd.mesh.num_vertices() == 9*(dim-1)
    assert len(amd.P1.boundary_nodes(1, method)) == len(amd.P1.boundary_nodes(3, method))
