from firedrake import *
from adapt_utils import *
import numpy as np
import matplotlib.pyplot as plt

op = Options()

# Simple mesh of two right angled triangles
amd = AnisotropicMetricDriver(AdaptiveMesh(UnitSquareMesh(1, 1)))
assert amd.mesh.num_vertices() == 4
coords = amd.mesh.coordinates.copy()
identity = as_matrix([[1/sqrt(2), 0], [0, 1/sqrt(2)]])
method = 'topological'


def fail(amd, msg):
    print(msg)
    plot(amd.mesh)
    plt.show()
    exit(0)


# Check nothing happens when we adapt with the identity metric
print('Testing hard-coded metric')
amd.p1metric.interpolate(identity)
amd.adapt_mesh()
try:
    assert np.allclose(coords.dat.data, amd.mesh.coordinates.dat.data)
    assert len(amd.P1.boundary_nodes(1, method)) == len(amd.P1.boundary_nodes(3, method))
except AssertionError:
    fail(amd, "FAIL: Hard-coded metric")

# Check isotropic metric does the same thing
print('Testing isotropic metric')
f = Function(amd.P1).assign(1/np.sqrt(2))
amd.p1metric.assign(isotropic_metric(f, normalise=False))
amd.adapt_mesh()
try:
    assert np.allclose(coords.dat.data, amd.mesh.coordinates.dat.data)
    assert len(amd.P1.boundary_nodes(1, method)) == len(amd.P1.boundary_nodes(3, method))
except AssertionError:
    fail(amd, "FAIL: Isotropic metric")

# Check anistropic refinement in x-direction
print('Testing anisotropic metric 0')
amd.p1metric.interpolate(identity)
amd.component_stretch()
amd.p1metric = amd.component_stretch_metrics[0]
amd.adapt_mesh()
try:
    assert amd.mesh.num_vertices() == 6
    assert len(amd.P1.boundary_nodes(1, method)) < len(amd.P1.boundary_nodes(3, method))
except AssertionError:
    fail(amd, "FAIL: Anisotropic metric 0")

# Check anistropic refinement in y-direction
print('Testing anisotropic metric 1')
amd.p1metric.interpolate(identity)
amd.component_stretch()
amd.p1metric = amd.component_stretch_metrics[1]
amd.adapt_mesh()
try:
    assert amd.mesh.num_vertices() == 6
    assert len(amd.P1.boundary_nodes(1, method)) > len(amd.P1.boundary_nodes(3, method))
except AssertionError:
    fail(amd, "FAIL: Anisotropic metric 1")

# Check metric intersection combines these appropriately
print('Testing metric intersection')
amd.p1metric.interpolate(identity)
amd.component_stretch()
amd.p1metric.assign(metric_intersection(amd.component_stretch_metrics[0],
                                        amd.component_stretch_metrics[1]))
amd.adapt_mesh()
try:
    assert amd.mesh.num_vertices() == 9
    assert len(amd.P1.boundary_nodes(1, method)) == len(amd.P1.boundary_nodes(3, method))
except AssertionError:
    fail(amd, "FAIL: Metric intersection")

print("All tests passed!")
