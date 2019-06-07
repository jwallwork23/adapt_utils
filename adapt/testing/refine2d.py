from firedrake import *

from adapt_utils import *

import numpy as np
import matplotlib.pyplot as plt

op = DefaultOptions()

# simple mesh of two right angled triangles
mesh = UnitSquareMesh(1, 1)
assert mesh.num_vertices() == 4
P1 = FunctionSpace(mesh, "CG", 1)
P1_ten = TensorFunctionSpace(mesh, "CG", 1)
f = Function(P1)
M = Function(P1_ten)

# check nothing happens when we adapt with the identity metric
print('Testing hard-coded metric')
M.interpolate(as_matrix([[1/np.sqrt(2), 0], [0, 1/np.sqrt(2)]]))
mesh2 = AnisotropicAdaptation(mesh, M).adapted_mesh
try:
    assert np.max(mesh.coordinates.dat.data - mesh2.coordinates.dat.data) < 1e-8
except:
    print("FAIL: Hard-coded metric")
    plot(mesh2)
    plt.show()
    exit(0)

# check isotropic metric does the same thing
print('Testing isotropic metric')
f.assign(2/np.sqrt(2))
M = isotropic_metric(f, noscale=True)
mesh2 = AnisotropicAdaptation(mesh, M).adapted_mesh
try:
    assert np.max(mesh.coordinates.dat.data - mesh2.coordinates.dat.data) < 1e-8
except:
    print("FAIL: Isotropic metric")
    plot(mesh2)
    plt.show()
    exit(0)

# check anistropic refinement in x-direction
print('Testing anisotropic metric 0')
M2 = anisotropic_refinement(M, direction=0)
mesh2 = AnisotropicAdaptation(mesh, M2).adapted_mesh
try:
    assert mesh2.num_vertices() == 6
    # TODO: check there are more cells in x-direction
except:
    print("FAIL: Anisotropic metric 0")
    plot(mesh2)
    plt.show()
    exit(0)

# check anistropic refinement in y-direction
print('Testing anisotropic metric 1')
M = interpolate(as_matrix([[1/np.sqrt(2), 0], [0, 1/np.sqrt(2)]]), P1_ten)
M3 = anisotropic_refinement(M, direction=1)
mesh2 = AnisotropicAdaptation(mesh, M3).adapted_mesh
try:
    assert mesh2.num_vertices() == 6
    # TODO: check there are more cells in y-direction
except:
    print("FAIL: Anisotropic metric 1")
    plot(mesh2)
    plt.show()
    exit(0)

# check metric intersection combines these appropriately
print('Testing metric intersection')
M4 = metric_intersection(M2, M3)
mesh2 = AnisotropicAdaptation(mesh, M4).adapted_mesh
try:
    assert mesh2.num_vertices() == 9
except:
    print("FAIL: Metric intersection")
    plot(mesh2)
    plt.show()
    exit(0)
