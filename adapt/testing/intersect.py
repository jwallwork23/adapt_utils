from firedrake import *
import numpy as np
from adapt_utils.adapt.metric import *

# test global intersection
mesh = UnitSquareMesh(1, 1)
P1_ten = TensorFunctionSpace(mesh, "CG", 1)
M1 = interpolate(as_matrix([[2, 0], [0, 2]]), P1_ten)
M2 = interpolate(as_matrix([[0.5, 0], [0, 0.5]]), P1_ten)
M = metric_intersection(M1, M2)
assert np.abs(M.dat.data[0][0, 0] - 2) < 1e-8
assert np.abs(M.dat.data[0][1, 1] - 2) < 1e-8
M2.interpolate(as_matrix([[2, 0], [0, 2]]))
M = metric_intersection(M1, M2)
assert np.abs(M.dat.data[0][0, 0] - 2) < 1e-8
assert np.abs(M.dat.data[0][1, 1] - 2) < 1e-8
M2.interpolate(as_matrix([[4, 0], [0, 4]]))
M = metric_intersection(M1, M2)
assert np.abs(M.dat.data[0][0, 0] - 4) < 1e-8
assert np.abs(M.dat.data[0][1, 1] - 4) < 1e-8

# test boundary intersection
mesh = UnitSquareMesh(3, 3)
P1_ten = TensorFunctionSpace(mesh, "CG", 1)
M1 = interpolate(as_matrix([[2, 0], [0, 2]]), P1_ten)
M2 = interpolate(as_matrix([[4, 0], [0, 4]]), P1_ten)
M = metric_intersection(M1, M2, bdy='on_boundary')
bdy = DirichletBC(P1_ten, 0, 'on_boundary').nodes
for i in range(len(M.dat.data)):
    if i in bdy:
        assert np.abs(M.dat.data[i][0, 0] - 4) < 1e-8
        assert np.abs(M.dat.data[i][1, 1] - 4) < 1e-8
    else:
        assert np.abs(M.dat.data[i][0, 0] - 2) < 1e-8
        assert np.abs(M.dat.data[i][1, 1] - 2) < 1e-8

# TODO: test 3d intersection
