from firedrake import *
from adapt_utils.adapt.metric import steady_metric
from adapt_utils.adapt.recovery import construct_hessian
from adapt_utils.options import *

import matplotlib.pyplot as plt

op = DefaultOptions()
#op.restrict = 'num_cells'
op.h_min = 1e-4
op.h_max = 0.1
op.num_adapt = 4
op.target_vertices = 1000
for i in range(2):
    for j in range(op.num_adapt):
        if j == 0:
            mesh = SquareMesh(40, 40, 2, 2)
            mesh.coordinates.dat.data[:] -= [1,1]
        P1 = FunctionSpace(mesh, "CG", 1)
        x, y = SpatialCoordinate(mesh)
        f = interpolate([x*x+y*y, atan(0.1/(sin(5*y)-2*x))+atan(0.5/(sin(3*y)-7*x))][i], P1)
        File('plots/sensor{:d}.pvd'.format(i)).write(f)
        H = construct_hessian(f)
        File('plots/hessian{:d}.pvd'.format(i)).write(H)
        M = steady_metric(f, H=H, op=op)
        File('plots/metric{:d}.pvd'.format(i)).write(M)
        mesh = adapt(mesh, M)
        File('plots/mesh{:d}.pvd'.format(i)).write(mesh.coordinates)
