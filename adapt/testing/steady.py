from firedrake import *
from adapt_utils.adapt.metric import steady_metric
from adapt_utils.adapt.recovery import construct_hessian
from adapt_utils.options import *
import numpy as np

import matplotlib.pyplot as plt

op = DefaultOptions()
#op.restrict = 'p_norm'
op.restrict = 'target'
op.h_min = 1e-4
op.h_max = 0.1
op.num_adapt = 4
for i in range(2):
    for k in range(3):
        op.target = pow(10, k)
        for j in range(op.num_adapt):
            if j == 0:
                mesh = SquareMesh(40, 40, 2, 2)
                mesh.coordinates.dat.data[:] -= [1,1]
            P1 = FunctionSpace(mesh, "CG", 1)
            x, y = SpatialCoordinate(mesh)
            f = interpolate([x*x+y*y, atan(0.1/(sin(5*y)-2*x))+atan(0.5/(sin(3*y)-7*x))][i], P1)
            H = construct_hessian(f)
            M = steady_metric(f, H=H, op=op)  # TODO: could use noscale option
            mesh = adapt(mesh, M)
            File('plots/mesh{:d}_{:d}.pvd'.format(i, k)).write(mesh.coordinates)
