from firedrake import *
from adapt_utils.adapt.metric import steady_metric
from adapt_utils.adapt.recovery import construct_hessian
from adapt_utils.options import *
import numpy as np

import matplotlib.pyplot as plt

# Sensor tests considered in [Olivier 2011].

def sensor(i, mesh):
    x, y = SpatialCoordinate(mesh)
    return [x*x+y*y,
            conditional(ge(abs(x*y), 2*pi/50), 0.01*sin(50*x*y), sin(50*x*y)),
            0.1*sin(50*x) + atan(0.1/(sin(5*y) - 2*x)),
            atan(0.1/(sin(5*y)-2*x))+atan(0.5/(sin(3*y)-7*x))][i]

op = DefaultOptions()
op.h_min = 1e-6
op.h_max = 0.1
op.num_adapt = 4
modes = ['target', 'p_norm', 'p_norm', 'p_norm']
orders = [None, 1, 2, None]
for op.normalisation, op.norm_order in zip(modes, orders):
    if op.normalisation == 'p_norm':
        normalisation = 'l-inf' if op.norm_order is None else 'l{:d}'.format(op.norm_order)
    else:
        normalisation = 'target'
    for i in range(4):
        for k in range(3):
            print("\nNormalisation {:s}  Sensor {:d}  Iteration {:d}".format(normalisation, i, k))
            op.target = pow(10, k)
            for j in range(op.num_adapt):
                if j == 0:
                    mesh = SquareMesh(40, 40, 2, 2)
                    mesh.coordinates.dat.data[:] -= [1, 1]
                mesh = adapt(mesh, steady_metric(sensor(i, mesh), mesh=mesh, op=op))
            File('plots/normalisation_{:s}__sensor_{:d}__mesh_{:d}.pvd'.format(normalisation, i, k)).write(mesh.coordinates)
