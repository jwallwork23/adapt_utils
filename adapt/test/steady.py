from firedrake import *
from adapt_utils.adapt.metric import steady_metric
from adapt_utils.options import *


# Sensor tests considered in [Olivier 2011].

# TODO: Rewrite this test in a nicer way

def sensor(i, mesh):
    x, y = SpatialCoordinate(mesh)
    return [x*x+y*y,
            conditional(ge(abs(x*y), 2*pi/50), 0.01*sin(50*x*y), sin(50*x*y)),
            0.1*sin(50*x) + atan(0.1/(sin(5*y) - 2*x)),
            atan(0.1/(sin(5*y)-2*x))+atan(0.5/(sin(3*y)-7*x))][i]


op = Options()
op.h_min = 1e-6
op.h_max = 0.1
op.num_adapt = 4
modes = ['complexity', 'error']
orders = [None, 1, 2]
levels = 3
for op.normalisation in modes:
    for op.norm_order in orders:
        normalisation = op.normalisation
        normalisation += 'l-inf' if op.norm_order is None else 'l{:d}'.format(op.norm_order)
        for i in range(4):
            for k in range(levels):
                print("\nNormalisation {:s}  Sensor {:d}  Iteration {:d}".format(normalisation, i, k))
                op.target = pow(10, k)
                for j in range(op.num_adapt):
                    if j == 0:
                        mesh = SquareMesh(40, 40, 2, 2)
                        mesh.coordinates.dat.data[:] -= [1, 1]
                    mesh = adapt(mesh, steady_metric(sensor(i, mesh), mesh=mesh, op=op))
                File('_'.join('plots/normalisation', normalisation, 'sensor', str(i), 'mesh', str(k) + '.pvd')).write(mesh.coordinates)
