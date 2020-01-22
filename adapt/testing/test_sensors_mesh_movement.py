from firedrake import *

from adapt_utils.adapt.testing.sensors import *
from adapt_utils.adapt.recovery import construct_hessian
from adapt_utils.adapt.r import *
from adapt_utils.norms import *
from adapt_utils.options import *

import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-n")
parser.add_argument("-debug")
parser.add_argument("-monitor")
args = parser.parse_args()

op = Options()
op.debug = bool(args.debug)
# op.pseudo_dt = 0.001

n = args.n or 1
n = int(n)
if n == 1:
    sensor = bowl
elif n == 2:
    sensor = hyperbolic
elif n == 3:
    sensor = multiscale
elif n == 4:
    sensor = interweaved
    op.r_adapt_rtol = 5.0e-4  # TODO: Try Quasi-Newton approach
else:
    raise ValueError
m = args.monitor or 'sensor'
try:
    assert m in ('sensor', 'frobenius')
except AssertionError:
    raise NotImplementedError
op.di = os.path.join(op.di, sensor.__name__, m)

n = 100
mesh = SquareMesh(n, n, 2, 2)
x, y = SpatialCoordinate(mesh)
mesh.coordinates.interpolate(as_vector([x-1, y-1]))
P1 = FunctionSpace(mesh, "CG", 1)

def myplot(f):
    plot(interpolate(f(mesh), P1))
    plt.title(f.__name__)
    plt.show()

def monitor_frobenius(mesh):
    H = construct_hessian(sensor(mesh), mesh=mesh, op=op)
    return 1.0 + local_frobenius_norm(H, space=P1)

def monitor_sensor(mesh, alpha=5.0):
    return 1.0 + alpha*abs(sensor(mesh))

if m == 'sensor':
    monitor = monitor_sensor
elif m == 'frobenius':
    monitor = monitor_frobenius

mm = MeshMover(mesh, monitor, op=op)
mm.adapt()
mesh.coordinates.assign(mm.x)
