from firedrake import *

import matplotlib.pyplot as plt

from adapt_utils.adapt.metric import *
from adapt_utils.options import Options
from adapt_utils.plotting import *


# Set parameters
kwargs = {
    'approach': 'hessian',
    'max_adapt': 4,
    'normalisation': 'complexity',
    'norm_order': 1,
    'target': 200.0,
    'h_min': 1.0e-06,
    'h_max': 1.0,
}
op = Options(**kwargs)
n = 100
alpha = 1

# Adapt mesh
data = {1: {}, 2: {}, 'avg': {}, 'int': {}}
for mode in data:
    mesh = UnitSquareMesh(n, n)
    for approach in range(op.max_adapt):
        x, y = SpatialCoordinate(mesh)
        P1_ten = TensorFunctionSpace(mesh, "CG", 1)

        f = exp(-alpha*abs(0.5 - x**2 - y**2))
        M1 = steady_metric(f, V=P1_ten, op=op, enforce_constraints=True)

        g = exp(-alpha*abs(0.5 - (1 - x)**2 - y**2))
        M2 = steady_metric(g, V=P1_ten, op=op, enforce_constraints=True)

        M = {1: M1, 2: M2, 'avg': metric_average(M1, M2), 'int': metric_intersection(M1, M2)}[mode]
        mesh = adapt(mesh, M)
    data[mode]['num_cells'] = mesh.num_cells()
    data[mode]['num_vertices'] = mesh.num_vertices()

    # Plot
    fig, axes = plt.subplots(figsize=(5, 5))
    triplot(mesh, axes=axes, interior_kw={'linewidth': 0.1}, boundary_kw={'color': 'k'})
    axes.axis(False)
    savefig('mesh_{:}'.format(mode), 'outputs/hessian', extensions=['png'])

# Print mesh data
for mode in data:
    print("Mode M_{:}".format(mode))
    print("number of elements = {:d}".format(data[mode]['num_cells']))
    print("number of vertices = {:d}".format(data[mode]['num_vertices']))
