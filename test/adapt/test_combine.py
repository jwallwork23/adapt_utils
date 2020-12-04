from firedrake import *

import matplotlib.pyplot as plt
from wurlitzer import pipes

from adapt_utils.adapt.metric import *
from adapt_utils.options import Options
from adapt_utils.plotting import *


# ---------------------------
# standard tests for pytest
# ---------------------------

def test_complexity(plot=False):
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

    # Loop over different metric construction modes
    data = {1: {}, 2: {}, 'avg': {}, 'int': {}}
    for mode in data:

        # Create domain [0, 1]²
        mesh = UnitSquareMesh(n, n)
        for i in range(op.max_adapt):
            x, y = SpatialCoordinate(mesh)
            P1_ten = TensorFunctionSpace(mesh, "CG", 1)

            # Create a metric focused around the arc x² + y² = ½
            f = exp(-alpha*abs(0.5 - x**2 - y**2))
            M1 = steady_metric(f, V=P1_ten, op=op, enforce_constraints=True)

            # Create a metric focused around the arc (1 - x)² + y² = ½
            g = exp(-alpha*abs(0.5 - (1 - x)**2 - y**2))
            M2 = steady_metric(g, V=P1_ten, op=op, enforce_constraints=True)

            # Choose metric according to mode
            M = {1: M1, 2: M2, 'avg': metric_average(M1, M2), 'int': metric_intersection(M1, M2)}[mode]
            # Adapt mesh
            with pipes() as (out, err):
                mesh = adapt(mesh, M)

        # Store entity counts
        data[mode]['complexity'] = metric_complexity(M)
        data[mode]['num_cells'] = mesh.num_cells()
        data[mode]['num_vertices'] = mesh.num_vertices()

        # Plot
        if plot:
            fig, axes = plt.subplots(figsize=(5, 5))
            triplot(mesh, axes=axes, interior_kw={'linewidth': 0.1}, boundary_kw={'color': 'k'})
            axes.axis(False)
            savefig('mesh_{:}'.format(mode), 'outputs/hessian', extensions=['png'])

    # Check complexity follows C(M1) ≤ C(M1 ∩ M2) and similarly for M2
    msg = "M_{:d} does not have {:s} {:s} than M_{:s}"
    i = 'complexity'
    for j in (1, 2):
        assert data[j][i] <= data['int'][i], msg.format(j, 'fewer', i, 'int')

    # Similarly for element counts, also with lower bound
    for i in ('num_cells', 'num_vertices'):
        for j in (1, 2):
            assert data['avg'][i] <= data[j][i], msg.format(j, 'more', i, 'avg')
            assert data[j][i] <= data['int'][i], msg.format(j, 'fewer', i, 'int')

    # Print mesh data to screen
    if plot:
        for mode in data:
            print("Mode M_{:}".format(mode))
            print("number of elements = {:d}".format(data[mode]['num_cells']))
            print("number of vertices = {:d}".format(data[mode]['num_vertices']))


if __name__ == "__main__":
    test_complexity()
