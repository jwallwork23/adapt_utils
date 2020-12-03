r"""
Test mesh movement and metric based adaptation in the steady state case for analytically defined
sensor functions.

Sensors as defined in

Olivier, Géraldine. Anisotropic metric-based mesh adaptation for unsteady CFD simulations involving
moving geometries. Diss. 2011.
"""
from firedrake import *

import pytest
import os
import matplotlib.pyplot as plt

from adapt_utils import *
from adapt_utils.test.adapt.sensors import *


# ---------------------------
# standard tests for pytest
# ---------------------------

@pytest.fixture(params=[bowl, hyperbolic, multiscale, interweaved])
def sensor(request):
    return request.param


@pytest.fixture(params=['complexity', 'error'])
def normalisation(request):
    return request.param


@pytest.fixture(params=[1, 2, None])
def norm_order(request):
    return request.param


def test_metric_based(sensor, normalisation, norm_order, plot_mesh=False, **kwargs):
    if os.environ.get('FIREDRAKE_ADAPT') == '0':
        pytest.xfail("Firedrake installation does not include Pragmatic")

    # Set parameters
    kwargs = {
        'approach': 'hessian',
        'h_min': 1.0e-06,
        'h_max': 1.0e-01,
        'max_adapt': kwargs.get('max_adapt', 4),
        'normalisation': normalisation,
        'norm_order': norm_order,
        'target': kwargs.get('target', 100.0 if normalisation == 'complexity' else 10.0),
    }
    op = Options(**kwargs)
    fname = '_'.join([sensor.__name__, normalisation, str(norm_order or 'inf')])
    fpath = os.path.dirname(__file__)

    # Setup initial mesh
    n = 100
    mesh = SquareMesh(n, n, 2, 2)
    x, y = SpatialCoordinate(mesh)
    mesh.coordinates.interpolate(as_vector([x-1, y-1]))

    # Adapt the mesh
    for i in range(op.max_adapt):
        P1 = FunctionSpace(mesh, "CG", 1)
        M = steady_metric(sensor(mesh), mesh=mesh, op=op)
        mesh = adapt(mesh, M)

    # Plot mesh
    if plot_mesh:
        fig, axes = plt.subplots()
        triplot(mesh, axes=axes, interior_kw={'linewidth': 0.1}, boundary_kw={'color': 'k'})
        axes.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(fpath, 'outputs', op.approach, fname + '.png'))
        plt.close()
        return

    # Save mesh coordinates to file
    if not os.path.exists(os.path.join(fpath, 'data', fname + '.npy')):
        np.save(os.path.join(fpath, 'data', fname), mesh.coordinates.dat.data)
        pytest.xfail("Needed to set up the test. Please try again.")
    loaded = np.load(os.path.join(fpath, 'data', fname + '.npy'))
    assert np.allclose(mesh.coordinates.dat.data, loaded), "Mesh does not match data"


# ---------------------------
# mesh plotting
# ---------------------------

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-sensor", help="""
        Choice of sensor function, from {'bowl', 'hyperbolic', 'multiscale', 'interweaved'}.
        """)
    parser.add_argument("-normalisation", help="Normalise by complexity or error.")
    parser.add_argument("-norm_order", help="Norm order for normalisation.")
    parser.add_argument("-target", help="Target complexity/error.")
    parser.add_argument("-num_adapt", help="Number of adaptations.")
    args = parser.parse_args()
    f = args.sensor or 'bowl'
    sensor = {
        'bowl': bowl,
        'hyperbolic': hyperbolic,
        'multiscale': multiscale,
        'interweaved': interweaved
    }[f]
    normalisation = args.normalisation or 'complexity'
    p = None if args.norm_order in ('none', 'inf') else int(args.norm_order or 1)
    target = float(args.target or 100.0)
    max_adapt = int(args.num_adapt or 4)

    test_metric_based(sensor, normalisation, p, plot_mesh=True, target=target, max_adapt=max_adapt)
