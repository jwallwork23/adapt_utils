r"""
Test mesh movement and metric based adaptation in the steady state case for analytically defined
sensor functions.

Sensors as defined in

Olivier, Géraldine. Anisotropic metric-based mesh adaptation for unsteady CFD simulations involving
moving geometries. Diss. 2011.
"""
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


def test_metric_based(sensor, normalisation, norm_order, plot_mesh=False):
    if os.environ.get('FIREDRAKE_ADAPT') == '0':
        pytest.xfail("Firedrake installation does not include Pragmatic")

    # Set parameters
    kwargs = {
        'approach': 'hessian',
        'h_min': 1.0e-06,
        'h_max': 1.0e-01,
        'max_adapt': 4,
        'normalisation': normalisation,
        'norm_order': norm_order,
        'target': 100.0 if normalisation == 'complexity' else 10.0
    }
    op = Options(**kwargs)
    fname = '_'.join([sensor.__name__, normalisation, str(norm_order or 'inf')])
    fpath = os.path.dirname(__file__)

    # Create domain
    n = 100
    mesh = SquareMesh(n, n, 2, 2)
    x, y = SpatialCoordinate(mesh)
    mesh.coordinates.interpolate(as_vector([x-1, y-1]))

    # Adapt the mesh
    P1 = FunctionSpace(mesh, "CG", 1)
    f = interpolate(sensor(mesh), P1)
    M = steady_metric(f, op=op)
    newmesh = pragmatic_adapt(mesh, M)

    # Plot mesh
    if plot_mesh:
        fig, axes = plt.subplots()
        triplot(newmesh, axes=axes)
        axes.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(fpath, 'outputs', op.approach, fname + '.png'))
        plt.close()

    # Save mesh coordinates to file
    if not os.path.exists(os.path.join(fpath, 'data', fname + '.npy')):
        np.save(os.path.join(fpath, 'data', fname), newmesh.coordinates.dat.data)
        if not plot_mesh:
            pytest.xfail("Needed to set up the test. Please try again.")
    loaded = np.load(os.path.join(fpath, 'data', fname + '.npy'))
    assert np.allclose(newmesh.coordinates.dat.data, loaded), "Mesh does not match data"


# ---------------------------
# mesh plotting
# ---------------------------

if __name__ == '__main__':
    for f in (bowl, hyperbolic, multiscale, interweaved):
        for n in ('complexity', 'error'):
            for p in (1, 2, None):
                test_metric_based(f, n, p, plot_mesh=True)
