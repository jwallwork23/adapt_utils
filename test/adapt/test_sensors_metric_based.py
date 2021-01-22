r"""
Test mesh movement and metric based adaptation in the steady state case for analytically defined
sensor functions defined in [Olivier 2011].

[Olivier 2011] Olivier, Géraldine. Anisotropic metric-based mesh adaptation for unsteady CFD
    simulations involving moving geometries. Diss. 2011.
"""
from firedrake import *

import matplotlib.pyplot as plt
import numpy as np
import os
import pytest
from wurlitzer import pipes

from adapt_utils import *
from adapt_utils.mesh import plot_quality, plot_aspect_ratio
from adapt_utils.test.adapt.sensors import *


# ---------------------------
# standard tests for pytest
# ---------------------------

@pytest.fixture(params=[bowl, hyperbolic, multiscale, interweaved])
def sensor(request):
    return request.param


@pytest.fixture(params=['L2', 'ZZ'])
def recovery(request):
    return request.param


# @pytest.fixture(params=['complexity', 'error'])
@pytest.fixture(params=['complexity', 'error'])
def normalisation(request):
    return request.param


@pytest.fixture(params=[1, 2, None])
def norm_order(request):
    return request.param


def test_sensors(sensor, recovery, normalisation, norm_order, plot_mesh=False, **kwargs):
    """
    For each sensor function and each configuration of the adaptation parameters, construct a
    metric, adapt the mesh and check that its coordinates match those computed previously.
    """
    if os.environ.get('FIREDRAKE_ADAPT') == '0':
        pytest.xfail("Firedrake installation doesn't include Pragmatic.")
    if sensor == multiscale and normalisation == 'error' and norm_order is None:
        pytest.xfail("L-infinity normalisation cannot cope with this problem!")
    if sensor == hyperbolic and recovery == 'ZZ':
        pytest.xfail("Zienkiewicz-Zhu gives singular matrices for this sensor.")  # FIXME: Add condition for sufficient number of elements in patch
    interp = kwargs.get('interp', recovery == 'ZZ')

    # Set parameters
    kwargs = {
        'approach': 'hessian',
        'max_adapt': kwargs.get('max_adapt', 4),
        'normalisation': normalisation,
        'norm_order': norm_order,
        'hessian_recovery': recovery,
        'target': kwargs.get('target', 100.0 if normalisation == 'complexity' else 10.0),
    }
    op = Options(**kwargs)
    fname = '_'.join([sensor.__name__, recovery.lower(), normalisation, str(norm_order or 'inf')])
    fpath = os.path.dirname(__file__)

    # Setup initial mesh
    n = 100
    mesh = SquareMesh(n, n, 2)
    mesh.coordinates.interpolate(as_vector([xi-1 for xi in SpatialCoordinate(mesh)]))

    # Adapt the mesh
    for i in range(op.max_adapt):
        f = sensor(mesh)
        if interp:
            f = interpolate(f, FunctionSpace(mesh, "CG", 1))
        M = steady_metric(f, mesh=mesh, op=op, enforce_contraints=False)
        with pipes() as (out, err):
            mesh = adapt(mesh, M)

    # Plot mesh
    if plot_mesh:
        print("Number of elements = {:d}".format(mesh.num_cells()))
        print("Number of vertices = {:d}".format(mesh.num_vertices()))

        fig, axes = plt.subplots(figsize=(5, 5))
        triplot(mesh, axes=axes, interior_kw={'linewidth': 0.1}, boundary_kw={'color': 'k'})
        axes.axis(False)
        savefig(fname, os.path.join(fpath, 'outputs', op.approach), extensions=['png'])
        plt.close()

        fig, axes = plt.subplots(figsize=(6, 5))
        fig, axes = plot_quality(mesh, fig=fig, axes=axes)
        savefig(fname + '_quality', os.path.join(fpath, 'outputs', op.approach), extensions=['png'])
        plt.close()

        fig, axes = plt.subplots(figsize=(6, 5))
        fig, axes = plot_aspect_ratio(mesh, fig=fig, axes=axes, levels=np.linspace(1, 2, 6))
        savefig(fname + '_aspect_ratio', os.path.join(fpath, 'outputs', op.approach), extensions=['png'])
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
    parser.add_argument("-recovery", help="Choose from 'L2' and 'ZZ'.")
    parser.add_argument("-normalisation", help="Normalise by complexity or error.")
    parser.add_argument("-norm_order", help="Norm order for normalisation.")
    parser.add_argument("-target", help="Target complexity/error.")
    parser.add_argument("-num_adapt", help="Number of adaptations.")
    parser.add_argument("-interpolate", help="Toggle whether to interpolate sensor into P1 space.")
    args = parser.parse_args()
    f = {
        'bowl': bowl,
        'hyperbolic': hyperbolic,
        'multiscale': multiscale,
        'interweaved': interweaved
    }[args.sensor or 'bowl']
    p = None if args.norm_order in ('none', 'inf') else int(args.norm_order or 1)
    target = float(args.target or 100.0)
    max_adapt = int(args.num_adapt or 4)
    interp = args.recovery == 'ZZ' or bool(args.interpolate or False)
    kwargs = dict(target=target, max_adapt=max_adapt, interp=interp)
    test_sensors(f, args.recovery, args.normalisation or 'complexity', p, plot_mesh=True, **kwargs)
