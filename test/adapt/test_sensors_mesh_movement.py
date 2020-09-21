r"""
Test mesh movement and metric based adaptation in the steady state case for analytically defined
sensor functions.

Sensors as defined in

Olivier, Géraldine. Anisotropic metric-based mesh adaptation for unsteady CFD simulations involving
moving geometries. Diss. 2011.
"""
import pytest
import os
import numpy as np
import matplotlib.pyplot as plt

from adapt_utils import *
from adapt_utils.test.adapt.sensors import *


# ---------------------------
# standard tests for pytest
# ---------------------------

@pytest.fixture(params=[bowl, hyperbolic, multiscale, interweaved])
def sensor(request):
    return request.param


@pytest.fixture(params=['shift_and_scale', 'frobenius'])
def monitor_type(request):
    return request.param


@pytest.fixture(params=['quasi_newton'])
def method(request):
    return request.param


def test_mesh_movement(sensor, monitor_type, method, plot_mesh=False):

    def shift_and_scale(mesh):
        alpha = 5.0
        return 1.0 + alpha*abs(sensor(mesh))

    def frobenius(mesh):
        alpha = 0.01
        P1 = FunctionSpace(mesh, "CG", 1)
        H = recover_hessian(interpolate(sensor(mesh), P1), op=op)
        return 1.0 + alpha*local_frobenius_norm(H, space=P1)

    if monitor_type == 'shift_and_scale':
        monitor = shift_and_scale
    elif monitor_type == 'frobenius':
        monitor = frobenius
        if sensor in (multiscale, interweaved):
            pytest.xfail("This particular case needs tweaking")  # FIXME
    else:
        raise ValueError("Monitor function type {:s} not recognised.".format(monitor_type))

    # Set parameters
    kwargs = {
        'approach': 'monge_ampere',
        'r_adapt_rtol': 1.0e-03,
        'nonlinear_method': method,
        'debug': plot_mesh,
    }
    op = Options(**kwargs)
    fname = '_'.join([sensor.__name__, monitor_type])
    fpath = os.path.dirname(__file__)

    # Create domain
    n = 100
    mesh = SquareMesh(n, n, 2, 2)
    x, y = SpatialCoordinate(mesh)
    mesh.coordinates.interpolate(as_vector([x-1, y-1]))
    orig_vol = assemble(Constant(1.0)*dx(domain=mesh))

    # Move the mesh and check for volume conservation
    mm = MeshMover(mesh, monitor, op=op)
    mm.adapt()
    mesh.coordinates.assign(mm.x)
    vol = assemble(Constant(1.0)*dx(domain=mesh))
    assert np.allclose(orig_vol, vol), "Volume is not conserved!"

    # Plot mesh
    if plot_mesh:
        fig, axes = plt.subplots()
        triplot(mesh, axes=axes)
        axes.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(fpath, 'outputs', op.approach, fname + '.png'))
        plt.close()

    # Save mesh coordinates to file
    if not os.path.exists(os.path.join(fpath, 'data', fname + '.npy')):
        np.save(os.path.join(fpath, 'data', fname), mm.x.dat.data)
        if not plot_mesh:
            pytest.xfail("Needed to set up the test. Please try again.")
    loaded = np.load(os.path.join(fpath, 'data', fname + '.npy'))
    assert np.allclose(mm.x.dat.data, loaded), "Mesh does not match data"


# ---------------------------
# mesh plotting
# ---------------------------

if __name__ == '__main__':
    nonlinear_method = 'quasi_newton'
    for f in (bowl, hyperbolic, multiscale, interweaved):
        for m in ('shift_and_scale', 'frobenius'):
            print("Running with montitor '{:s}' and sensor '{:s}'".format(m, sensor.__name__))
            test_mesh_movement(f, m, nonlinear_method, plot_mesh=True)
