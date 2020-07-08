r"""
Test mesh movement in the steady state case for analytically defined monitor functions.

The monitor functions are modified from those found in

H. Weller, P. Browne, C. Budd, and M. Cullen, Mesh adaptation on the sphere using op-
timal transport and the numerical solution of a Monge--Amp\ère type equation, J. Comput.
Phys., 308 (2016), pp. 102--123, https://doi.org/10.1016/j.jcp.2015.12.018.
"""
from firedrake import *

import pytest
import os
import numpy as np
import matplotlib.pyplot as plt

from adapt_utils.adapt.r import MeshMover
from adapt_utils.options import Options


def ring_monitor(mesh):
    """
    An analytically defined monitor function which concentrates mesh density in
    a narrow ring within the unit square domain.
    """
    x, y = SpatialCoordinate(mesh)
    alpha = 10.0  # Controls amplitude of the ring
    beta = 200.0  # Controls width of the ring
    gamma = 0.15  # Controls radius of the ring
    return 1.0 + alpha*pow(cosh(beta*((x-0.5)*(x-0.5) + (y-0.5)*(y-0.5) - gamma)), -2)


def bell_monitor(mesh):
    """
    An analytically defined monitor function which concentrates mesh density in
    a bell region within the unit square domain.
    """
    x, y = SpatialCoordinate(mesh)
    alpha = 10.0  # Controls amplitude of the bell
    beta = 400.0  # Controls width of the bell
    return 1.0 + alpha*pow(cosh(beta*((x-0.5)*(x-0.5) + (y-0.5)*(y-0.5))), -2)


monitors = {
    'ring': ring_monitor,
    'bell': bell_monitor,
}


# ---------------------------
# standard tests for pytest
# ---------------------------

@pytest.fixture(params=['ring', 'bell'])
def monitor_type(request):
    return request.param


@pytest.fixture(params=['relaxation', 'quasi_newton'])
def method(request):
    return request.param


def test_mesh_movement(monitor_type, method, plot_mesh=False):
    monitor = monitors[monitor_type]
    fname = '_'.join([monitor_type, method])
    fname = os.path.join(os.path.dirname(__file__), fname)

    op = Options(r_adapt_rtol=1.0e-03, nonlinear_method=method)

    mesh = UnitSquareMesh(20, 20)
    orig_vol = assemble(Constant(1.0)*dx(domain=mesh))
    mm = MeshMover(mesh, monitor, op=op)
    mm.adapt()

    mesh.coordinates.assign(mm.x)
    vol = assemble(Constant(1.0)*dx(domain=mesh))
    assert np.allclose(orig_vol, vol)

    if plot_mesh:
        fig, axes = plt.subplots()
        triplot(mesh, axes=axes)
        axes.axis('off')
        plt.tight_layout()
        plt.savefig(fname + '.png')

    if not os.path.exists(fname + '.npy'):
        np.save(fname, mm.x.dat.data)
        pytest.xfail("Needed to set up the test. Please try again.")
    loaded = np.load(fname + '.npy')
    assert np.allclose(mm.x.dat.data, loaded)


if __name__ == '__main__':
    for monitor in monitors:
        test_mesh_movement(monitor, 'quasi_newton', plot_mesh=True)
