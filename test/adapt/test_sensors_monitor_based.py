r"""
Test mesh movement and metric based adaptation in the steady state case for analytically defined
sensor functions defined in [Olivier 2011].

[Olivier 2011] Olivier, Géraldine. Anisotropic metric-based mesh adaptation for unsteady CFD
    simulations involving moving geometries. Diss. 2011.
"""
from firedrake import *

import pytest
import os
import numpy as np
import matplotlib.pyplot as plt

from adapt_utils.adapt.r import MeshMover
from adapt_utils.adapt.recovery import recover_hessian
from adapt_utils.norms import local_frobenius_norm
from adapt_utils.options import Options
from adapt_utils.plotting import *
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


@pytest.fixture(params=['quasi_newton', 'relaxation'])
def method(request):
    return request.param


def test_mesh_movement(sensor, monitor_type, method, plot_mesh=False):
    alpha = Constant(1.0)

    def shift_and_scale(mesh):
        """
        Adapt to the scaled and shifted sensor magnitude.
        """
        alpha.assign(5.0)
        return 1.0 + alpha*abs(sensor(mesh))

    def frobenius(mesh):
        """
        Adapt to the Frobenius norm of the sensor Hessian.
        """
        alpha.assign(0.1)
        P1 = FunctionSpace(mesh, "CG", 1)
        f = sensor(mesh)
        # f = interpolate(f, P1)
        H = recover_hessian(f, mesh=mesh, op=op)
        return 1.0 + alpha*local_frobenius_norm(H, space=P1)

    if monitor_type == 'shift_and_scale':
        monitor = shift_and_scale
    elif monitor_type == 'frobenius':
        monitor = frobenius
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
    fname = '_'.join([sensor.__name__, monitor_type, method])
    fpath = os.path.dirname(__file__)

    # Create domain
    n = 100
    mesh = SquareMesh(n, n, 2)
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
        fig, axes = plt.subplots(figsize=(5, 5))
        triplot(mesh, axes=axes, interior_kw={'linewidth': 0.1}, boundary_kw={'color': 'k'})
        axes.axis(False)
        savefig(fname, os.path.join(fpath, 'outputs', op.approach), extensions=['png'])
        plt.close()
        return

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
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-sensor", help="""
        Choice of sensor function, from {'bowl', 'hyperbolic', 'multiscale', 'interweaved'}.
        """)
    parser.add_argument("-monitor", help="Monitor function.")
    parser.add_argument("-nonlinear_method", help="Nonlinear solver method.")
    # parser.add_argument("-interpolate", help="Toggle whether to interpolate sensor into P1 space.")
    args = parser.parse_args()
    f = {
        'bowl': bowl,
        'hyperbolic': hyperbolic,
        'multiscale': multiscale,
        'interweaved': interweaved
    }[args.sensor or 'bowl']
    # interp = bool(args.interpolate or False)
    m = args.monitor or 'frobenius'
    test_mesh_movement(f, m, args.nonlinear_method or 'quasi_newton', plot_mesh=True)
