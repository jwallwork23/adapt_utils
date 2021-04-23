r"""
Test mesh movement in the steady state case for analytically defined monitor functions
from [Weller et al. 2016].

[Weller et al. 2016] H. Weller, P. Browne, C. Budd, and M. Cullen, Mesh adaptation on the
    sphere using optimal transport and the numerical solution of a Monge-Amp\ère type
    equation, J. Comput. Phys., 308 (2016), pp. 102--123,
    https://doi.org/10.1016/j.jcp.2015.12.018.
"""
from firedrake import *

import os
import pytest

from adapt_utils.adapt.r import MeshMover
from adapt_utils.options import Options
from adapt_utils.plotting import *


def ring(mesh=None, x=None):
    """
    An analytically defined monitor function which concentrates mesh density in
    a narrow ring within the unit square domain.
    """
    alpha = 10.0  # Controls amplitude of the ring
    beta = 200.0  # Controls width of the ring
    gamma = 0.15  # Controls radius of the ring
    return 1.0 + alpha*pow(cosh(beta*((x[0]-0.5)*(x[0]-0.5) + (x[1]-0.5)*(x[1]-0.5) - gamma)), -2)


def bell(mesh=None, x=None):
    """
    An analytically defined monitor function which concentrates mesh density in
    a bell region within the unit square domain.
    """
    alpha = 10.0  # Controls amplitude of the bell
    beta = 400.0  # Controls width of the bell
    return 1.0 + alpha*pow(cosh(beta*((x[0]-0.5)*(x[0]-0.5) + (x[1]-0.5)*(x[1]-0.5))), -2)


# ---------------------------
# standard tests for pytest
# ---------------------------

@pytest.fixture(params=[ring, bell])
def monitor(request):
    return request.param


@pytest.fixture(params=['relaxation', 'quasi_newton'])
def method(request):
    return request.param


def test_analytical(monitor, method, plot_mesh=False):
    """
    Check that the moved mesh matches that obtained previously.
    """
    fname = '_'.join([monitor.__name__, method])
    fpath = os.path.dirname(__file__)

    op = Options(approach='monge_ampere', r_adapt_rtol=1.0e-03, nonlinear_method=method)

    mesh = UnitSquareMesh(20, 20)
    orig_vol = assemble(Constant(1.0)*dx(domain=mesh))
    mm = MeshMover(mesh, monitor, op=op)
    mm.adapt()

    mesh.coordinates.assign(mm.x)
    vol = assemble(Constant(1.0)*dx(domain=mesh))
    assert np.allclose(orig_vol, vol), "Volume is not conserved!"

    if plot_mesh:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(figsize=(5, 5))
        triplot(mesh, axes=axes, interior_kw={'linewidth': 0.1}, boundary_kw={'color': 'k'})
        axes.axis(False)
        savefig(fname, os.path.join(fpath, 'outputs'), extensions=['png'])

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
    for m in [ring, bell]:
        test_analytical(m, 'quasi_newton', plot_mesh=True)
