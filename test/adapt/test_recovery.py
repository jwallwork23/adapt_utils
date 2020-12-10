from firedrake import *

import matplotlib.pyplot as plt
import numpy as np
import pytest

from adapt_utils.adapt.recovery import *
from adapt_utils.mesh import make_consistent
from adapt_utils.norms import lp_norm
from adapt_utils.options import Options
from adapt_utils.plotting import *
from adapt_utils.swe.utils import recover_vorticity


def uniform_mesh(dim, n, l=1):
    if dim == 2:
        return SquareMesh(n, n, l)
    elif dim == 3:
        return CubeMesh(n, n, n, l)
    else:
        raise ValueError("Dimension {:d} not supported".format(dim))


def l2_norm(array, nodes=None):
    """
    Compute the l2-norm over some set of nodes.
    """
    if nodes is not None:
        array = array[nodes]
    array = [np.dot(ai, ai) for ai in array]
    return lp_norm(array)


def recover_gradient_sinusoidal(n, recovery, no_boundary, plot=False):
    """
    Apply Zienkiewicz-Zhu and global L2 projection to recover
    the gradient of a sinusoidal function.
    """
    op = Options(gradient_recovery=recovery)
    kwargs = dict(levels=np.linspace(0, 6.4, 50), cmap='coolwarm')

    # Function of interest and its exact gradient
    func = lambda xx, yy: sin(pi*xx)*sin(pi*yy)
    gradient = lambda xx, yy: as_vector([pi*cos(pi*xx)*sin(pi*y),
                                         pi*sin(pi*xx)*cos(pi*yy)])
    # hessian = lambda xx, yy: as_matrix([[-pi*pi*func(xx, yy), pi*pi*cos(pi*xx)*cos(pi*yy)],
    #                                     [pi*pi*cos(pi*xx)*cos(pi*yy), -pi*pi*func(xx, yy)]])

    # Domain
    mesh = uniform_mesh(2, n)
    x, y = SpatialCoordinate(mesh)
    plex, offset, coordinates = make_consistent(mesh)

    # Spaces
    P1_vec = VectorFunctionSpace(mesh, "CG", 1)
    P1 = FunctionSpace(mesh, "CG", 1)
    nodes = None
    if no_boundary:
        nodes = set(range(len(mesh.coordinates.dat.data)))
        nodes = list(nodes.difference(set(DirichletBC(P1, 0, 'on_boundary').nodes)))

    # P1 interpolant
    u_h = interpolate(func(x, y), P1)

    # Exact gradient interpolated into P1 space
    sigma = interpolate(gradient(x, y), P1_vec)
    sigma_l2 = l2_norm(sigma.dat.data, nodes=nodes)

    # Recovered gradient
    sigma_rec = recover_gradient(u_h, op=op)
    relative_error = l2_norm(sigma.dat.data - sigma_rec.dat.data, nodes=nodes)/sigma_l2

    # Plotting
    if plot:
        fig, axes = plt.subplots(ncols=2, figsize=(11, 5))
        axes[0].set_title("Exact")
        fig.colorbar(tricontourf(sigma, axes=axes[0], **kwargs), ax=axes[0])
        triplot(mesh, axes=axes[0])
        axes[0].axis(False)
        axes[0].set_xlim([-0.1, 1.1])
        axes[0].set_ylim([-0.1, 1.1])
        axes[1].set_title("Global L2 projection" if method == 'L2' else "Zienkiewicz-Zhu")
        fig.colorbar(tricontourf(sigma_rec, axes=axes[1], **kwargs), ax=axes[1])
        triplot(mesh, axes=axes[1])
        axes[1].axis(False)
        axes[1].set_xlim([-0.1, 1.1])
        axes[1].set_ylim([-0.1, 1.1])
        plt.show()
    return relative_error


# ---------------------------
# standard tests for pytest
# ---------------------------

@pytest.fixture(params=[2, 3])
def dim(request):
    return request.param


@pytest.fixture(params=[True, False])
def interp(request):
    return request.param


@pytest.fixture(params=['L2', 'ZZ'])
def recovery(request):
    return request.param


@pytest.fixture(params=[True, False])
def no_boundary(request):
    return request.param


def test_gradient_linear(dim, recovery):
    r"""
    Given a simple linear field :math:`f = x`, we
    check that the recovered gradient matches the
    analytical gradient, :math:`\nabla f = (1, 0)`.
    """
    op = Options(gradient_recovery=recovery)

    # Define test function
    mesh = uniform_mesh(dim, 3)
    x = SpatialCoordinate(mesh)
    P1 = FunctionSpace(mesh, "CG", 1)
    f = interpolate(x[0], P1)

    # Recover gradient using specified method
    g = recover_gradient(f, op=op)

    # Compare with analytical solution
    dfdx = np.zeros(dim)
    dfdx[0] = 1
    analytical = [dfdx for i in g.dat.data]
    assert np.allclose(g.dat.data, analytical)


def test_vorticity_anticlockwise():
    r"""
    Given a velocity field :math:`\mathbf u = (u,v)`,
    the curl is defined by

  ..math::
        \mathrm{curl}(\mathbf u) :=
            \frac{\partial v}{\partial x}
            - \frac{\partial u}{\partial y}.

    For a simple velocity field,

  ..math::
        \mathbf u = 0.5* (-y, x),

    the vorticity should be unity everywhere.
    """
    mesh = uniform_mesh(2, 3)
    x, y = SpatialCoordinate(mesh)
    P1_vec = VectorFunctionSpace(mesh, "CG", 1)

    # Define test function
    u, v = -0.5*y, 0.5*x
    uv = interpolate(as_vector([u, v]), P1_vec)

    # Recover vorticity using L2 projection
    zeta = recover_vorticity(uv)

    # Compare with analytical solution
    dudy, dvdx = -0.5, 0.5
    curl_uv = dvdx - dudy
    analytical = [curl_uv for i in zeta.dat.data]
    assert np.allclose(zeta.dat.data, analytical)


def test_hessian_bowl(dim, interp, recovery, plot_mesh=False):
    r"""
    Check that the recovered Hessian of the quadratic
    function

  ..math::
        u(\mathbf x) = \frac12(\mathbf x \cdot \mathbf x)

    is the identity matrix.

    :arg interp: toggle whether or not to interpolate
        into :math:`\mathbb P1` space before recovering
        the Hessian.
    """
    if not interp and recovery == 'ZZ':
        pytest.xfail("Zienkiewicz-Zhu requires a Function, rather than an expression.")
    op = Options(hessian_recovery=recovery)
    n = 100 if dim == 2 else 40
    if dim == 3 and interp:
        pytest.xfail("We cannot expect recovery to be good here.")
    mesh = uniform_mesh(dim, n, 2)
    x = SpatialCoordinate(mesh)
    mesh.coordinates.interpolate(as_vector([xi - 1 for xi in x]))

    # Construct Hessian
    f = 0.5*dot(x, x)
    if interp:
        f = interpolate(f, FunctionSpace(mesh, "CG", 1))
    H = recover_hessian(f, mesh=mesh, op=op)

    # Construct analytical solution
    I = interpolate(Identity(dim), H.function_space())

    # Check correspondence
    tol = 1.0e-5 if dim == 3 else 0.8 if interp else 1.0e-6
    assert np.allclose(H.dat.data, I.dat.data, atol=tol)
    if not plot_mesh:
        return
    if dim != 2:
        raise ValueError("Cannot plot in {:d} dimensions".format(dim))

    # Plot errors on scatterplot
    H_arr = np.abs(H.dat.data - I.dat.data)
    ones = np.ones(len(H_arr))
    fig, axes = plt.subplots(figsize=(5, 6))
    for i in range(2):
        for j in range(2):
            axes.scatter((2*i + j + 1)*ones, H_arr[:, i, j], marker='x')
    axes.set_yscale('log')
    axes.set_xlim([0, 5])
    axes.set_xticks([1, 2, 3, 4])
    axes.set_xticklabels(["(0,0)", "(0,1)", "(1,0)", "(1,1)"])
    axes.set_xlabel("Hessian component")
    if interp:
        axes.set_yticks([1e-15, 1e-10, 1e-5, 1])
    axes.set_ylabel("Absolute error")
    savefig("hessian_errors_bowl_{:s}".format(recovery), "outputs/hessian", extensions=["pdf"])


def test_gradient_convergence(recovery, no_boundary):
    r"""
    Check l2 convergence rate of global L2 projection and Zienkiewicz-Zhu recovery.

    On the domain interior, ZZ yields ultraconvergence of :math:`\mathcal O(h^4)`.
    Since the patches break down on the boundary, inclusion of boundary nodes means
    that we can only hope for :math:`\mathcal O(h^2)` convergence. This latter rate
    is also expected for L2 projection.
    """
    relative_error = []
    istart, iend = 3, 7
    tol = 0.1
    order = 2 if recovery == 'L2' else 4 if no_boundary else 2
    name = 'Zienkiewicz-Zhu' if recovery == 'ZZ' else 'L2 projection'
    for i in range(istart, iend):
        relative_error.append(recover_gradient_sinusoidal(2**i, recovery, no_boundary))
        if i > istart:
            rate, expected = relative_error[-2]/relative_error[-1], 2**order
            msg = "{:s} convergence rate {:.2f} < {:.2f}"
            assert rate > (2 - tol)**order, msg.format(name, rate, expected)
    return relative_error


def plot_gradient_convergence(relative_error_zz, relative_error_l2, no_boundary):
    """
    Plot convergence curves for both Zienkiewicz-Zhu and L2 projection recovery methods applied to
    the sinusoidal test case.
    """
    fig, axes = plt.subplots(figsize=(5, 5))
    elements = [2**(2*i+1) for i in range(istart, iend)]
    axes.loglog(elements, relative_error_zz, '--x', label=r'$Z^2$')
    axes.loglog(elements, relative_error_l2, '--x', label=r'$\mathcal L_2$')
    axes.set_xlabel("Element count")
    axes.set_xticks([100, 1000, 10000])
    axes.set_ylabel("Relative error")
    axes.legend()
    axes.grid(True)
    fname = 'gradient_recovery_convergence'
    if no_boundary:
        fname += '_interior'
    savefig(fname, 'outputs', extensions=['pdf'])


# ---------------------------
# plotting
# ---------------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-level', help="Mesh resolution level in each direction.")
    parser.add_argument('-convergence', help="Check convergence.")
    parser.add_argument('-no_boundary', help="Only compute l2 error at interior nodes.")
    parser.add_argument('-plot', help="Toggle plotting.")
    args = parser.parse_args()
    no_bdy = bool(0 if args.no_boundary == "0" else args.no_boundary or False)
    if bool(args.convergence or False):
        zz = test_gradient_convergence('ZZ', no_bdy)
        l2 = test_gradient_convergence('L2', no_bdy)
        plot_gradient_convergence(zz, l2, no_bdy)
    else:
        recover_gradient_sinusoidal(2**int(args.level or 3), no_bdy, plot=bool(args.plot or False))
    # test_hessian_bowl(2, True, 'L2', plot_mesh=True)
    # test_hessian_bowl(2, False, 'L2', plot_mesh=True)
    # test_hessian_bowl(2, True, 'ZZ', plot_mesh=True)
