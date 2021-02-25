from firedrake import *

import matplotlib.pyplot as plt
import math
import numpy as np
import os
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


def recover_gradient_sinusoidal(n, recovery, no_boundary, plot=False, norm_type='l2'):
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
    P0 = FunctionSpace(mesh, "DG", 0)
    h = interpolate(CellSize(mesh), P0)
    nodes = None
    if no_boundary:
        nodes = set(range(len(mesh.coordinates.dat.data)))
        nodes = list(nodes.difference(set(DirichletBC(P1, 0, 'on_boundary').nodes)))

    # P1 interpolant
    u_h = interpolate(func(x, y), P1)

    # Exact gradient interpolated into P1 space
    sigma = interpolate(gradient(x, y), P1_vec)
    if norm_type == 'l2':
        sigma_norm = l2_norm(sigma.dat.data, nodes=nodes)
    else:
        assert nodes is None
        sigma_norm = norm(sigma, norm_type=norm_type)

    # Recovered gradient
    sigma_rec = recover_gradient(u_h, op=op)
    if norm_type == 'l2':
        relative_error = l2_norm(sigma.dat.data - sigma_rec.dat.data, nodes=nodes)/sigma_norm
    else:
        relative_error = errornorm(sigma, sigma_rec, norm_type=norm_type)/sigma_norm

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
    return h.dat.data[0], relative_error


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


def test_field_linear(dim):
    r"""
    Given a simple linear field :math:`f = \sum_{i=1}^n x_i`
    projected into P0 space, check that the recovered
    P1 field matches the uninterpolated field.
    """
    if dim == 3:
        pytest.xfail("Needs tweaking")  # FIXME
    mesh = uniform_mesh(dim, 6)
    x = SpatialCoordinate(mesh)
    P0 = FunctionSpace(mesh, "DG", 0)
    P1 = FunctionSpace(mesh, "CG", 1)
    f = interpolate(sum(x), P0)
    exact = interpolate(sum(x), P1)
    f_zz = recover_zz(f, to_recover='field')
    assert np.isclose(lp_norm(f_zz.dat.data - exact.dat.data)/lp_norm(exact.dat.data), 0.0)


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
        if recovery == 'ZZ':
            axes.set_yticks([1e-30, 1e-25, 1e-20, 1e-15, 1e-10, 1e-5, 1])
        else:
            axes.set_yticks([1e-15, 1e-10, 1e-5, 1])
    axes.set_ylabel("Absolute error")
    savefig("hessian_errors_bowl_{:s}".format(recovery), "outputs/hessian", extensions=["pdf"])


def test_gradient_convergence(recovery, no_boundary, norm_type='l2'):
    r"""
    Check convergence rate of global L2 projection and Zienkiewicz-Zhu recovery
    under a given norm.

    On the domain interior, ZZ yields l2 ultraconvergence of :math:`\mathcal O(h^4)`.
    Since the patches break down on the boundary, inclusion of boundary nodes means
    that we can only hope for :math:`\mathcal O(h^2)` convergence. In fact, we are
    able to attain :math:`\mathcal O(h^{2.5})`. This is also attained for L2 projection.
    """
    cell_size = []
    relative_error = []
    istart, iend = 3, 7
    order = {
        'L2': {'l2': 2 if no_boundary else 2.5, 'L2': 1.5, 'L1': 2},
        'ZZ': {'l2': 4 if no_boundary else 2.5, 'L2': 1.5, 'L1': 2},
    }[recovery][norm_type]
    name = 'Zienkiewicz-Zhu' if recovery == 'ZZ' else 'L2 projection'
    for i in range(istart, iend):
        h, err = recover_gradient_sinusoidal(2**i, recovery, no_boundary, norm_type=norm_type)
        cell_size.append(h)
        relative_error.append(err)
        if i > istart:
            tol = 0.15 if i == istart + 1 else 0.1
            rate, expected = math.log(relative_error[-2]/relative_error[-1], 2), order
            msg = "{:s} {:s} convergence rate {:.2f} < {:.2f}"
            assert rate > (1 - tol)*order, msg.format(name, norm_type, rate, expected)
            print(msg[:-9].format(name, norm_type, rate))
    return cell_size, relative_error


def plot_gradient_convergence(cell_size, rel_error_zz, rel_error_l2, no_boundary, norm_type='l2'):
    """
    Plot convergence curves for both Zienkiewicz-Zhu and L2 projection recovery methods applied to
    the sinusoidal test case.
    """
    from mpltools import annotation

    # Plot convergence curves on a log-log axis
    fig, axes = plt.subplots(figsize=(5, 5))
    axes.loglog(cell_size, rel_error_zz, '--x', label='Zienkiewicz-Zhu')
    axes.loglog(cell_size, rel_error_l2, '--x', label=r'$\mathcal L_2$ projection')
    axes.set_xlabel("Element size")
    axes.set_xlim([0.01, 0.3])
    assert norm_type[0] in ('l', 'L'), "Norm type '{:s}' not recognised".format(norm_type)
    _norm_type = r"\ell" if norm_type[0] == 'l' else r"\mathcal L"
    _order = norm_type[1]
    label = r"Relative ${{{:s}}}_{{{:s}}}$ error".format(_norm_type, _order)
    axes.set_ylabel(label)
    axes.grid(True)

    # Add slope markers
    slope = 2.5 if norm_type[0] == 'l' else 2.0 if norm_type == 'L1' else 1.5
    y = 3.0e-02 if norm_type[0] == 'L' else 1.0e-04 if no_boundary else 7.0e-03
    annotation.slope_marker((0.1, y), slope, ax=axes, invert=not no_boundary, size_frac=0.15)
    if no_boundary:
        annotation.slope_marker((0.1, 8.0e-04), 4, invert=True, ax=axes, size_frac=0.15)

    # Save to file
    fname = 'gradient_recovery_convergence_{:s}'.format(norm_type)
    if no_boundary:
        fname += '_interior'
    plot_dir = 'outputs'
    savefig(fname, plot_dir, extensions=['pdf'])

    fname = 'legend_gradient_recovery'
    if not os.path.exists(os.path.join(plot_dir, fname + '.pdf')):
        fig2, axes2 = plt.subplots()
        lines, labels = axes.get_legend_handles_labels()
        legend = axes2.legend(lines, labels, fontsize=18, frameon=False, ncol=2)
        fig2.canvas.draw()
        axes2.set_axis_off()
        bbox = legend.get_window_extent().transformed(fig2.dpi_scale_trans.inverted())
        savefig(fname, plot_dir, bbox_inches=bbox, extensions=['pdf'], tight=False)


# ---------------------------
# plotting
# ---------------------------

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('recover', help="Choose 'gradient' or 'hessian'")
    parser.add_argument('-level', help="Mesh resolution level in each direction.")
    parser.add_argument('-convergence', help="Check convergence.")
    parser.add_argument('-no_boundary', help="Only compute error at interior nodes.")
    parser.add_argument('-norm_type', help="Choose from 'l2' or 'Lp', for any p >= 1.")
    args = parser.parse_args()
    no_bdy = bool(0 if args.no_boundary == "0" else args.no_boundary or False)
    norm_type = args.norm_type or 'l2'
    if args.recover == 'gradient':
        if bool(args.convergence or False):
            cell_size, zz = test_gradient_convergence('ZZ', no_bdy, norm_type=norm_type)
            cell_size, l2 = test_gradient_convergence('L2', no_bdy, norm_type=norm_type)
            plot_gradient_convergence(cell_size, zz, l2, no_bdy, norm_type=norm_type)
        else:
            recover_gradient_sinusoidal(2**int(args.level or 3), no_bdy, plot=True)
    else:
        test_hessian_bowl(2, True, 'L2', plot_mesh=True)
        # test_hessian_bowl(2, False, 'L2', plot_mesh=True)
        test_hessian_bowl(2, True, 'ZZ', plot_mesh=True)
