from firedrake import *

import matplotlib.pyplot as plt
import numpy as np
import pytest

from adapt_utils.adapt.recovery import *
from adapt_utils.mesh import get_patch, make_consistent
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


def recover_gradient(n, plot=False):
    """
    Apply Zienkiewicz-Zhu recovery for the gradient of a sinusoidal function.
    """
    plotted = {vertex_type: not plot for vertex_type in ('interior', 'boundary', 'corner')}
    kwargs = dict(levels=np.linspace(0, 6.4, 50), cmap='coolwarm')

    # Function of interest and its exact gradient
    k = 2*pi
    func = lambda xx, yy: sin(k*xx)*sin(k*yy)
    gradient = lambda xx, yy: as_vector([k*cos(k*xx)*sin(k*y), k*sin(k*xx)*cos(k*yy)])
    vandermonde = lambda xx, yy: np.array([1.0, xx, yy])

    # Domain
    mesh = uniform_mesh(2, n)
    x, y = SpatialCoordinate(mesh)
    plex, offset, coordinates = make_consistent(mesh)

    # Spaces
    P1_vec = VectorFunctionSpace(mesh, "CG", 1)
    P0_vec = VectorFunctionSpace(mesh, "DG", 0)
    P1 = FunctionSpace(mesh, "CG", 1)

    # P1 interpolant
    u_h = Function(P1)
    u_h.interpolate(func(x, y))

    # Exact gradient interpolated into P1 space
    sigma = Function(P1_vec)
    sigma.interpolate(gradient(x, y))

    # Direct differentiation
    sigma_h = interpolate(grad(u_h), P0_vec)

    # Zienkiewicz-Zhu
    sigma_ZZ = Function(P1_vec)
    if plot:
        fig, axes = plt.subplots(ncols=3, figsize=(16, 5))
    for vvv in range(*plex.getDepthStratum(0)):
        patch = get_patch(vvv, plex=plex, coordinates=coordinates)
        elements = set(patch['elements'].keys())
        orig_elements = set(patch['elements'].keys())

        # Classify vertex
        vertex_type = None
        if len(elements) == 6:
            vertex_type = 'interior'
        elif len(elements) == 3:
            vertex_type = 'boundary'
        elif len(elements) in (1, 2):
            vertex_type = 'corner'
        else:
            raise ValueError

        # Extend patch for boundary cases
        if len(elements) == 1:
            for v in patch['vertices']:
                if len(plex.getSupport(v)) == 4:
                    patch = get_patch(v, plex=plex, coordinates=coordinates, extend=elements)
                    break
            elements = set(patch['elements'].keys())
        if len(elements) != 6:
            for v in patch['vertices']:
                if len(plex.getSupport(v)) == 6:
                    patch = get_patch(v, plex=plex, coordinates=coordinates, extend=elements)
                    break
            elements = set(patch['elements'].keys())

        # Plot one example of each patch type
        if not plotted[vertex_type]:
            ax = axes[{'interior': 0, 'boundary': 1, 'corner': 2}[vertex_type]]
            ax.set_title(vertex_type.capitalize())
            triplot(mesh, axes=ax)
            for v in patch['vertices']:
                colour = 'C4' if v == vvv else 'C2'
                marker = 'o' if v == vvv else 'x'
                ax.plot(*coordinates(v), marker, color=colour)
            for k in elements:
                colour = 'C1' if k in orig_elements else 'C5'
                ax.plot(*patch['elements'][k]['centroid'], '^', color=colour)
            ax.axis(False)
            plotted[vertex_type] = True

        # Assemble local system
        A = np.zeros((3, 3))
        b = np.zeros((3, 2))
        for k in elements:
            c = patch['elements'][k]['centroid']
            P = vandermonde(*c)
            A += np.tensordot(P, P, axes=0)
            b += np.tensordot(P, sigma_h.at(c), axes=0)

        # Solve local system
        a = np.linalg.solve(A, b)
        sigma_ZZ.dat.data[offset(vvv)] = np.dot(vandermonde(*coordinates(vvv)), a)
    relative_error_sigma_ZZ = errornorm(sigma, sigma_ZZ)/norm(sigma)

    # Global L2 projection
    p1trial = TrialFunction(P1_vec)
    p1test = TestFunction(P1_vec)
    sigma_L = Function(P1_vec)
    a = inner(p1test, p1trial)*dx
    L = inner(p1test, sigma_h)*dx
    solve(a == L, sigma_L, solver_parameters={'ksp_type': 'cg'})
    relative_error_sigma_L = errornorm(sigma, sigma_L)/norm(sigma)

    # Plotting
    if plot:
        plt.show()

        # Plot exact gradient
        fig, axes = plt.subplots(ncols=3, figsize=(16, 5))
        axes[0].set_title("Exact")
        fig.colorbar(tricontourf(sigma, axes=axes[0], **kwargs), ax=axes[0])
        triplot(mesh, axes=axes[0])
        axes[0].axis(False)
        axes[0].set_xlim([-0.1, 1.1])
        axes[0].set_ylim([-0.1, 1.1])

        # Plot L2 projected gradient
        axes[1].set_title("Global L2 projection")
        fig.colorbar(tricontourf(sigma_L, axes=axes[1], **kwargs), ax=axes[1])
        triplot(mesh, axes=axes[1])
        axes[1].axis(False)
        axes[1].set_xlim([-0.1, 1.1])
        axes[1].set_ylim([-0.1, 1.1])

        # Plot ZZ recovered gradient
        axes[2].set_title("Zienkiewicz-Zhu")
        fig.colorbar(tricontourf(sigma_ZZ, axes=axes[2], **kwargs), ax=axes[2])
        triplot(mesh, axes=axes[2], boundary_kw={'color': 'k'})
        axes[2].axis(False)
        axes[2].set_xlim([-0.1, 1.1])
        axes[2].set_ylim([-0.1, 1.1])
        plt.show()
    return relative_error_sigma_ZZ, relative_error_sigma_L


# ---------------------------
# standard tests for pytest
# ---------------------------

@pytest.fixture(params=[2, 3])
def dim(request):
    return request.param


@pytest.fixture(params=[True, False])
def interp(request):
    return request.param


# @pytest.fixture(params=['parts', 'dL2'])
@pytest.fixture(params=['dL2'])
def hessian_recovery(request):
    return request.param


def test_gradient_linear(dim):
    r"""
    Given a simple linear field :math:`f = x`, we
    check that the recovered gradient matches the
    analytical gradient, :math:`\nabla f = (1, 0)`.
    """
    mesh = uniform_mesh(dim, 3)
    x = SpatialCoordinate(mesh)
    P1 = FunctionSpace(mesh, "CG", 1)

    # Define test function
    f = interpolate(x[0], P1)

    # Recover gradient using L2 projection
    g = recover_gradient(f)

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


def test_hessian_bowl(dim, interp, hessian_recovery, plot_mesh=False):
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
    op = Options(hessian_recovery=hessian_recovery)
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
    savefig("hessian_errors_bowl", "outputs/hessian", extensions=["pdf"])


def test_gradient_recovery(plot=False):
    """
    Check convergence rate of global L2 projection and Zienkiewicz-Zhu recovery.
    """
    relative_error_zz = []
    relative_error_l2 = []
    istart, iend = 2, 7
    for i in range(istart, iend):
        errors = recover_gradient(2**i)
        relative_error_zz.append(errors[0])
        relative_error_l2.append(errors[1])
        if i > istart:
            rate, expected = relative_error_zz[-2]/relative_error_zz[-1], 3
            msg = "Zienkiewicz-Zhu convergence rate {:.2f} < {:.2f}"
            assert rate > expected, msg.format(rate, expected)
            rate, expected = relative_error_l2[-2]/relative_error_l2[-1], 2
            assert rate > expected, msg.format(rate, expected)
            msg = "L2 projection convergence rate {:.2f} < {:.2f}"
    if plot:
        fig, axes = plt.subplots(figsize=(5, 5))
        elements = [2**(2*i+1) for i in range(istart, iend)]
        axes.loglog(elements, relative_error_zz, '--x', label='ZZ')
        axes.loglog(elements, relative_error_l2, '--x', label='L2')
        axes.set_xlabel("Element count")
        axes.set_ylabel("Relative error")
        axes.set_yticks([0.01, 0.1, 1])
        axes.set_yticklabels([r"{{{:.0f}}}\%".format(100*e) for e in axes.get_yticks()])
        axes.legend()
        axes.grid(True)
        savefig('gradient_recovery_convergence', 'outputs', extensions=['pdf'])


# ---------------------------
# plotting
# ---------------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-level', help="Mesh resolution level in each direction.")
    parser.add_argument('-convergence', help="Check convergence.")
    parser.add_argument('-plot', help="Toggle plotting.")
    args = parser.parse_args()
    if bool(args.convergence or False):
        test_gradient_recovery(plot=True)
    else:
        recover_gradient(2**int(args.level or 3), bool(args.plot or False))
    # test_hessian_bowl(2, True, 'dL2', plot_mesh=True)
    # test_hessian_bowl(2, False, 'dL2', plot_mesh=True)
