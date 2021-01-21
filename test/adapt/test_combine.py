from firedrake import *

import matplotlib.pyplot as plt
import numpy as np
import os
import pytest
from wurlitzer import pipes

from adapt_utils.adapt.metric import *
from adapt_utils.adapt.r import MeshMover
from adapt_utils.options import Options
from adapt_utils.plotting import *


def uniform_mesh(dim, n):
    if dim == 2:
        return UnitSquareMesh(n, n)
    elif dim == 3:
        return UnitCubeMesh(n, n, n)
    else:
        raise ValueError("Dimension {:d} not supported".format(dim))


def metric_combine(metric1, metric2, mode):
    return {
        1: metric1,
        2: metric2,
        'avg': metric_average(metric1, metric2),
        'int': metric_intersection(metric1, metric2),
    }[mode]


def monitor_combine(monitor1, monitor2, mode):
    return {
        1: monitor1,
        2: monitor2,
        'avg': lambda mesh: 0.5*(monitor1(mesh) + monitor2(mesh)),
        'int': lambda mesh: max_value(monitor1(mesh), monitor2(mesh)),
    }[mode]


# ---------------------------
# standard tests for pytest
# ---------------------------

@pytest.fixture(params=[2, 3])
def dim(request):
    return request.param


def test_intersection(dim):
    """
    Check that metric intersection DTRT when
    applied to two isotropic metrics.
    """
    mesh = uniform_mesh(dim, 3)
    P1_ten = TensorFunctionSpace(mesh, "CG", 1)
    I = Identity(dim)
    M1 = interpolate(2*I, P1_ten)
    M2 = interpolate(1*I, P1_ten)
    M = metric_intersection(M1, M2)
    assert np.allclose(M.dat.data, M1.dat.data)
    M2.interpolate(2*I)
    M = metric_intersection(M1, M2)
    assert np.allclose(M.dat.data, M1.dat.data)
    assert np.allclose(M.dat.data, M2.dat.data)
    M2.interpolate(4*I)
    M = metric_intersection(M1, M2)
    assert np.allclose(M.dat.data, M2.dat.data)


def test_intersection_boundary(dim):
    """
    Check that metric intersection DTRT when
    applied to two isotropic metrics on the
    boundary alone.
    """
    mesh = uniform_mesh(dim, 3)
    P1_ten = TensorFunctionSpace(mesh, "CG", 1)
    I = Identity(dim)
    M1 = interpolate(2*I, P1_ten)
    M2 = interpolate(4*I, P1_ten)
    M = metric_intersection(M1, M2, boundary_tag='on_boundary')

    # Get underlying arrays
    M = M.dat.data
    M1 = M1.dat.data
    M2 = M2.dat.data

    # Check intersection did what we expected
    bnodes = DirichletBC(P1_ten, 0, 'on_boundary').nodes
    nodes = set(range(mesh.num_vertices()))
    inodes = np.array(list(nodes.difference(set(bnodes))))
    assert np.allclose(M[inodes], M1[inodes])
    assert np.allclose(M[bnodes], M2[bnodes])


def test_combine_metric(plot_mesh=False):
    """
    The complexity of the metric intersection
    should be greater than or equal to the
    complexity of the constituent metrics.

    The same can be said for the number of
    elements and vertices in the resulting
    mesh. We observe that the number of
    elements and vertices in the metric
    average is usually lower.
    """
    kwargs = {
        'approach': 'hessian',
        'max_adapt': 4,
        'normalisation': 'complexity',
        'norm_order': 1,
        'target': 200.0,
        'h_min': 1.0e-06,
        'h_max': 1.0,
    }
    op = Options(**kwargs)
    kwargs = {
        'op': op,
        'enforce_constraints': True,
    }

    # Loop over different metric construction modes
    data = {1: {}, 2: {}, 'avg': {}, 'int': {}}
    for mode in data:

        # Create domain [0, 1]²
        mesh = uniform_mesh(2, 100)
        for i in range(op.max_adapt):
            x, y = SpatialCoordinate(mesh)
            P1_ten = TensorFunctionSpace(mesh, "CG", 1)

            # Create a metric focused around the arc x² + y² = ½
            f = exp(-abs(0.5 - x**2 - y**2))
            M1 = steady_metric(f, V=P1_ten, **kwargs)

            # Create a metric focused around the arc (1 - x)² + y² = ½
            g = exp(-abs(0.5 - (1 - x)**2 - y**2))
            M2 = steady_metric(g, V=P1_ten, **kwargs)

            # Choose metric according to mode
            M = metric_combine(M1, M2, mode)

            # Adapt mesh
            with pipes() as (out, err):
                mesh = adapt(mesh, M)

        # Store entity counts
        data[mode]['complexity'] = metric_complexity(M)
        data[mode]['num_cells'] = mesh.num_cells()
        data[mode]['num_vertices'] = mesh.num_vertices()

        # Plot
        if plot_mesh:
            fig, axes = plt.subplots(figsize=(5, 5))
            triplot(mesh, axes=axes, interior_kw={'linewidth': 0.1}, boundary_kw={'color': 'k'})
            axes.axis(False)
            savefig('mesh_{:}'.format(mode), 'outputs/hessian', extensions=['png'])

    # Check complexity follows C(M1) ≤ C(M1 ∩ M2) and similarly for M2
    msg = "M_{:d} does not have {:s} {:s} than M_{:s}"
    i = 'complexity'
    for j in (1, 2):
        assert data[j][i] <= data['int'][i], msg.format(j, 'fewer', i, 'int')

    # Similarly for element counts, also with lower bound
    for i in ('num_cells', 'num_vertices'):
        for j in (1, 2):
            assert data['avg'][i] <= data[j][i], msg.format(j, 'more', i, 'avg')
            assert data[j][i] <= data['int'][i], msg.format(j, 'fewer', i, 'int')

    # Print mesh data to screen
    if plot_mesh:
        for mode in data:
            print("Mode M_{:}".format(mode))
            print("number of elements = {:d}".format(data[mode]['num_cells']))
            print("number of vertices = {:d}".format(data[mode]['num_vertices']))


def test_combine_monitor(plot_mesh=False):
    """
    For monitor functions based on the sensors
    above, compare the meshes resulting from
    averaging and intersection (visually).

    In terms of testing, we just check for
    convergence and that volume is conserved.
    """
    alpha = Constant(10.0)  # Controls magnitude of arc
    beta = Constant(10.0)   # Controls width of arc

    def monitor1(mesh):
        x, y = SpatialCoordinate(mesh)
        return 1.0 + alpha*exp(-beta*abs(0.5 - x**2 - y**2))

    def monitor2(mesh):
        x, y = SpatialCoordinate(mesh)
        return 1.0 + alpha*exp(-beta*abs(0.5 - (1 - x)**2 - y**2))

    # Set parameters
    kwargs = {
        'approach': 'monge_ampere',
        'r_adapt_rtol': 1.0e-08,
        'nonlinear_method': 'relaxation',
    }
    op = Options(**kwargs)
    fpath = os.path.dirname(__file__)

    for mode in [1, 2, 'avg', 'int']:
        fname = {1: 'monitor1', 2: 'monitor2', 'avg': 'average', 'int': 'intersection'}[mode]
        monitor = monitor_combine(monitor1, monitor2, mode)

        # Create domain
        mesh = uniform_mesh(2, 100)
        orig_vol = assemble(Constant(1.0)*dx(domain=mesh))

        # Move the mesh and check for volume conservation
        mm = MeshMover(mesh, monitor, op=op)
        mm.adapt()
        mesh.coordinates.assign(mm.x)
        vol = assemble(Constant(1.0)*dx(domain=mesh))
        assert np.isclose(orig_vol, vol), "Volume is not conserved!"

        h = interpolate(CellSize(mesh), mm.P0).vector().gather()
        print("Combination method '{:s}':".format(fname))
        print("Minimum cell size = {:.4e}".format(h.min()))
        print("Maximum cell size = {:.4e}".format(h.max()))
        print("Mean cell size    = {:.4e}".format(np.mean(h)))

        # Plot mesh
        if plot_mesh:
            fig, axes = plt.subplots(figsize=(5, 5))
            triplot(mesh, axes=axes, interior_kw={'linewidth': 0.1}, boundary_kw={'color': 'k'})
            axes.axis(False)
            savefig(fname, os.path.join(fpath, 'outputs', op.approach), extensions=['png'])
            plt.close()


# ---------------------------
# mesh plotting
# ---------------------------

if __name__ == "__main__":
    # test_combine_metric(plot_mesh=True)
    test_combine_monitor(plot_mesh=True)
