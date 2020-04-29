from firedrake import *

import numpy as np

from adapt_utils.options import Options
from adapt_utils.adapt.recovery import construct_hessian, construct_boundary_hessian
from adapt_utils.adapt.kernels import *


__all__ = ["steady_metric", "isotropic_metric", "metric_with_boundary", "metric_intersection",
           "metric_relaxation", "metric_complexity", "combine_metrics", "time_normalise",
           "metric_average"]


def steady_metric(f=None, H=None, projector=None, normalise=True, op=Options()):
    r"""
    Computes the steady metric for mesh adaptation. Based on Nicolas Barral's function
    ``computeSteadyMetric``, from ``adapt.py``, 2016.

    Clearly at least one of `f` and `H` must not be provided.

    :kwarg f: Field to compute the Hessian of.
    :kwarg H: Reconstructed Hessian associated with `f` (if already computed).
    :kwarg projector: :class:`DoubleL2Projector` object to compute Hessian.
    :kwarg normalise: If `False` then we simply take the diagonal matrix with `f` in modulus.
    :kwarg op: `Options` class object providing min/max cell size values.
    :return: Steady metric associated with Hessian `H`.
    """
    if f is None:
        try:
            assert H is not None
        except AssertionError:
            raise ValueError("Please supply either field for recovery, or Hessian thereof.")
    elif H is None:
        H = construct_hessian(f, op=op) if projector is None else projector.project(f)
    V = H.function_space()
    mesh = V.mesh()
    dim = mesh.topological_dimension()
    assert dim in (2, 3)  # TODO: test 3d case works
    assert op.normalisation in ('complexity', 'error')

    # Functions to hold metric and its determinant
    M = Function(V).assign(0.0)

    # Turn Hessian into a metric
    op.print_debug("METRIC: Constructing metric from Hessian...")
    kernel = eigen_kernel(metric_from_hessian, dim)
    op2.par_loop(kernel, V.node_set, M.dat(op2.RW), H.dat(op2.READ))
    op.print_debug("METRIC: Done!")

    if not normalise:
        return M

    # Apply Lp normalisation
    op.print_debug("METRIC: Normalising metric in space...")
    space_normalise(M, op=op)
    op.print_debug("METRIC: Done!")

    # Enforce maximum/minimum element sizes and anisotropy
    op.print_debug("METRIC: Enforcing elemental constraints...")
    enforce_element_constraints(M, op=op)
    op.print_debug("METRIC: Done!")

    return M


def space_normalise(M, op=Options()):
    """
    Normalise steady metric `M`.

    NOTE: In the case of 'error' normalisation, 'target' is the inverse of error.
    """
    mesh = M.function_space().mesh()
    d = mesh.topological_dimension()
    p = op.norm_order

    # Infinite p
    if p is None:
        if op.normalisation == 'complexity':
            integral = assemble(sqrt(det(M))*dx)
            M *= pow(op.target/integral, 2/d)
        else:
            M *= d*op.target
        return

    # Finite p
    integral = assemble(pow(det(M), p/(2*p + d))*dx)
    assert integral > 1.0e-08
    if op.normalisation == 'complexity':
        integral = pow(op.target/integral, 2/d)
    else:
        integral = d*op.target*pow(integral, 1/p)
    P1 = FunctionSpace(mesh, "CG", 1)
    M.interpolate(integral*pow(det(M), -1/(2*p + d))*M)
    return


def enforce_element_constraints(M, op=Options()):
    """
    Post-process a metric `M` so that it obeys the maximum and minimum elemental size constraints
    and maximal anisotropy specified by :class:`Options` class `op`.
    """
    dim = M.function_space().mesh().topological_dimension()
    kernel = eigen_kernel(postproc_metric, dim, op=op)
    op2.par_loop(kernel, M.function_space().node_set, M.dat(op2.RW))


def isotropic_metric(f, noscale=False, normalise=True, op=Options()):
    r"""
    Given a scalar error indicator field `f`, construct an associated isotropic metric field.

    :arg f: Function to adapt to.
    :kwarg noscale: Toggle scaling by `target` complexity.
    :kwarg normalise: If `False` then we simply take the diagonal matrix with `f` in modulus.
    :kwarg op: :class:`Options` object providing min/max cell size values.
    :return: Isotropic metric corresponding to `f`.
    """
    try:
        assert f is not None
        assert len(f.ufl_element().value_shape()) == 0
    except AssertionError:
        raise ValueError("Provide a scalar function to compute an isotropic metric w.r.t.")
    V = f.function_space()
    mesh = V.mesh()
    V_ten = TensorFunctionSpace(mesh, V.ufl_element().family(), V.ufl_element().degree())
    dim = mesh.topological_dimension()
    assert dim in (2, 3)

    # Scale indicator
    assert op.normalisation in ('complexity', 'error')
    rescale = 1 if noscale else op.target

    # Project into P1 space
    op.print_debug("METRIC: Constructing isotropic metric...")
    M_diag = project(max_value(abs(rescale*f), 1e-10), V)
    M_diag.interpolate(abs(M_diag))  # Ensure positivity
    op.print_debug("METRIC: Done!")
    if not normalise:
        return interpolate(M_diag*Identity(dim), V_ten)

    # Normalise
    op.print_debug("METRIC: Normalising metric...")
    p = op.norm_order
    detM = Function(V).assign(M_diag)
    if p is not None:
        assert p >= 1
        detM *= M_diag
        M_diag *= pow(detM, -1/(2*p + dim))
        detM.interpolate(pow(detM, p/(2*p + dim)))
    if op.normalisation == 'complexity':
        M_diag *= pow(rescale/assemble(detM*ds), 2/dim)
    else:
        if p is not None:
            M_diag *= pow(assemble(detM*ds), 1/p)
    M_diag = max_value(1/pow(op.h_max, 2), min_value(M_diag, 1/pow(op.h_min, 2)))
    M_diag = max_value(M_diag, M_diag/pow(op.max_anisotropy, 2))
    op.print_debug("METRIC: Done!")
    return interpolate(M_diag*Identity(dim), V_ten)


def metric_intersection(*metrics, bdy=None):
    r"""
    Intersect a metric field, i.e. intersect (globally) over all local metrics.

    :arg metrics: metrics to be intersected.
    :param bdy: specify domain boundary to intersect over.
    :return: intersection of metrics M1 and M2.
    """
    n = len(metrics)
    assert n > 0
    M = metrics[0]
    for i in range(1, n):
        M = _metric_intersection_pair(M, metrics[i], bdy=bdy)
    return M


def _metric_intersection_pair(M1, M2, bdy=None):
    if bdy is not None:
        raise NotImplementedError  # FIXME: boundary intersection below does not work
    V = M1.function_space()
    node_set = V.boundary_nodes(bdy, 'topological') if bdy is not None else V.node_set
    dim = V.mesh().topological_dimension()
    assert dim in (2, 3)
    assert V == M2.function_space()
    M12 = M1.copy(deepcopy=True)
    kernel = eigen_kernel(intersect, dim)
    op2.par_loop(kernel, node_set, M12.dat(op2.RW), M1.dat(op2.READ), M2.dat(op2.READ))
    return M12


def metric_relaxation(*metrics, weights=None):
    r"""
    As an alternative to intersection, pointwise metric information may be combined using a convex
    combination. Whilst this method does not have as clear an interpretation as metric intersection,
    it has the benefit that the combination may be weighted towards one of the metrics in question.

    :arg metrics: metrics to be combined
    :kwarg weights: weights with which to average
    :return: convex combination
    """
    n = len(metrics)
    assert n > 0
    if weights is None:
        weights = np.ones(n)/n
    else:
        assert len(weights) == n
    V = metrics[0].function_space()
    M = Function(V)
    for i, Mi in enumerate(metrics):
        assert Mi.function_space() == V
        M += Mi*weights[i]
    return M


def metric_average(*metrics):
    return metric_relaxation(*metrics)


def combine_metrics(*metrics, average=True):
    return metric_average(*metrics) if average else metric_intersection(*metrics)


def metric_complexity(M):
    r"""
    Compute the complexity of a metric, which approximates the number of vertices in a mesh adapted
    based thereupon.
    """
    return assemble(sqrt(det(M))*dx)


def time_normalise(hessians, timesteps_per_remesh=None, op=Options()):
    r"""
    Normalise a list of Hessians in time as dictated by equation (1) in [Barral et al. 2016].

    :arg hessians: list of Hessians to be time-normalised.
    :kwarg timesteps_per_remesh: list of time integrals of 1/timestep over each remesh step.
        For constant timesteps, this equates to the number of timesteps per remesh step.
    :kwarg op: :class:`Options` object providing desired average instantaneous metric complexity.
    """
    p = op.norm_order
    n = len(hessians)
    if timesteps_per_remesh is None:
        timesteps_per_remesh = np.ones(n)*np.floor(op.end_time/op.dt)/op.num_meshes
    else:
        assert len(timesteps_per_remesh) == n
        assert n > 0
    d = hessians[0].function_space().mesh().topological_dimension()
    z = zip(hessians, timesteps_per_remesh)

    if op.normalisation == 'error':
        raise NotImplementedError  # TODO

    # Compute global normalisation coefficient
    op.print_debug("METRIC: Computing global metric time normalisation factor...")
    glob_norm = pow(op.target/sum(assemble(pow(det(H)*tpr**2, p/(2*p + d))*dx) for H, tpr in z), 2/d)
    op.print_debug("METRIC: Done!")
    op.print_debug("METRIC: Global normalisation factor: {:.4e}".format(glob_norm))

    op.print_debug("METRIC: Normalising metric in time...")
    # Normalise on each window
    for H, tpr in z:
        H.interpolate(glob_norm*pow(det(H)*tpr**2, -1/(2*p + d))*H)
        H.rename("Time-accurate {:s}".format(H.dat.name))
        enforce_element_constraints(H, op=op)  # Enforce max/min element sizes and anisotropy
    op.print_debug("METRIC: Done!")


# TODO: Test identities hold
def get_density_and_quotients(M):
    r"""
    Since metric fields are symmetric, they admit an orthogonal eigendecomposition,

  ..math::
        M(x) = V(x) \Lambda(x) V(x)^T,

    where :math:`V` and :math:`\Sigma` are matrices holding the eigenvectors and eigenvalues,
    respectively. Since metric fields are positive definite, we know :math:`\Lambda` is positive.

    The eigenvectors can be interpreted as defining the principal directions. The eigenvalue matrix
    can be decomposed further to give two meaningful fields: the metric density :math:`d` and
    anisotropic quotients, encapsulated by the diagonal matrix :math:`R`. These give rise to the
    decomposition

  ..math::
        M(x) = d(x)^\frac23 V(x) R(x)^{-\frac23} V(x)^T

    and are given by

  ..math::
        d = \sum_{i=1}^n h_i,\quad r_i = h_i^3/d,\quad \forall i=1:n,

    where :math:`h_i := \frac1{\sqrt{\lambda_i}}`.
    """
    fs_ten = M.function_space()
    mesh = fs_ten.mesh()
    fs_vec = VectorFunctionSpace(mesh, fs_ten.ufl_element())
    fs = FunctionSpace(mesh, fs_ten.ufl_element())
    dim = mesh.topological_dimension()

    # Setup fields
    V = Function(fs_ten, name="Eigenvectors")
    Λ = Function(fs_vec, name="Eigenvalues")
    h = Function(fs_vec, name="Sizes")
    density = Function(fs, name="Metric density")
    quotients = Function(fs_vec, name="Anisotropic quotients")

    # Compute eigendecomposition
    kernel = eigen_kernel(get_eigendecomposition, dim)
    op2.par_loop(kernel, fs_ten.node_set, V.dat(op2.RW), Λ.dat(op2.RW), M.dat(op2.READ))

    # Extract density and quotients
    h.interpolate(as_vector([1/sqrt(Λ[i]) for i in range(dim)]))
    d = Constant(1.0)
    for i in range(dim):
        d = d/h[i]
    density.interpolate(d)
    quotients.interpolate(as_vector([h[i]**3/d for i in range(dim)]))
    return density, quotients


def get_metric_coefficient(a, b, op=Options()):
    r"""
    Solve algebraic problem to get scaling coefficient for interior/boundary metric. See
    [Loseille et al. 2010] for details.

    :arg a: determinant integral associated with interior metric.
    :arg b: determinant integral associated with boundary metric.
    :kwarg op: `Options` class object providing min/max cell size values.
    :return: Scaling coefficient.
    """
    from sympy.solvers import solve
    from sympy import Symbol

    c = Symbol('c')
    sol = solve(a*pow(c, -0.6) + b*pow(c, -0.5) - op.target, c)
    assert len(sol) == 1
    return Constant(sol[0])


# TODO: test
def metric_with_boundary(f=None, H=None, h=None, mesh=None, degree=1, op=Options()):
    r"""
    Computes a Hessian-based steady metric for mesh adaptation, intersected with the corresponding
    boundary metric. The approach used here follows that of [Loseille et al. 2010].

    Clearly at least one of `f` and `H` must not be provided.

    :kwarg f: Field to compute the Hessian of.
    :kwarg H: Reconstructed Hessian associated with `f` (if already computed).
    :kwarg h: Reconstructed boundary Hessian associated with `f` (if already computed).
    :kwarg degree: Polynomial degree of Hessian.
    :kwarg op: `Options` class object providing min/max cell size values.
    :return: Intersected interior and boundary metric associated with Hessian `H`.
    """
    if f is None:
        try:
            assert not (H is None or h is None)
        except AssertionError:
            raise ValueError("Please supply either field for recovery, or Hessians thereof.")
    else:
        mesh = mesh or f.function_space().mesh()
        H = H or construct_hessian(f, op=op)
        if h is None:
            h = construct_boundary_hessian(f, op=op)
            h.interpolate(abs(h))
    V = h.function_space()
    V_ten = H.function_space()
    mesh = V.mesh()
    dim = mesh.topological_dimension()
    # assert dim in (2, 3)  # TODO
    assert dim == 2
    assert op.normalisation in ('complexity', 'error')

    # Functions to hold metric and boundary Hessian
    M_int = Function(V_ten).assign(0.0)
    M_bdy = Function(V_ten).assign(0.0)
    detM_int = Function(V).assign(0.0)
    detM_bdy = Function(V).assign(h)

    # Turn interior Hessian into a metric
    kernel = eigen_kernel(metric_from_hessian, dim, normalise=True, op=op)
    op2.par_loop(kernel, V_ten.node_set, M_int.dat(op2.RW), detM_int.dat(op2.RW), H.dat(op2.READ))

    # Normalise
    p = op.norm_order
    if p is not None:
        assert p >= 1
        h *= pow(h, -1/(2*p + dim-1))
        detM_bdy.interpolate(pow(detM_bdy, p/(2*p + dim-1)))

    # Solve algebraic problem for metric scale parameter, as in [Loseille et al. 2010]
    if op.normalisation == 'complexity':
        a = pow(op.target/assemble(detM_int*dx), 2/dim)
        b = pow(op.target/assemble(detM_bdy*ds), 2/(dim-1))
        # TODO: not sure about exponents here
        C = get_metric_coefficient(a, b, op=op)
        h *= C
    else:
        raise NotImplementedError  # TODO

    # Construct boundary metric
    h = max_value(1/pow(op.h_max, 2), min_value(h, 1/pow(op.h_min, 2)))
    # h = max_value(h, h/pow(op.max_anisotropy, 2))
    if dim == 2:
        M_bdy.interpolate(as_matrix([[1/pow(op.h_max, 2), 0], [0, h]]))
    else:
        raise NotImplementedError  # TODO

    # Scale interior metric
    if op.normalisation == 'complexity':
        M_int *= C
    else:
        raise NotImplementedError  # TODO
    kernel = eigen_kernel(scale_metric, dim, op=op)
    op2.par_loop(kernel, V_ten.node_set, M_int.dat(op2.RW))

    return metric_intersection(M_int, M_bdy, bdy=True)
