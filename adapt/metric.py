from firedrake import *

from adapt_utils.options import Options
from adapt_utils.adapt.recovery import construct_hessian, construct_boundary_hessian
from adapt_utils.adapt.kernels import *


__all__ = ["steady_metric", "isotropic_metric", "metric_with_boundary", "metric_intersection",
           "metric_relaxation", "metric_complexity", "combine_metrics"]


def steady_metric(f=None, H=None, mesh=None, noscale=False, degree=1, op=Options()):
    r"""
    Computes the steady metric for mesh adaptation. Based on Nicolas Barral's function
    ``computeSteadyMetric``, from ``adapt.py``, 2016.

    Clearly at least one of `f` and `H` must not be provided.

    :kwarg f: Field to compute the Hessian of.
    :kwarg H: Reconstructed Hessian associated with `f` (if already computed).
    :kwarg noscale: If `noscale == True` then we simply take the Hessian with eigenvalues in modulus.
    :kwarg degree: polynomial degree of Hessian.
    :kwarg op: `Options` class object providing min/max cell size values.
    :return: Steady metric associated with Hessian `H`.
    """
    if f is None:
        try:
            assert not H is None
        except AssertionError:
            raise ValueError("Please supply either field for recovery, or Hessian thereof.")
    elif H is None:
        mesh = mesh or f.function_space().mesh()
        H = construct_hessian(f, mesh=mesh, degree=degree, op=op)
    V = H.function_space()
    mesh = V.mesh()
    dim = mesh.topological_dimension()
    assert dim in (2, 3)  # TODO: test 3d case works
    assert op.normalisation in ('complexity', 'error')

    # Functions to hold metric and its determinant
    M = Function(V).assign(0.0)
    detH = Function(FunctionSpace(mesh, "CG", 1)).assign(0.0)

    # Turn Hessian into a metric
    kernel = eigen_kernel(metric_from_hessian, dim, noscale=noscale, op=op)
    op2.par_loop(kernel, V.node_set, M.dat(op2.RW), detH.dat(op2.RW), H.dat(op2.READ))

    if noscale:
        return M

    # Scale by target complexity / desired error
    det = assemble(detH*dx)
    if op.normalisation == 'complexity':
        assert det > 1e-8
        M *= pow(op.target/det, 2/dim)
    else:
        M *= dim*op.target  # NOTE in the 'error' case this is the inverse thereof
        if op.norm_order is not None:
            assert det > 1e-8
            M *= pow(det, 1/op.norm_order)
    kernel = eigen_kernel(scale_metric, dim, op=op)
    op2.par_loop(kernel, V.node_set, M.dat(op2.RW))

    return M

def isotropic_metric(f, noscale=False, degree=1, op=Options()):
    r"""
    Given a scalar error indicator field `f`, construct an associated isotropic metric field.

    :arg f: Function to adapt to.
    :kwarg noscale: If `noscale == True` then we simply take the diagonal matrix with `f` in modulus.
    :kwarg degree: polynomial degree of Hessian.
    :kwarg op: `Options` class providing min/max cell size values.
    :return: Isotropic metric corresponding to `f`.
    """
    try:
        assert not f is None
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
    M_diag = project(max_value(abs(rescale*f), 1e-10), V)
    M_diag.interpolate(abs(M_diag))  # Ensure positivity

    # Normalise
    if not noscale:
        p = op.norm_order
        detM = Function(V).assign(M_diag)
        if p is not None:
            assert p >= 1
            detM *= M_diag
            M_diag *= pow(detM, -1/(2*p + dim))
            detM.interpolate(pow(detM, p/(2*p + dim)))
        if op.normalisation == 'complexity':
            M_diag *= pow(op.target/assemble(detM*ds), 2/dim)
        else:
            if p is not None:
                M_diag *= pow(assemble(detM*ds), 1/p)
        M_diag = max_value(1/pow(op.h_max, 2), min_value(M_diag, 1/pow(op.h_min, 2)))
        M_diag = max_value(M_diag, M_diag/pow(op.max_anisotropy, 2))

    return interpolate(M_diag*Identity(dim), V_ten)

def metric_intersection(M1, M2, bdy=None):
    r"""
    Intersect a metric field, i.e. intersect (globally) over all local metrics.

    :arg M1: first metric to be intersected.
    :arg M2: second metric to be intersected.
    :param bdy: specify domain boundary to intersect over.
    :return: intersection of metrics M1 and M2.
    """
    V = M1.function_space()
    mesh = V.mesh()
    dim = mesh.topological_dimension()
    assert dim in (2, 3)
    assert V == M2.function_space()
    M12 = M1.copy()
    # FIXME: boundary intersection does not work
    node_set = V.boundary_nodes(bdy, 'topological') if bdy is not None else V.node_set
    kernel = eigen_kernel(intersect, dim)
    op2.par_loop(kernel, node_set, M12.dat(op2.RW), M1.dat(op2.READ), M2.dat(op2.READ))
    return M12

def metric_relaxation(M1, M2, alpha=0.5):
    r"""
    As an alternative to intersection, pointwise metric information may be combined using a convex
    combination. Whilst this method does not have as clear an interpretation as metric intersection,
    it has the benefit that the combination may be weighted towards one of the metrics in question.

    :arg M1: first metric to be combined.
    :arg M2: second metric to be combined.
    :param alpha: scalar parameter in [0,1].
    :return: convex combination of metrics M1 and M2 with parameter alpha.
    """
    V = M1.function_space()
    assert V == M2.function_space()
    M = Function(V)
    M += alpha*M1 + (1-alpha)*M2
    return M

def combine_metrics(M1, M2, average=True):
    if average:
        return metric_relaxation(M1, M2)
    else:
        return metric_intersection(M1, M2)

def metric_complexity(M):
    r"""
    Compute the complexity of a metric, which approximates the number of vertices in a mesh adapted
    based thereupon.
    """
    return assemble(sqrt(det(M))*dx)

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
        H = H or construct_hessian(f, mesh=mesh, degree=degree, op=op)
        if h is None:
            h = construct_boundary_hessian(f, mesh=mesh, degree=degree, op=op)
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
    kernel = eigen_kernel(metric_from_hessian, dim, noscale=False, op=op)
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
