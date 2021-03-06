"""
**********************************************************************************************
*  NOTE: Much of the code in this file is based on https://github.com/taupalosaurus/darwin.  *
**********************************************************************************************
"""
from firedrake import *

import numpy as np

from adapt_utils.adapt.kernels import *
import adapt_utils.adapt.recovery as recovery
from adapt_utils.linalg import check_spd
from adapt_utils.misc import prod
from adapt_utils.options import Options


__all__ = ["metric_complexity", "cell_size_metric", "volume_and_surface_contributions",
           "steady_metric", "isotropic_metric", "space_normalise", "space_time_normalise",
           "boundary_steady_metric", "combine_metrics", "metric_intersection", "metric_average",
           "HessianMetricRecoverer", "enforce_element_constraints"]


def metric_complexity(M, boundary=False):
    """
    Compute the complexity of a metric, which approximates the number of vertices in a mesh adapted
    based thereupon.
    """
    if boundary:
        return assemble(sqrt(det(M))*ds)
    else:
        return assemble(sqrt(det(M))*dx)


def steady_metric(f=None, H=None, projector=None, mesh=None, V=None, **kwargs):
    r"""
    Computes the steady metric for mesh adaptation.

    Clearly at least one of `f` and `H` must not be provided.

    :kwarg f: Field to compute the Hessian of.
    :kwarg H: Reconstructed Hessian associated with `f` (if already computed).
    :kwarg projector: :class:`DoubleL2Projector` object to compute Hessian.
    :kwarg normalise: toggle spatial normalisation.
    :kwarg enforce_constraints: Toggle enforcement of element size/anisotropy constraints.
    :kwarg op: :class:`Options` parameter class.
    :return: Steady metric associated with Hessian `H`.
    """
    kwargs.setdefault('normalise', True)
    kwargs.setdefault('enforce_constraints', True)
    kwargs.setdefault('noscale', False)
    kwargs.setdefault('op', Options())
    op = kwargs.get('op')
    if f is None:
        try:
            assert H is not None
        except AssertionError:
            raise ValueError("Please supply either field for recovery, or Hessian thereof.")
    elif H is None:
        H = recovery.recover_hessian(f, mesh=mesh, V=V, op=op) if projector is None else projector.project(f)
    V = V or H.function_space()
    if not hasattr(H, 'function_space'):
        H = interpolate(H, V)
    if mesh is None:
        mesh = V.mesh()
    dim = mesh.topological_dimension()
    assert dim in (2, 3)

    # Functions to hold metric and its determinant
    M = Function(V).assign(0.0)

    # Turn Hessian into a metric
    op.print_debug("METRIC: Constructing metric from Hessian...")
    kernel = eigen_kernel(metric_from_hessian, dim)
    op2.par_loop(kernel, V.node_set, M.dat(op2.RW), H.dat(op2.READ))

    # Apply Lp normalisation
    if kwargs.get('normalise'):
        op.print_debug("METRIC: Normalising metric in space...")
        noscale = kwargs.get('noscale')
        integral = kwargs.get('integral')
        space_normalise(M, noscale=noscale, f=f, integral=integral, op=op)

    # Enforce maximum/minimum element sizes and anisotropy
    if kwargs.get('enforce_constraints'):
        op.print_debug("METRIC: Enforcing elemental constraints...")
        enforce_element_constraints(M, op=op)

    # Check a valid metric
    if op.debug:
        check_spd(M)

    return M


def boundary_steady_metric(H, mesh=None, boundary_tag='on_boundary', **kwargs):
    """
    Computes a boundary metric from a boundary Hessian, following the approach of
    [Loseille et al. 2011].

    :kwarg H: reconstructed boundary Hessian
    :kwarg normalise: toggle spatial normalisation.
    :kwarg enforce_constraints: Toggle enforcement of element size/anisotropy constraints.
    :kwarg op: :class:`Options` parameter class.
    :return: boundary metric associated with Hessian `H`.
    """
    kwargs.setdefault('normalise', True)
    kwargs.setdefault('enforce_constraints', True)
    kwargs.setdefault('noscale', False)
    op = kwargs.get('op', Options())
    if mesh is None:
        mesh = H.function_space().mesh()
    dim = mesh.topological_dimension()
    if dim not in (2, 3):
        raise ValueError("Dimensions other than 2D and 3D not considered.")
    P1_ten = TensorFunctionSpace(mesh, "CG", 1)
    M = Function(P1_ten, name="Boundary metric")

    # Ensure positive definite
    op.print_debug("METRIC: Ensuring positivity of boundary metric...")
    M.interpolate(abs(H))

    # Apply Lp normalisation
    if kwargs.get('normalise'):
        op.print_debug("METRIC: Normalising boundary metric in space...")
        noscale = kwargs.get('noscale')
        integral = kwargs.get('integral')
        space_normalise(M, noscale=noscale, integral=integral, boundary=True, op=op)

    # Enforce maximum/minimum element sizes and anisotropy
    if kwargs.get('enforce_constraints'):
        op.print_debug("METRIC: Enforcing elemental constraints on boundary metric...")
        enforce_element_constraints(M, boundary_tag='on_boundary', op=op)

    # Check a valid metric
    if op.debug:
        check_spd(M)

    return M


class HessianMetricRecoverer(recovery.DoubleL2ProjectorHessian):
    """
    Subclass of :class:`DoubleL2RecovererHessian` which enables metric construction.
    """
    def construct_metric(self, sol, **kwargs):
        kwargs['op'] = self.op
        kwargs.pop('fields')
        return steady_metric(H=self.project(sol), **kwargs)


# TODO: Check equivalent to normalising in space and time separately
def space_time_normalise(hessians, timestep_integrals=None, enforce_constraints=True, op=Options()):
    r"""
    Normalise a list of Hessians in space and time as dictated by equation (1) in
    [Barral et al. 2016].

    :arg hessians: list of Hessians to be time-normalised.
    :kwarg timestep_integrals: list of time integrals of 1/timestep over each remesh step.
        For constant timesteps, this equates to the number of timesteps per remesh step.
    :kwarg op: :class:`Options` object providing desired average instantaneous metric complexity.
    """
    p = op.norm_order
    if p is not None:
        assert p >= 1
    n = len(hessians)
    timestep_integrals = timestep_integrals or np.ones(n)*np.floor(op.end_time/op.dt)/op.num_meshes
    assert len(timestep_integrals) == n
    d = hessians[0].function_space().mesh().topological_dimension()

    # Target space-time complexity
    N_st = op.target*sum(timestep_integrals)  # Multiply instantaneous target by number of timesteps

    # Compute global normalisation coefficient
    op.print_debug("METRIC: Computing global metric time normalisation factor...")
    integral = 0.0
    for i, H in enumerate(hessians):  # TODO: Check whether square is correct
        if p is None:
            integral += assemble(sqrt(det(H))*timestep_integrals[i]*dx)
        else:
            integral += assemble(pow(det(H)*timestep_integrals[i]**2, p/(2*p + d))*dx)
    if op.normalisation == 'complexity':
        glob_norm = pow(N_st/integral, 2/d)
    elif op.normalisation == 'error':  # FIXME
        glob_norm = d*N_st
        # glob_norm = d*op.target
        if p is not None:
            glob_norm *= pow(integral, 1/p)
    else:
        raise ValueError("Normalisation approach {:s} not recognised.".format(op.normalisation))
    op.print_debug("METRIC: Target space-time complexity = {:.4e}".format(N_st))
    op.print_debug("METRIC: Global normalisation factor = {:.4e}".format(glob_norm))

    # Normalise on each window
    op.print_debug("METRIC: Normalising metric in time...")
    for i, H in enumerate(hessians):
        H_expr = glob_norm*H
        if p is not None:
            H_expr *= pow(det(H)*timestep_integrals[i]**2, -1/(2*p + d))
        H.interpolate(H_expr)
        H.rename("Time-accurate {:s}".format(H.dat.name))

        # Enforce max/min element sizes and anisotropy
        if enforce_constraints:
            op.print_debug("METRIC: Enforcing size and ansisotropy constraints...")
            enforce_element_constraints(H, op=op)


def space_normalise(M, f=None, integral=None, boundary=False, **kwargs):
    r"""
    Normalise steady metric `M` based on field `f`.

    Four normalisation approaches are implemented. These include two 'straightforward' approaches:

      * 'norm'       - simply divide by the norm of `f`;
      * 'abs'        - divide by the maximum of `abs(f)` and some minimum tolerated value (see
                       equation (18) of [1]);

    and two :math:`\mathcal L_p` normalisation approaches, for :math:`p\geq1` or :math:`p=\infty`:

      * 'complexity' - Lp normalisation with a target metric complexity (see equation (2.10) of [2]);
      * 'error'      - Lp normalisation with a target interpolation error (see equation (7) of [3]).

    NOTE: In the case of 'error' normalisation, 'target' is the inverse of error.


    [1] Pain et al. "Tetrahedral mesh optimisation and adaptivity for steady-state and transient
        finite element calculations", Comput. Methods Appl. Mech. Engrg. 190 (2001) pp.3771-3796.

    [2] Loseille & Alauzet, "Continuous mesh framework part II: validations and applications",
        SIAM Numer. Anal. 49 (2011) pp.61-86.

    [3] Alauzet & Olivier, "An L∞-Lp space-time anisotropic mesh adaptation strategy for
        time-dependent problems", V European Conference on Computational Fluid Dynamics (2010).
    """
    kwargs.setdefault('op', Options())
    kwargs.setdefault('f_min', 1.0e-08)
    kwargs.setdefault('noscale', False)
    op = kwargs.get('op')

    # --- Simple normalisation approaches

    if op.normalisation in ('norm', 'abs'):
        assert f is not None
        tol = kwargs.get('f_min')
        denominator = norm(f) if op.normalisation == 'norm' else max_value(abs(f), tol)
        M.interpolate(M/denominator)
        return

    # --- Lp normalisation

    if op.normalisation not in ('complexity', 'error'):
        raise ValueError("Normalisation approach {:s} not recognised.".format(op.normalisation))
    p = op.norm_order
    mesh = M.function_space().mesh()
    d = mesh.topological_dimension()
    if boundary:
        d -= 1

    if integral is None:
        target = 1 if kwargs.get('noscale') else op.target
        form = pow(det(M), 0.5) if p is None else pow(det(M), p/(2*p + d))
        integral = assemble(form*ds) if boundary else assemble(form*dx)
        if op.normalisation == 'complexity':
            integral = pow(target/integral, 2/d)
        else:
            integral = 1 if p is None else pow(integral, 1/p)
            integral *= d*target
    assert integral > 1.0e-08
    determinant = 1 if p is None else pow(det(M), -1/(2*p + d))
    M.interpolate(integral*determinant*M)


def enforce_element_constraints(M, op=Options(), boundary_tag=None):
    """
    Post-process a metric `M` so that it obeys the following constraints specified by
    :class:`Options` class `op`:

      * `h_min`          - minimum element size;
      * `h_max`          - maximum element size;
      * `max_anisotropy` - maximum element anisotropy.
    """
    V = M.function_space()
    kernel = eigen_kernel(postproc_metric, V.mesh().topological_dimension(), op=op)
    node_set = V.node_set if boundary_tag is None else DirichletBC(V, 0, boundary_tag).node_set
    op2.par_loop(kernel, node_set, M.dat(op2.RW))


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
        M(x) = d(x)^\frac2n V(x) R(x)^{-\frac2n} V(x)^T

    and are given by

  ..math::
        d = \sum_{i=1}^n h_i,\quad r_i = h_i^n/d,\quad \forall i=1:n,

    where :math:`h_i := \frac1{\sqrt{\lambda_i}}`.
    """
    fs_ten = M.function_space()
    mesh = fs_ten.mesh()
    fe = (fs_ten.ufl_element().family(), fs_ten.ufl_element().degree())
    fs_vec = VectorFunctionSpace(mesh, *fe)
    fs = FunctionSpace(mesh, *fe)
    dim = mesh.topological_dimension()

    # Setup fields
    V = Function(fs_ten, name="Eigenvectors")
    Λ = Function(fs_vec, name="Eigenvalues")
    density = Function(fs, name="Metric density")
    quotients = Function(fs_vec, name="Anisotropic quotients")

    # Compute eigendecomposition
    kernel = eigen_kernel(get_eigendecomposition, dim)
    op2.par_loop(kernel, fs_ten.node_set, V.dat(op2.RW), Λ.dat(op2.RW), M.dat(op2.READ))

    # Extract density and quotients
    h = [1/sqrt(Λ[i]) for i in range(dim)]
    density.interpolate(1/prod(h))
    quotients.interpolate(as_vector([density*h[i]**dim for i in range(dim)]))
    return density, quotients


def isotropic_metric(f, **kwargs):
    r"""
    Given a scalar error indicator field `f`, construct an associated isotropic metric field.

    :arg f: Function to adapt to.
    :kwarg normalise: If `False` then we simply take the diagonal matrix with `f` in modulus.
    :kwarg op: :class:`Options` object providing min/max cell size values.
    :return: Isotropic metric corresponding to `f`.
    """
    kwargs.setdefault('normalise', True)
    kwargs.setdefault('enforce_constraints', True)
    kwargs.setdefault('noscale', False)
    kwargs.setdefault('op', Options())
    op = kwargs.get('op')
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

    # Project into P1 space
    op.print_debug("METRIC: Constructing isotropic metric...")
    M_diag = project(max_value(abs(f), 1e-12), V)
    M_diag.interpolate(abs(M_diag))  # Ensure positivity

    # Normalise
    if kwargs.get('normalise'):
        op.print_debug("METRIC: Normalising metric...")
        assert op.normalisation in ('complexity', 'error')
        rescale = 1.0 if kwargs.get('noscale') else op.target
        p = op.norm_order
        detM = M_diag.copy(deepcopy=True)
        if p is not None:
            assert p >= 1
            detM *= M_diag
            M_diag *= pow(detM, -1/(2*p + dim))
            detM.interpolate(pow(detM, p/(2*p + dim)))
        if op.normalisation == 'complexity':
            M_diag *= pow(rescale/assemble(detM*dx), 2/dim)
        else:  # FIXME!!!
            if p is not None:
                M_diag *= pow(assemble(detM*dx), 1/p)
            M_diag *= dim/rescale

    # Enforce maximum/minimum element sizes
    if kwargs.get('enforce_constraints'):
        op.print_debug("METRIC: Enforcing elemental constraints...")
        M_diag = max_value(1/pow(op.h_max, 2), min_value(M_diag, 1/pow(op.h_min, 2)))

    # Check a valid metric
    M = interpolate(M_diag*Identity(dim), V_ten)
    if op.debug:
        check_spd(M)
    return M


def cell_size_metric(mesh, op=Options()):
    P1 = FunctionSpace(mesh, "CG", 1)
    return isotropic_metric(interpolate(pow(CellSize(mesh), -2), P1), normalise=False, op=op)


# --- Metric combination methods

def metric_intersection(*metrics, **kwargs):
    r"""
    Intersect a metric field, i.e. intersect (globally) over all local metrics.

    :arg metrics: metrics to be intersected.
    :param boundary_tag: specify domain boundary to intersect over.
    :return: intersection of metrics M1 and M2.
    """
    n = len(metrics)
    assert n > 0
    M = metrics[0]
    for i in range(1, n):
        M = _metric_intersection_pair(M, metrics[i], **kwargs)
    return M


def _metric_intersection_pair(M1, M2, boundary_tag=None, **kwargs):
    V = M1.function_space()
    node_set = V.node_set if boundary_tag is None else DirichletBC(V, 0, boundary_tag).node_set
    dim = V.mesh().topological_dimension()
    assert dim in (2, 3)
    assert V == M2.function_space()
    M12 = M1.copy(deepcopy=True)
    kernel = eigen_kernel(intersect, dim)
    op2.par_loop(kernel, node_set, M12.dat(op2.RW), M1.dat(op2.READ), M2.dat(op2.READ))
    return M12


def metric_relaxation(*metrics, function_space=None, weights=None):
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
    V = function_space or metrics[0].function_space()
    M = Function(V)
    for i, Mi in enumerate(metrics):
        if not isinstance(Mi, Function) or Mi.function_space() != V:
            Mi = interpolate(Mi, V)
        M += Mi*weights[i]
    return M


def metric_average(*metrics, **kwargs):
    return metric_relaxation(*metrics, **kwargs)


def combine_metrics(*metrics, average=True, **kwargs):
    return metric_relaxation(*metrics, **kwargs) if average else metric_intersection(*metrics, **kwargs)


def volume_and_surface_contributions(interior_hessian, boundary_hessian, op=Options(), **kwargs):
    r"""
    Solve algebraic problem to get scaling coefficient for interior and boundary metrics to
    obtain a given metric complexity. See [Loseille et al. 2011] for details.

    :arg interior_hessian: component from domain interior.
    :arg boundary_hessian: component from domain boundary.
    :kwarg interior_hessian_scaling: positive scaling function for interior Hessian.
    :kwarg boundary_hessian_scaling: positive scaling function for boundary Hessian.
    :kwarg op: :class:`Options` parameters object.
    :return: Scaling coefficient.
    """
    from sympy import Symbol, solve

    # Collect parameters
    d = interior_hessian.function_space().mesh().topological_dimension()
    assert d in (2, 3)
    p = op.norm_order
    g = kwargs.get('interior_hessian_scaling', Constant(1.0))
    gbar = kwargs.get('boundary_hessian_scaling', Constant(1.0))
    op.print_debug("METRIC: original target complexity = {:.4e}".format(op.target))

    # Compute optimal coefficient for mixed interior-boundary Hessian
    if p is None:
        a = metric_complexity(interior_hessian, boundary=False)
        b = metric_complexity(boundary_hessian, boundary=True)
    else:
        a = assemble(pow(g, d/(2*p + d))*pow(det(interior_hessian), p/(2*p + d))*dx)
        b = assemble(pow(gbar, d/(2*p + d - 1))*pow(det(boundary_hessian), p/(2*p + d - 1))*ds)
    op.print_debug("METRIC: a = {:.4e}".format(a))
    op.print_debug("METRIC: b = {:.4e}".format(b))
    c = Symbol('c')
    # C = float(solve(a*pow(c, -d/(2*p + d)) + b*pow(c, -d/(2*p + d - 1)) - op.target, c)[0])
    C = float(solve(a*pow(c, d/2) + b*pow(c, (d-1)/2) - op.target, c)[0])  # NOTE: Differs from paper
    op.print_debug("METRIC: C = {:.4e}".format(C))
    return C
