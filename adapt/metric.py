from firedrake import *
try:
    from firedrake.slate.slac.compiler import PETSC_ARCH
except:
    import os

    PETSC_ARCH = os.environ.get('PETSC_ARCH')
    PETSC_DIR = os.environ.get('PETSC_DIR')
    PETSC_ARCH = os.path.join(PETSC_DIR, PETSC_ARCH)
    if not os.path.exists(os.path.join(PETSC_ARCH, 'include/eigen3')):
        PETSC_ARCH = '/usr/local'

import numpy as np
import numpy
from numpy import linalg as la
from scipy import linalg as sla

from adapt_utils.options import DefaultOptions
from adapt_utils.adapt.recovery import construct_hessian, construct_boundary_hessian
from adapt_utils.adapt.kernels import *


__all__ = ["steady_metric", "isotropic_metric", "metric_with_boundary", "anisotropic_refinement", "metric_intersection", "metric_relaxation", "metric_complexity"]


# TODO: par_loop
def steady_metric(f=None, H=None, mesh=None, noscale=False, degree=1, op=DefaultOptions()):
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
    assert not np.all(np.array([f, H]) == None)
    if H is None:
        if mesh is None:
            mesh = f.function_space().mesh()
        H = construct_hessian(f, mesh=mesh, degree=degree, op=op)
    V = H.function_space()
    mesh = V.mesh()
    dim = mesh.topological_dimension()
    assert dim in (2, 3)

    # Functions to hold metric and its determinant
    M = Function(V)
    detH = Function(FunctionSpace(mesh, "CG", 1))

    # Set parameters
    ia2 = pow(op.max_anisotropy, -2)
    ih_min2 = pow(op.h_min, -2)
    ih_max2 = pow(op.h_max, -2)
    assert op.normalisation in ('complexity', 'error')
    rescale = 1 if noscale else op.target
    if f is not None:
        rescale /= max(norm(f), op.f_min)
        #rescale = interpolate(rescale/max_value(abs(f), op.f_min))
    p = op.norm_order

    msg = "WARNING: minimum element size reached as {m:.2e}"
    for k in range(mesh.num_vertices()):

        # Ensure local Hessian is symmetric
        H_loc = H.dat.data[k]
        for i in range(dim-1):
            for j in range(i+1, dim):
                mean_diag = 0.5*(H_loc[i][j] + H_loc[j][i])
                H_loc[i][j] = mean_diag
                H_loc[j][i] = mean_diag

        # Find eigenpairs of Hessian
        lam, v = la.eigh(H_loc)

        # Take eigenvalues in modulus, so that the metric is SPD
        det = 1.
        for i in range(dim):
            lam[i] = max(abs(lam[i]), 1e-10)  # Truncate eigenvalues to avoid round-off error
            det *= lam[i]

        # Reconstruct edited Hessian
        for l in range(dim):
            for i in range(dim):
                for j in range(i, dim):
                    M.dat.data[k][i, j] += lam[l]*v[l][i]*v[l][j]
        for i in range(1, dim):
            for j in range(i):
                M.dat.data[k][i, j] = M.dat.data[k][j, i]

        # Apply Lp normalisation
        if not noscale:
            if p is None:
                if op.normalisation == 'complexity':
                    detH.dat.data[k] = np.sqrt(det)
            elif p >= 1:
                M.dat.data[k] *= pow(det, -1./(2*p + dim))
                detH.dat.data[k] = pow(det, p/(2.*p + dim))

    if noscale:
        return M

    # Scale by target complexity / desired error
    if op.normalisation == 'complexity':
        C = pow(op.target/assemble(detH*dx), 2/dim)
        M *= C
    else:
        M *= dim*op.target  # NOTE in the 'error' case this is the inverse thereof
        if p is not None:
            M *= pow(assemble(detH*dx), 1/p)

    for k in range(mesh.num_vertices()):

        # Find eigenpairs of metric
        lam, v = la.eigh(M.dat.data[k])

        # Impose maximum and minimum element sizes and maximum anisotropy
        det = 1.
        for i in range(dim):
            lam[i] = min(ih_min2, max(ih_max2, abs(lam[i])))
        lam_max = max(lam)
        for i in range(dim):
            lam[i] = max(lam[i], ia2*lam_max)
            if lam[i] < ia2*lam_max:
                lam[i] = ia2*lam_max
                raise Warning("Maximum anisotropy reached")
        if lam_max >= 0.9999*ih_min2:
            print(msg.format(m=np.sqrt(min(1./lam))))

        # Reconstruct edited Hessian
        for l in range(dim):
            for i in range(dim):
                for j in range(i, dim):
                    M.dat.data[k][i, j] += lam[l]*v[l][i]*v[l][j]
        for i in range(1, dim):
            for j in range(i):
                M.dat.data[k][i, j] = M.dat.data[k][j, i]
    return M

def isotropic_metric(f, noscale=False, op=DefaultOptions()):
    r"""
    Given a scalar error indicator field `f`, construct an associated isotropic metric field.

    :arg f: Function to adapt to.
    :kwarg noscale: If `noscale == True` then we simply take the diagonal matrix with `f` in modulus.
    :kwarg op: `Options` class providing min/max cell size values.
    :return: Isotropic metric corresponding to `f`.
    """
    assert len(f.ufl_element().value_shape()) == 0
    mesh = f.function_space().mesh()
    P1 = FunctionSpace(mesh, "CG", 1)
    dim = mesh.topological_dimension()
    assert dim in (2, 3)

    # Functions to hold metric and its diagonal entry
    M = Function(TensorFunctionSpace(mesh, "CG", 1))
    g = Function(P1)

    # Scale indicator
    assert op.normalisation in ('complexity', 'error')
    rescale = 1 if noscale else op.target
    if f is not None and not noscale:
        rescale /= max(norm(f), op.f_min)

    # Project into P1 space
    g.project(max_value(abs(rescale*f), 1e-10))
    g.interpolate(abs(g))  # ensure positivity

    # Normalise
    if not noscale:
        a2 = pow(op.max_anisotropy, 2)
        h_min2 = pow(op.h_min, 2)
        h_max2 = pow(op.h_max, 2)
        p = op.norm_order

        detM = Function(P1)
        detM.assign(g)
        if p is not None:
            assert p >= 1
            detM *= g
            g *= pow(detM, -1/(2*p + dim))
            detM.interpolate(pow(detM, p/(2*p + dim)))
        if op.normalisation == 'complexity':
            g *= pow(op.target/assemble(detM*ds), 2/dim)
        else:
            if p is not None:
                g *= pow(assemble(detM*ds), 1/p)

        g = max_value(1/h_max2, min_value(g, 1/h_min2))
        g = max_value(g, g/a2)

    # Construct metric
    if dim == 2:
        M.interpolate(as_matrix([[g, 0], [0, g]]))
    else:
        M.interpolate(as_matrix([[g, 0, 0], [0, g, 0], [0, 0, g]]))

    return M

# FIXME
# TODO: par_loop
def metric_with_boundary(f=None, mesh=None, H=None, op=DefaultOptions()):
    r"""
    Computes a Hessian-based steady metric for mesh adaptation, intersected with the corresponding
    boundary metric. The approach used here follows that of [Loseille et al. 2010].

    Clearly at least one of `f` and `H` must not be provided.

    :kwarg f: Field to compute the Hessian of.
    :kwarg H: Reconstructed Hessian associated with `f` (if already computed).
    :kwarg op: `Options` class object providing min/max cell size values.
    :return: Intersected interior and boundary metric associated with Hessian `H`.
    """
    assert not np.all(np.array([f, H]) == None)
    if H is None:
        if mesh is None:
            mesh = f.function_space().mesh()
        H = construct_hessian(f, mesh=mesh, op=op)
    P1_ten = H.function_space()
    mesh = P1_ten.mesh()
    dim = mesh.topological_dimension()
    #assert dim in (2, 3)
    try:
        assert dim == 2
    except:
        raise NotImplementedError  # TODO

    # Functions to hold metric and boundary Hessian
    M_int = Function(P1_ten)
    M_bdy = Function(P1_ten)
    h = construct_boundary_hessian(f, mesh=mesh, op=op)
    P1 = h.function_space()
    detM_int = Function(P1)
    detM_bdy = Function(P1)

    # Set parameters
    a2 = pow(op.max_anisotropy, 2)
    h_min2 = pow(op.h_min, 2)
    h_max2 = pow(op.h_max, 2)
    #assert op.normalisation in ('complexity', 'error')
    try:
        assert op.normalisation == 'complexity'
    except:
        raise NotImplementedError  # TODO
    rescale = op.target
    if f is not None:
        rescale /= max(norm(f), op.f_min)
    p = op.norm_order

    # Compute interior metric
    msg = "WARNING: minimum element size reached as {m:.2e}"
    for k in range(mesh.num_vertices()):

        # Ensure local Hessian is symmetric
        H_loc = H.dat.data[k]
        for i in range(dim-1):
            for j in range(i+1, dim):
                mean_diag = 0.5*(H_loc[i][j] + H_loc[j][i])
                H_loc[i][j] = mean_diag
                H_loc[j][i] = mean_diag

        # Find eigenpairs of Hessian
        lam, v = la.eigh(H_loc)

        # Take eigenvalues in modulus, so that the metric is SPD
        det = 1.
        for i in range(dim):
            lam[i] = max(abs(lam[i]), 1e-10)  # Truncate eigenvalues to avoid round-off error
            det *= lam[i]

        # Reconstruct edited Hessian
        for l in range(dim):
            for i in range(dim):
                for j in range(i, dim):
                    M_int.dat.data[k][i, j] += lam[l]*v[l][i]*v[l][j]
        for i in range(1, dim):
            for j in range(i):
                M_int.dat.data[k][i, j] = M_int.dat.data[k][j, i]

        # Apply Lp normalisation
        if p is None:
            if op.normalisation == 'complexity':
                detM_int.dat.data[k] = np.sqrt(det)
        elif p >= 1:
            M_int.dat.data[k] *= pow(det, -1./(2*p + dim))
            detM_int.dat.data[k] = pow(det, p/(2.*p + dim))

    # Normalise
    h.interpolate(abs(h))
    detM_bdy.assign(h)
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
    h = max_value(1/h_max2, min_value(h, 1/h_min2))
    #h = max_value(h, h/a2)

    # Construct boundary metric
    if dim == 2:
        M_bdy.interpolate(as_matrix([[1/h_max2, 0], [0, h]]))
    else:
       raise NotImplementedError  # TODO

    # Scale interior metric
    if op.normalisation == 'complexity':
        M_int *= C
    else:
       raise NotImplementedError  # TODO

    for k in range(mesh.num_vertices()):

        # Find eigenpairs of metric
        lam, v = la.eigh(M_int.dat.data[k])

        # Impose maximum and minimum element sizes and maximum anisotropy
        det = 1.
        for i in range(dim):
            lam[i] = min(1/h_min2, max(1/h_max2, abs(lam[i])))
        lam_max = max(lam)
        for i in range(dim):
            lam[i] = max(lam[i], lam_max/a2)
            if lam[i] < lam_max/a2:
                lam[i] = lam_max/a2
                raise Warning("Maximum anisotropy reached")
        if lam_max >= 0.9999/h_min2:
            print(msg.format(m=np.sqrt(min(1./lam))))

        # Reconstruct edited Hessian
        for l in range(dim):
            for i in range(dim):
                for j in range(i, dim):
                    M_int.dat.data[k][i, j] += lam[l]*v[l][i]*v[l][j]
        for i in range(1, dim):
            for j in range(i):
                M_int.dat.data[k][i, j] = M_int.dat.data[k][j, i]

    # Intersect interior and boundary metrics
    M = metric_intersection(M_int, M_bdy, bdy=True)

    return M

def anisotropic_refinement(metric, direction=0):
    r"""
    Anisotropically refine a mesh (or, more precisely, the metric field associated with a mesh)
    in such a way as to approximately half the element size in a canonical direction (x- or y-), by
    scaling of the corresponding eigenvalue.

    :param M: metric to refine.
    :param direction: 0 or 1, corresponding to x- or y-direction, respectively.
    :return: anisotropically refined metric.
    """
    M = metric.copy()
    fs = M.function_space()
    dim = fs.mesh().topological_dimension()
    kernel = op2.Kernel(anisotropic_refinement_kernel(dim, direction), "anisotropic", cpp=True, include_dirs=["%s/include/eigen3" % PETSC_ARCH])
    op2.par_loop(kernel, fs.node_set, M.dat(op2.RW))
    return M

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
    node_set = DirichletBC(V, 0, bdy).nodes if bdy is not None else V.node_set
    kernel = op2.Kernel(intersect_kernel(dim), "intersect", cpp=True, include_dirs=["%s/include/eigen3" % PETSC_ARCH])
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

def metric_complexity(M):
    r"""
    Compute the complexity of a metric, which approximates the number of vertices in a mesh adapted
    based thereupon.
    """
    return assemble(sqrt(det(M))*dx)

def get_metric_coefficient(a, b, op=DefaultOptions()):
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
