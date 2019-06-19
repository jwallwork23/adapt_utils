from firedrake import *

import numpy as np
import numpy
from numpy import linalg as la
from scipy import linalg as sla

from adapt_utils.options import DefaultOptions
from adapt_utils.adapt.recovery import construct_hessian


__all__ = ["steady_metric", "isotropic_metric", "anisotropic_refinement", "metric_intersection", "metric_relaxation", "metric_complexity"]


def steady_metric(f, H=None, mesh=None, noscale=False, op=DefaultOptions()):
    r"""
    Computes the steady metric for mesh adaptation. Based on Nicolas Barral's function
    ``computeSteadyMetric``, from ``adapt.py``, 2016.

    :arg f: field to compute the Hessian of.
    :arg H: reconstructed Hessian associated with `f` (if already computed).
    :param op: `Options` class object providing min/max cell size values.
    :return: steady metric associated with Hessian H.
    """
    if H is None:
        H = construct_hessian(f, mesh=mesh, op=op)
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

        # Generate local Hessian
        H_loc = H.dat.data[k]
        for i in range(dim-1):
            for j in range(i+1, dim):
                mean_diag = 0.5*(H_loc[i][j] + H_loc[j][i])
                H_loc[i][j] = mean_diag
                H_loc[j][i] = mean_diag

        # Find eigenpairs of Hessian and truncate eigenvalues
        lam, v = la.eigh(H_loc)

        # Truncate eigenvalues to avoid round-off error
        det = 1.
        for i in range(dim):
            lam[i] = max(abs(lam[i]), 1e-10)
            det *= lam[i]

        # Reconstruct edited Hessian and rescale
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

    # Scale by target complexity / desired error
    if not noscale:
        if op.normalisation == 'complexity':
            M *= pow(op.target/assemble(detH*dx), 2/dim)
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

    :arg f: function to adapt to.
    :param op: `Options` class providing min/max cell size values.
    :return: isotropic metric corresponding to `f`.
    """
    assert len(f.ufl_element().value_shape()) == 0
    mesh = f.function_space().mesh()
    P1 = FunctionSpace(mesh, "CG", 1)
    dim = mesh.topological_dimension()
    assert dim in (2, 3)

    # Functions to hold metric and its diagonal entry
    M = Function(TensorFunctionSpace(mesh, "CG", 1))
    g = Function(P1)

    # Set parameters
    a2 = pow(op.max_anisotropy, 2)
    h_min2 = pow(op.h_min, 2)
    h_max2 = pow(op.h_max, 2)
    assert op.normalisation in ('complexity', 'error')
    rescale = 1 if noscale else op.target
    if f is not None:
        rescale /= max(norm(f), op.f_min)
    p = op.norm_order

    # Project into P1 space
    g.project(max_value(abs(rescale*f), 1e-10))
    g.interpolate(abs(g))  # ensure positivity

    # Normalise
    if not noscale:
        detM = Function(P1)
        detM.assign(g)
        if p is not None:
            assert p >= 1
            detM *= g
            g *= pow(detM, -1/(2*p + dim))
            detM.interpolate(pow(detM, p/(2*p + dim)))
        if op.normalisation == 'complexity':
            g *= pow(op.target/assemble(detM*dx), 2/dim)
        else:
            if p is not None:
                g *= pow(assemble(detM*dx), 1/p)

    # Construct metric
    alpha = max_value(1/h_max2, min_value(g, 1/h_min2))
    alpha = max_value(alpha, alpha/a2)
    if dim == 2:
        M.interpolate(as_matrix([[alpha, 0], [0, alpha]]))
    else:
        M.interpolate(as_matrix([[alpha, 0, 0], [0, alpha, 0], [0, 0, alpha]]))

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
    fs = metric.function_space()
    M = Function(fs)
    mesh = fs.mesh()
    dim = mesh.topological_dimension()
    assert dim in (2, 3)
    scale = 4 if dim == 2 else 8  # TODO: check this
    for k in range(mesh.num_vertices()):
        lam, v = la.eigh(metric.dat.data[k])
        lam[direction] *= scale
        # TODO: these loops could be done more efficiently by just adding extra terms in the skew direction
        for l in range(dim):
            for i in range(dim):
                for j in range(i, dim):
                    M.dat.data[k][i, j] += lam[l]*v[l][i]*v[l][j]
        for i in range(1, dim):
            for j in range(i):
                M.dat.data[k][i, j] = M.dat.data[k][j, i]
    return M

def local_metric_intersection(M1, M2, dim=2):
    r"""
    Intersect two metrics `M1` and `M2` defined at a particular point in space.

    :arg M1: the first metric.
    :arg M2: the second metric.
    :kwarg dim: spatial dimension of space.
    """
    sqM1 = sla.sqrtm(M1)
    sqiM1 = la.inv(sqM1)  # Note inverse and square root commute whenever both are defined
    lam, v = la.eigh(np.dot(np.transpose(sqiM1), np.dot(M2, sqiM1)))
    M12hat = np.zeros((dim, dim))
    for i in range(dim):
        M12hat[i, i] = max(lam[i], 1)
    M12 = np.dot(v, np.dot(M12hat, np.transpose(v)))
    return np.dot(np.transpose(sqM1), np.dot(M12, sqM1))

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
    M = M1.copy()
    for i in DirichletBC(V, 0, bdy).nodes if bdy is not None else range(V.mesh().num_vertices()):
        M.dat.data[i][:,:] = local_metric_intersection(M1.dat.data[i], M2.dat.data[i], dim=dim)
    return M

def metric_relaxation(M1, M2, alpha=0.5):
    r"""
    Alternatively to intersection, pointwise metric information may be combined using a convex
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

