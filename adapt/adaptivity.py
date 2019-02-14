from firedrake import *

import numpy as np
import numpy
from numpy import linalg as la
from scipy import linalg as sla

from adapt.options import DefaultOptions


__all__ = ["construct_gradient", "construct_hessian", "steady_metric", "isotropic_metric", "iso_P2",
           "pointwise_max", "anisotropic_refinement", "gradate_metric", "local_metric_intersection",
           "metric_intersection", "metric_convex_combination", "symmetric_product",
           "metric_complexity", "normalise_indicator"]


def construct_gradient(f, mesh=None):
    """
    Assuming the function `f` is P1 (piecewise linear and continuous), direct differentiation will
    give a gradient which is P0 (piecewise constant and discontinuous). Since we would prefer a
    smooth gradient, an L2 projection gradient recovery technique is performed, which makes use of
    the Cl\'ement interpolation operator.

    :arg f: (scalar) P1 solution field.
    :return: reconstructed gradient associated with `f`.
    """
    # NOTE: A P1 field is not actually strictly required
    if mesh is None:
        mesh = f.function_space().mesh()
    W = VectorFunctionSpace(mesh, "CG", 1)
    g = Function(W)
    psi = TestFunction(W)
    Lg = (inner(g, psi) - inner(grad(f), psi)) * dx
    params = {'snes_rtol': 1e8,
              'ksp_rtol': 1e-5,
              'ksp_gmres_restart': 20,
              'pc_type': 'sor'}
    NonlinearVariationalSolver(NonlinearVariationalProblem(Lg, g), solver_parameters=params).solve()
    return g


def construct_hessian(f, g=None, mesh=None, op=DefaultOptions()):
    """
    Assuming the smooth solution field has been approximated by a function `f` which is P1, all
    second derivative information has been lost. As such, the Hessian of `f` cannot be directly
    computed. We provide two means of recovering it, as follows.

    (1) "Integration by parts" ('parts'):
    This involves solving the PDE $H = \nabla^T\nabla f$ in the weak sense. Code is based on the
    Monge-Amp\`ere tutorial provided on the Firedrake website:
    https://firedrakeproject.org/demos/ma-demo.py.html.

    (2) "Double L2 projection" ('dL2'):
    This involves two applications of the L2 projection operator given by `computeGradient`, above.

    :arg f: P1 solution field.
    :kwarg g: gradient (if already computed).
    :param op: AdaptOptions class object providing min/max cell size values.
    :return: reconstructed Hessian associated with ``f``.
    """
    # NOTE: A P1 field is not actually strictly required
    if mesh is None:
        mesh = f.function_space().mesh()
    V = TensorFunctionSpace(mesh, "CG", 1)
    H = Function(V)
    tau = TestFunction(V)
    nhat = FacetNormal(mesh)  # Normal vector
    if op.hessian_recovery == 'parts':
        Lh = (inner(tau, H) + inner(div(tau), grad(f))) * dx
        Lh -= (tau[0, 1] * nhat[1] * f.dx(0) + tau[1, 0] * nhat[0] * f.dx(1)) * ds
        # Term not in Firedrake tutorial:
        Lh -= (tau[0, 0] * nhat[1] * f.dx(0) + tau[1, 1] * nhat[0] * f.dx(1)) * ds
    elif op.hessian_recovery == 'dL2':
        if g is None:
            g = construct_gradient(f, mesh=mesh)
        Lh = (inner(tau, H) + inner(div(tau), g)) * dx
        Lh -= (tau[0, 1] * nhat[1] * g[0] + tau[1, 0] * nhat[0] * g[1]) * ds
        Lh -= (tau[0, 0] * nhat[1] * g[0] + tau[1, 1] * nhat[0] * g[1]) * ds
    H_prob = NonlinearVariationalProblem(Lh, H)
    NonlinearVariationalSolver(H_prob, solver_parameters={'snes_rtol': 1e8,
                                                          'ksp_rtol': 1e-5,
                                                          'ksp_gmres_restart': 20,
                                                          'pc_type': 'sor'}).solve()
    return H


def steady_metric(f, H=None, mesh=None, op=DefaultOptions()):
    """
    Computes the steady metric for mesh adaptation. Based on Nicolas Barral's function
    ``computeSteadyMetric``, from ``adapt.py``, 2016.

    :arg f: P1 solution field.
    :arg H: reconstructed Hessian associated with `f` (if already computed).
    :param op: AdaptOptions class object providing min/max cell size values.
    :return: steady metric associated with Hessian H.
    """
    # NOTE: A P1 field is not actually strictly required
    if H is None:
        H = construct_hessian(f, mesh=mesh, op=op)
    V = H.function_space()
    mesh = V.mesh()

    ia2 = 1. / pow(op.max_anisotropy, 2)  # Inverse square max aspect ratio
    ih_min2 = 1. / pow(op.h_min, 2)  # Inverse square minimal side-length
    ih_max2 = 1. / pow(op.h_max, 2)  # Inverse square maximal side-length
    M = Function(V)

    msg = "WARNING: minimum element size reached as {m:.2e}"

    if op.normalisation == 'manual':

        f_min = 1e-6  # Minimum tolerated value for the solution field

        for i in range(mesh.topology.num_vertices()):

            # Generate local Hessian, avoiding round-off error
            H_loc = H.dat.data[i] * op.target_vertices / max(np.sqrt(assemble(f * f * dx)), f_min)
            mean_diag = 0.5 * (H_loc[0][1] + H_loc[1][0])
            H_loc[0][1] = mean_diag
            H_loc[1][0] = mean_diag

            # Find eigenpairs and truncate eigenvalues
            lam, v = la.eig(H_loc)
            v1, v2 = v[0], v[1]
            lam1 = min(ih_min2, max(ih_max2, abs(lam[0])))
            lam2 = min(ih_min2, max(ih_max2, abs(lam[1])))
            lam_max = max(lam1, lam2)
            lam1 = max(lam1, ia2 * lam_max)
            lam2 = max(lam2, ia2 * lam_max)
            if (lam[0] >= 0.9999 * ih_min2) or (lam[1] >= 0.9999 * ih_min2):
                print(msg.format(m=np.sqrt(min(1. / lam[0], 1. / lam[1]))))

            # Reconstruct edited Hessian
            M.dat.data[i][0, 0] = lam1 * v1[0] * v1[0] + lam2 * v2[0] * v2[0]
            M.dat.data[i][0, 1] = lam1 * v1[0] * v1[1] + lam2 * v2[0] * v2[1]
            M.dat.data[i][1, 0] = M.dat.data[i][0, 1]
            M.dat.data[i][1, 1] = lam1 * v1[1] * v1[1] + lam2 * v2[1] * v2[1]
    else:
        detH = Function(FunctionSpace(mesh, "CG", 1))
        for i in range(mesh.topology.num_vertices()):

            # Generate local Hessian
            H_loc = H.dat.data[i]
            mean_diag = 0.5 * (H_loc[0][1] + H_loc[1][0])
            H_loc[0][1] = mean_diag
            H_loc[1][0] = mean_diag

            # Find eigenpairs of Hessian and truncate eigenvalues
            lam, v = la.eig(H_loc)
            v1, v2 = v[0], v[1]
            lam1 = max(abs(lam[0]), 1e-10)  # \ To avoid round-off error
            lam2 = max(abs(lam[1]), 1e-10)  # /
            det = lam1 * lam2

            # Reconstruct edited Hessian and rescale
            M.dat.data[i][0, 0] = lam1 * v1[0] * v1[0] + lam2 * v2[0] * v2[0]
            M.dat.data[i][0, 1] = lam1 * v1[0] * v1[1] + lam2 * v2[0] * v2[1]
            M.dat.data[i][1, 0] = M.dat.data[i][0, 1]
            M.dat.data[i][1, 1] = lam1 * v1[1] * v1[1] + lam2 * v2[1] * v2[1]
            M.dat.data[i] *= pow(det, -1. / (2 * op.norm_order + 2))
            detH.dat.data[i] = pow(det, op.norm_order / (2. * op.norm_order + 2))

        # Scale by the target number of vertices and Hessian complexity
        M *= op.target_vertices / assemble(detH * dx)

        for i in range(mesh.topology.num_vertices()):
            # Find eigenpairs of metric and truncate eigenvalues
            lam, v = la.eig(M.dat.data[i])
            v1, v2 = v[0], v[1]
            lam1 = min(ih_min2, max(ih_max2, abs(lam[0])))
            lam2 = min(ih_min2, max(ih_max2, abs(lam[1])))
            lam_max = max(lam1, lam2)
            lam1 = max(lam1, ia2 * lam_max)
            lam2 = max(lam2, ia2 * lam_max)
            if (lam[0] >= 0.9999 * ih_min2) or (lam[1] >= 0.9999 * ih_min2):
                print(msg.format(m=np.sqrt(min(1. / lam[0], 1. / lam[1]))))

            # Reconstruct edited Hessian
            M.dat.data[i][0, 0] = lam1 * v1[0] * v1[0] + lam2 * v2[0] * v2[0]
            M.dat.data[i][0, 1] = lam1 * v1[0] * v1[1] + lam2 * v2[0] * v2[1]
            M.dat.data[i][1, 0] = M.dat.data[i][0, 1]
            M.dat.data[i][1, 1] = lam1 * v1[1] * v1[1] + lam2 * v2[1] * v2[1]
    return M


def normalise_indicator(f, op=DefaultOptions()):
    """
    Normalise error indicator `f` using procedure defined by `op`.

    :arg f: error indicator to normalise.
    :param op: option parameters object.
    :return: normalised indicator.
    """
    scale_factor = min(max(norm(f), op.min_norm), op.max_norm)
    if scale_factor < 1.00001*op.min_norm:
        print("WARNING: minimum norm attained")
    elif scale_factor > 0.99999*op.max_norm:
        print("WARNING: maximum norm attained")
    f.interpolate(Constant(op.target_vertices / scale_factor) * abs(f))
    # NOTE: If `project` is used then positivity cannot be guaranteed

    return f


def isotropic_metric(f, bdy=None, op=DefaultOptions()):
    """
    Given a scalar error indicator field `f`, construct an associated isotropic metric field.

    :arg f: function to adapt to.
    :param bdy: specify domain boundary to compute metric on.
    :param op: AdaptOptions class object providing min/max cell size values.
    :return: isotropic metric corresponding to `f`.
    """
    h_min2 = pow(op.h_min, 2)
    h_max2 = pow(op.h_max, 2)
    scalar = len(f.ufl_element().value_shape()) == 0
    mesh = f.function_space().mesh()

    # Project into P1 space (if required)
    g = Function(FunctionSpace(mesh, "CG", 1) if scalar else VectorFunctionSpace(mesh, "CG", 1))
    if (f.ufl_element().family() == 'Lagrange') and (f.ufl_element().degree() == 1):
        g.assign(f)
    else:
        g.project(f)

    # Establish metric
    V = TensorFunctionSpace(mesh, "CG", 1)
    M = Function(V)
    if bdy is not None:
        for i in DirichletBC(V, 0, bdy).nodes:
            if scalar:
                alpha = max(1. / h_max2, min(g.dat.data[i], 1. / h_min2))
                beta = alpha
            else:
                alpha = max(1. / h_max2, min(g.dat.data[i, 0], 1. / h_min2))
                beta = max(1. / h_max2, min(g.dat.data[i, 1], 1. / h_min2))
            M.dat.data[i][0, 0] = alpha
            M.dat.data[i][1, 1] = beta

            if (alpha >= 0.9999 / h_min2) or (beta >= 0.9999 / h_min2):
                print("WARNING: minimum element size reached!")
    else:
        for i in range(len(M.dat.data)):
            if scalar:
                alpha = max(1. / h_max2, min(g.dat.data[i], 1. / h_min2))
                beta = alpha
            else:
                alpha = max(1. / h_max2, min(g.dat.data[i, 0], 1. / h_min2))
                beta = max(1. / h_max2, min(g.dat.data[i, 1], 1. / h_min2))
            M.dat.data[i][0, 0] = alpha
            M.dat.data[i][1, 1] = beta

            if (alpha >= 0.9999 / h_min2) or (beta >= 0.9999 / h_min2):
                print("WARNING: minimum element size reached!")
        #if scalar:  # FIXME
        #    alpha = Max(1./h_max2, Min(g, 1./h_min2))
        #    beta = alpha
        #else:
        #    alpha = Max(1./h_max2, Min(g[0], 1./h_min2))
        #    beta = Max(1./h_max2, Min(g[1], 1./h_min2))
        #M.interpolate(as_tensor([[alpha, 0],[0, beta]]))

    return M


def iso_P2(mesh):
    """
    Uniformly refine a mesh (in each canonical direction) using an iso-P2 refinement. That is, nodes
    of a quadratic element on the initial mesh become vertices of the new mesh.
    """
    return MeshHierarchy(mesh, 1).__getitem__(1)


def anisotropic_refinement(M, direction=0):
    """
    Anisotropically refine a mesh (or, more precisely, the metric field `M` associated with a mesh)
    in such a way as to approximately half the element size in a canonical direction (x- or y-), by
    scaling of the corresponding eigenvalue.

    :param M: metric to refine.
    :param direction: 0 or 1, corresponding to x- or y-direction, respectively.
    :return: anisotropically refined metric.
    """
    for i in range(len(M.dat.data)):
        lam, v = la.eig(M.dat.data[i])
        v1, v2 = v[0], v[1]
        lam[direction] *= 4
        M.dat.data[i][0, 0] = lam[0] * v1[0] * v1[0] + lam[1] * v2[0] * v2[0]
        M.dat.data[i][0, 1] = lam[0] * v1[0] * v1[1] + lam[1] * v2[0] * v2[1]
        M.dat.data[i][1, 0] = M.dat.data[i][0, 1]
        M.dat.data[i][1, 1] = lam[0] * v1[1] * v1[1] + lam[1] * v2[1] * v2[1]
    return M


def gradate_metric(M, iso=False, op=DefaultOptions()):  # TODO: Implement this in pyop2
    """
    Perform anisotropic metric gradation in the method described in Alauzet 2010, using linear
    interpolation. Python code found here is based on the PETSc code of Nicolas Barral's function
    ``DMPlexMetricGradation2d_Internal``, found in ``plex-metGradation.c``, 2017.

    :arg M: metric to be gradated.
    :param op: AdaptOptions class object providing parameter values.
    :return: gradated metric.
    """
    try:
        assert(M.ufl_element().family() == 'Lagrange' and M.ufl_element().degree() == 1)
    except:
        ValueError("Metric field must be P1.")
    try:
        assert(M.ufl_element().value_shape() == (2, 2))
    except:
        NotImplementedError('Only 2x2 metric fields considered so far.')
    ln_beta = np.log(op.max_element_growth)

    # Get vertices and edges of mesh
    V = M.function_space()
    M_grad = Function(V).assign(M)
    mesh = V.mesh()
    plex = mesh._plex
    vStart, vEnd = plex.getDepthStratum(0)  # Vertices
    eStart, eEnd = plex.getDepthStratum(1)  # Edges
    numVer = vEnd - vStart
    xy = mesh.coordinates.dat.data

    # Establish arrays for storage and a list of tags for vertices
    v12 = np.zeros(2)
    v21 = np.zeros(2)  # Could work only with the upper triangular part for speed
    verTag = np.zeros(numVer) + 1
    correction = True
    i = 0

    while correction and (i < 500):
        i += 1
        correction = False

        # Loop over edges of mesh
        for e in range(eStart, eEnd):
            cone = plex.getCone(e)  # Get vertices associated with edge e
            iVer1 = cone[0] - vStart  # Vertex 1 index
            iVer2 = cone[1] - vStart  # Vertex 2 index
            if (verTag[iVer1] < i) and (verTag[iVer2] < i):
                continue

            # Assemble local metrics and calculate edge lengths
            met1 = M_grad.dat.data[iVer1]
            met2 = M_grad.dat.data[iVer2]
            v12[0] = xy[iVer2][0] - xy[iVer1][0]
            v12[1] = xy[iVer2][1] - xy[iVer1][1]
            v21[0] = - v12[0]
            v21[1] = - v12[1]

            if iso:  # TODO: This does not currently work
                # eta2_12 = 1. / pow(1. + (v12[0] * v12[0] + v12[1] * v12[1]) * ln_beta / met1[0, 0], 2)
                eta2_12 = 1. / pow(1. + sqrt(symmetric_product(met1, v12)) * ln_beta, 2)
                # eta2_21 = 1. / pow(1. + (v21[0] * v21[0] + v21[1] * v21[1]) * ln_beta / met2[0, 0], 2)
                eta2_21 = 1. / pow(1. + sqrt(symmetric_product(met2, v21)) * ln_beta, 2)
                # print('#### gradate_metric DEBUG: 1,1 entries ', met1[0, 0], met2[0, 0])
                # print('#### gradate_metric DEBUG: scale factors', eta2_12, eta2_21)
                redMet1 = eta2_21 * met2
                redMet2 = eta2_12 * met1
            else:

                # Intersect metric with a scaled 'grown' metric to get reduced metric
                # eta2_12 = 1. / pow(1. + symmetric_product(met1, v12) * ln_beta, 2)
                eta2_12 = 1. / pow(1. + sqrt(symmetric_product(met1, v12)) * ln_beta, 2)
                #eta2_21 = 1. / pow(1. + symmetric_product(met2, v21) * ln_beta, 2)
                eta2_21 = 1. / pow(1. + sqrt(symmetric_product(met2, v21)) * ln_beta, 2)
                # print('#### gradate_metric DEBUG: scale factors', eta2_12, eta2_21)
                # print('#### gradate_metric DEBUG: determinants', la.det(met1), la.det(met2))
                redMet1 = local_metric_intersection(met1, eta2_21 * met2)
                redMet2 = local_metric_intersection(met2, eta2_12 * met1)

            # Calculate difference in order to ascertain whether the metric is modified
            diff = abs(met1[0, 0] - redMet1[0, 0])
            diff += abs(met1[0, 1] - redMet1[0, 1])
            diff += abs(met1[1, 1] - redMet1[1, 1])
            diff /= (abs(met1[0, 0]) + abs(met1[0, 1]) + abs(met1[1, 1]))
            if diff > 1e-3:
                M_grad.dat.data[iVer1] = redMet1
                verTag[iVer1] = i + 1
                correction = True

            # Repeat above process using other reduced metric
            diff = abs(met2[0, 0] - redMet2[0, 0])
            diff += abs(met2[0, 1] - redMet2[0, 1])
            diff += abs(met2[1, 1] - redMet2[1, 1])
            diff /= (abs(met2[0, 0]) + abs(met2[0, 1]) + abs(met2[1, 1]))
            if diff > 1e-3:
                M_grad.dat.data[iVer2] = redMet2
                verTag[iVer2] = i + 1
                correction = True

    return M_grad


def local_metric_intersection(M1, M2):
    """
    Intersect two metrics `M1` and `M2` defined at a particular point in space.
    """
    # print('#### local_metric_intersection DEBUG: attempting to compute sqrtm of matrix with determinant ', la.det(M1))
    sqM1 = sla.sqrtm(M1)
    sqiM1 = la.inv(sqM1)  # Note inverse and square root commute whenever both are defined
    lam, v = la.eig(np.dot(np.transpose(sqiM1), np.dot(M2, sqiM1)))
    M12 = np.dot(v, np.dot([[max(lam[0], 1), 0], [0, max(lam[1], 1)]], np.transpose(v)))
    return np.dot(np.transpose(sqM1), np.dot(M12, sqM1))


def metric_intersection(M1, M2, bdy=None):
    """
    Intersect a metric field, i.e. intersect (globally) over all local metrics.

    :arg M1: first metric to be intersected.
    :arg M2: second metric to be intersected.
    :param bdy: specify domain boundary to intersect over.
    :return: intersection of metrics M1 and M2.
    """
    V = M1.function_space()
    assert V == M2.function_space()
    M = Function(V).assign(M1)
    for i in DirichletBC(V, 0, bdy).nodes if bdy is not None else range(V.mesh().num_vertices()):
        M.dat.data[i] = local_metric_intersection(M1.dat.data[i], M2.dat.data[i])
        # print('#### metric_intersection DEBUG: det(Mi) = ', la.det(M1.dat.data[i]))
    return M


def metric_convex_combination(M1, M2, alpha=0.5):
    """
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
    return project(alpha*M1+(1-alpha)*M2, V)


def symmetric_product(A, b):
    """
    Compute the product of 2-vector `b` with itself, under the scalar product $b^T A b$ defined by
    the 2x2 matrix `A`.
    """

    # assert(isinstance(A, numpy.ndarray) | isinstance(A, Function))
    # assert(isinstance(b, list) | isinstance(b, numpy.ndarray) | isinstance(b, Function))

    def bAb(A, b):
        return b[0] * A[0, 0] * b[0] + 2 * b[0] * A[0, 1] * b[1] + b[1] * A[1, 1] * b[1]

    if isinstance(A, numpy.ndarray) | isinstance(A, list):
        if isinstance(b, list) | isinstance(b, numpy.ndarray):
            return bAb(A, b)
        else:
            return [bAb(A, b.dat.data[i]) for i in range(len(b.dat.data))]
    else:
        if isinstance(b, list) | isinstance(b, numpy.ndarray):
            return [bAb(A.dat.data[i], b) for i in range(len(A.dat.data))]
        else:
            return [bAb(A.dat.data[i], b.dat.data[i]) for i in range(len(A.dat.data))]


def pointwise_max(f, g):
    """
    Take the pointwise maximum (in modulus) of arrays `f` and `g`.
    """
    fu = f.ufl_element()
    gu = g.ufl_element()
    try:
        assert (len(f.dat.data) == len(g.dat.data))
    except:
        msg = "Function space mismatch: {f1:s} {d1:d} vs. {f2:s} {d2:d}"
        raise ValueError(msg.format(f1=fu.family(), d1=fu.degree(), f2=gu.family(), d2=gu.degree()))
    for i in range(len(f.dat.data)):
        if fu.value_size() == 1:
            if np.abs(g.dat.data[i]) > np.abs(f.dat.data[i]):
                f.dat.data[i] = g.dat.data[i]
        else:
            for j in range(fu.value_size()):
                if np.abs(g.dat.data[i, j]) > np.abs(f.dat.data[i, j]):
                    f.dat.data[i, j] = g.dat.data[i, j]
    return f


def metric_complexity(M):
    """
    Compute the complexity of a metric, which approximates the number of vertices in a mesh adapted
    based thereupon.
    """
    return assemble(sqrt(det(M)) * dx)

