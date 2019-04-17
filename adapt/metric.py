from firedrake import *
from firedrake.slate.slac.compiler import PETSC_DIR

import numpy as np
import numpy
from numpy import linalg as la
from scipy import linalg as sla

from adapt_utils.options import DefaultOptions


__all__ = ["construct_gradient", "construct_hessian", "steady_metric", "isotropic_metric", "iso_P2",
           "pointwise_max", "anisotropic_refinement", "gradate_metric", "metric_intersection",
           "metric_relaxation", "symmetric_product", "metric_complexity", "normalise_indicator"]


def construct_gradient(f, mesh=None, op=DefaultOptions()):
    r"""
    Assuming the function `f` is P1 (piecewise linear and continuous), direct differentiation will
    give a gradient which is P0 (piecewise constant and discontinuous). Since we would prefer a
    smooth gradient, we solve an auxiliary finite element problem in P1 space. This "L2 projection"
    gradient recovery technique makes use of the Cl\'ement interpolation operator. That `f` is P1
    is not actually a requirement.

    :arg f: (scalar) P1 solution field.
    :kwarg mesh: mesh upon which Hessian is to be constructed. This must be applied if `f` is not a 
                 Function, but a ufl expression.
    :param op: `Options` class object providing min/max cell size values.
    :return: reconstructed gradient associated with `f`.
    """
    if mesh is None:
        mesh = f.function_space().mesh()
    P1_vec = VectorFunctionSpace(mesh, "CG", 1)
    g = TrialFunction(P1_vec)
    φ = TestFunction(P1_vec)
    # TODO: include an option to swap between these two: 'parts' vs 'L2'
    a = inner(g, φ)*dx
    # L = inner(grad(f), φ)*dx
    L = f*dot(φ, FacetNormal(mesh))*ds - f*div(φ)*dx  # enables f to be P0
    g = Function(P1_vec)
    solve(a == L, g, solver_parameters=op.hessian_solver_parameters)
    return g


def construct_hessian(f, mesh=None, op=DefaultOptions()):
    r"""
    Assuming the smooth solution field has been approximated by a function `f` which is P1, all
    second derivative information has been lost. As such, the Hessian of `f` cannot be directly
    computed. We provide two means of recovering it, as follows. That `f` is P1 is not actually
    a requirement.

    (1) "Integration by parts" ('parts'):
    This involves solving the PDE $H = \nabla^T\nabla f$ in the weak sense. Code is based on the
    Monge-Amp\`ere tutorial provided on the Firedrake website:
    https://firedrakeproject.org/demos/ma-demo.py.html.

    (2) "Double L2 projection" ('dL2'):
    This involves two applications of the L2 projection operator. In this mode, we are permitted
    to recover the Hessian of a P0 field, since no derivatives of `f` are required.

    :arg f: P1 solution field.
    :kwarg mesh: mesh upon which Hessian is to be constructed. This must be applied if `f` is not a 
                 Function, but a ufl expression.
    :param op: `Options` class object providing min/max cell size values.
    :return: reconstructed Hessian associated with `f`.
    """
    if mesh is None:
        mesh = f.function_space().mesh()
    P1_ten = TensorFunctionSpace(mesh, "CG", 1)
    n = FacetNormal(mesh)  # Normal vector

    # Integration by parts applied to the Hessian definition
    if op.hessian_recovery == 'parts':
        H = TrialFunction(P1_ten)
        τ = TestFunction(P1_ten)
        a = inner(tau, H)*dx
        L = -inner(div(τ), grad(f))*dx
        L += (τ[0, 1]*n[1]*f.dx(0) + τ[1, 0]*n[0]*f.dx(1))*ds
        L += (τ[0, 0]*n[1]*f.dx(0) + τ[1, 1]*n[0]*f.dx(1))*ds

        H = Function(P1_ten)
        solve(a == L, H, solver_parameters=op.hessian_solver_parameters)

    # Double L2 projection, using a mixed formulation for the gradient and Hessian
    elif op.hessian_recovery == 'dL2':
        P1_vec = VectorFunctionSpace(mesh, "CG", 1)
        V = P1_ten*P1_vec
        H, g = TrialFunctions(V)
        τ, φ = TestFunctions(V)
        a = inner(τ, H)*dx
        a += inner(φ, g)*dx
        a += inner(div(τ), g)*dx
        a += -(τ[0, 1]*n[1]*g[0] + τ[1, 0]*n[0]*g[1])*ds
        a += -(τ[0, 0]*n[1]*g[0] + τ[1, 1]*n[0]*g[1])*ds
        # L = inner(grad(f), φ)*dx
        L = f*dot(φ, n)*ds - f*div(φ)*dx  # enables f to be P0

        q = Function(V)
        solve(a == L, q)  # TODO: Solver parameters?
        H = q.split()[0]

    return H


def steady_metric(f, H=None, mesh=None, op=DefaultOptions()):
    r"""
    Computes the steady metric for mesh adaptation. Based on Nicolas Barral's function
    ``computeSteadyMetric``, from ``adapt.py``, 2016.

    :arg f: P1 solution field.
    :arg H: reconstructed Hessian associated with `f` (if already computed).
    :param op: `Options` class object providing min/max cell size values.
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

    if op.restrict == 'num_cells':
        f_min = 1e-6  # Minimum tolerated value for the solution field

        for i in range(mesh.num_vertices()):

            # Generate local Hessian, avoiding round-off error
            H_loc = H.dat.data[i] * op.target_vertices / max(sqrt(assemble(f*f*dx)), f_min)
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

    elif op.restrict == 'anisotropy':
        detH = Function(FunctionSpace(mesh, "CG", 1))

        for i in range(mesh.num_vertices()):

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

        for i in range(mesh.num_vertices()):
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
    else:
        raise ValueError("Restriction by {:s} not recognised.".format(op.restrict))
    return M


def normalise_indicator(f, op=DefaultOptions()):
    r"""
    Normalise error indicator `f` using procedure defined by `op`.

    :arg f: error indicator to normalise.
    :param op: `Options` parameters object.
    :return: normalised indicator.
    """
    # scale_factor = min(max(norm(abs(f)), op.min_norm), op.max_norm)
    scale_factor = min(max(sqrt(assemble(f*f*dx)), op.min_norm), op.max_norm)
    if scale_factor < 1.00001*op.min_norm:
        print("WARNING: minimum norm attained")
    elif scale_factor > 0.99999*op.max_norm:
        print("WARNING: maximum norm attained")
    f.interpolate(Constant(op.target_vertices / scale_factor) * abs(f))
    # NOTE: If `project` is used then positivity cannot be guaranteed

    return f


def isotropic_metric(f, bdy=None, op=DefaultOptions()):
    r"""
    Given a scalar error indicator field `f`, construct an associated isotropic metric field.

    :arg f: function to adapt to.
    :param bdy: specify domain boundary to compute metric on.
    :param op: `Options` class providing min/max cell size values.
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


# TODO: This is not really to do with metrics
def iso_P2(mesh):
    r"""
    Uniformly refine a mesh (in each canonical direction) using an iso-P2 refinement. That is, nodes
    of a quadratic element on the initial mesh become vertices of the new mesh.
    """
    return MeshHierarchy(mesh, 1).__getitem__(1)


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
    r"""
    Perform anisotropic metric gradation in the method described in Alauzet 2010, using linear
    interpolation. Python code found here is based on the PETSc code of Nicolas Barral's function
    ``DMPlexMetricGradation2d_Internal``, found in ``plex-metGradation.c``, 2017.

    :arg M: metric to be gradated.
    :param op: `Options` class providing parameter values.
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


def metric_intersection(M1, M2, bdy=None):
    r"""
    Intersect a metric field, i.e. intersect (globally) over all local metrics.

    :arg M1: first metric to be intersected.
    :arg M2: second metric to be intersected.
    :param bdy: specify domain boundary to intersect over.
    :return: intersection of metrics M1 and M2.
    """
    V = M1.function_space()
    assert V == M2.function_space()
    M = Function(V).assign(M1)
    kernel_str = """
#include <iostream>
#include <Eigen/Dense>

void intersect(double M_[4], const double * A_, const double * B_) {
  Eigen::Map<Eigen::Matrix<double, 2, 2, Eigen::RowMajor> > M((double *)M_);
  Eigen::Map<Eigen::Matrix<double, 2, 2, Eigen::RowMajor> > A((double *)A_);
  Eigen::Map<Eigen::Matrix<double, 2, 2, Eigen::RowMajor> > B((double *)B_);

  Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 2, 2, Eigen::RowMajor>> eigensolver(A);
  Eigen::Matrix<double, 2, 2, Eigen::RowMajor> Q = eigensolver.eigenvectors();
  Eigen::Matrix<double, 2, 2, Eigen::RowMajor> D = eigensolver.eigenvalues().array().sqrt().matrix().asDiagonal();
  Eigen::Matrix<double, 2, 2, Eigen::RowMajor> Sq = Q * D * Q.transpose();
  Eigen::Matrix<double, 2, 2, Eigen::RowMajor> Sqi = Q * D.inverse() * Q.transpose();
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 2, 2, Eigen::RowMajor>> eigensolver2(Sqi.transpose() * B * Sqi);
  Q = eigensolver2.eigenvectors();
  D = eigensolver2.eigenvalues().array().max(1).matrix().asDiagonal();
  M = Sq.transpose() * Q * D * Q.transpose() * Sq;
}
"""
    kernel = op2.Kernel(kernel_str, "intersect", cpp=True, include_dirs=["%s/include/eigen3" % d for d in PETSC_DIR])
    op2.par_loop(kernel, V.node_set if bdy is None else DirichletBC(V, 0, bdy).node_set, M.dat(op2.RW), M1.dat(op2.READ), M2.dat(op2.READ))
    return M


def metric_relaxation(M1, M2, alpha=Constant(0.5)):
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
    kernel_str = """
#include <Eigen/Dense>

void relax(double M_[4], const double * A_, const double * B_, const double * alpha) {
  Eigen::Map<Eigen::Matrix<double, 2, 2, Eigen::RowMajor> > M((double *)M_);
  Eigen::Map<Eigen::Matrix<double, 2, 2, Eigen::RowMajor> > A((double *)A_);
  Eigen::Map<Eigen::Matrix<double, 2, 2, Eigen::RowMajor> > B((double *)B_);

  M = (*alpha)*A + (1.0 - *alpha)*B;
}"""
    kernel = op2.Kernel(kernel_str, "relax", cpp=True, include_dirs=["%s/include/eigen3" % d for d in PETSC_DIR])
    op2.par_loop(kernel, V.node_set, M.dat(op2.RW), M1.dat(op2.READ), M2.dat(op2.READ), alpha.dat(op2.READ))
    return M


def abs_matmult(A, b):
    V = b.function_space()
    assert V.ufl_element().family() == A.function_space().ufl_element().family()
    assert V.ufl_element().degree() == A.function_space().ufl_element().degree()
    assert V.mesh() == A.function_space().mesh()
    v = Function(V)
    kernel_str = """
#include <Eigen/Dense>

void product(double y_[2], const double * A_, const double * b_) {
  Eigen::Map<Eigen::Vector2d > y((double *)y_);
  Eigen::Map<Eigen::Matrix<double, 2, 2, Eigen::RowMajor> > A((double *)A_);
  Eigen::Map<Eigen::Vector2d > b((double *)b_);

  y = A.array().abs().matrix()*b.array().abs().matrix();
}
"""
    kernel = op2.Kernel(kernel_str, "product", cpp=True, include_dirs=["%s/include/eigen3" % d for d in PETSC_DIR])
    op2.par_loop(kernel, V.node_set, v.dat(op2.RW), A.dat(op2.READ), b.dat(op2.READ))
    return v


def symmetric_product(A, b):
    r"""
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


# TODO: This is not really to do with metrics
def pointwise_max(f, g):
    r"""
    Take the pointwise maximum (in modulus) of arrays `f` and `g`.
    """
    fu = f.ufl_element()
    gu = g.ufl_element()
    try:
        assert (f.function_space() == g.function_space())
    except:
        msg = "Function space mismatch: {f1:s} {d1:d} vs. {f2:s} {d2:d}"
        raise ValueError(msg.format(f1=fu.family(), d1=fu.degree(), f2=gu.family(), d2=gu.degree()))
    try:
        assert fu.family() == 'Lagrange' and fu.degree() == 1
    except:
        raise NotImplementedError
    h = Function(f.function_space()).assign(np.finfo(0.).min)

    # TODO: Test
    max_kernel = """
for (int i=0; i<z.dofs; i++) {
    for (int j=0; j<3; j++) {
        z[i][j] = fmax(x[i][j], y[i][j]);
    }
}
"""
    par_loop(max_kernel, dx, {'x': (f, READ), 'y': (g, READ), 'z': (h, WRITE)})

    #for i in range(len(f.dat.data)):
    #    if fu.value_size() == 1:
    #        if np.abs(g.dat.data[i]) > np.abs(f.dat.data[i]):
    #            f.dat.data[i] = g.dat.data[i]
    #    else:
    #        for j in range(fu.value_size()):
    #            if np.abs(g.dat.data[i, j]) > np.abs(f.dat.data[i, j]):
    #                f.dat.data[i, j] = g.dat.data[i, j]
    return h


def metric_complexity(M):
    r"""
    Compute the complexity of a metric, which approximates the number of vertices in a mesh adapted
    based thereupon.
    """
    return assemble(sqrt(det(M))*dx)

