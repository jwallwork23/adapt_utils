from firedrake import *
from firedrake.slate.slac.compiler import PETSC_DIR

import numpy as np

from adapt_utils.options import DefaultOptions
from adapt_utils.adapt.recovery import construct_hessian


__all__ = ["steady_metric", "isotropic_metric", "anisotropic_refinement", "gradate_metric",
           "metric_intersection", "metric_relaxation", "metric_complexity"]


def steady_metric(f, H=None, mesh=None, op=DefaultOptions()):
    r"""
    Computes the steady metric for mesh adaptation. Based on Nicolas Barral's function
    ``computeSteadyMetric``, from ``adapt.py``, 2016.

    :arg f: field to compute Hessian w.r.t.
    :arg H: reconstructed Hessian associated with `f` (if already computed).
    :param op: `Options` class object providing min/max cell size values.
    :return: steady metric associated with Hessian H.
    """
    if mesh is None:
        if f is not None:
            mesh = f.function_space().mesh()
        elif H is not None:
            mesh = H.function_space().mesh()
    if H is None:
        H = construct_hessian(f, mesh=mesh, op=op)
    else:
        try:
            assert H.ufl_element().family() == 'Lagrange'
            assert H.ufl_element().degree() == 1
        except:
            ValueError("Hessian must be P1.")
    V = H.function_space()
    M = Function(V)
    P1 = FunctionSpace(mesh, "CG", 1)

    ia2 = Constant(1/op.max_anisotropy**2)
    ih_min2 = Constant(1/op.h_min**2)
    ih_max2 = Constant(1/op.h_max**2)
    f_min = 1e-3

    if op.restrict in ('error', 'num_cells'):
        if op.restrict == 'error':
            #rescale = Constant(1/op.desired_error)
            #rescale = interpolate(1/(op.desired_error*max_value(abs(f), f_min)), P1)
            rescale = Constant(1/op.desired_error / max(norm(f), f_min))
        elif op.restrict == 'num_cells':
            rescale = Constant(op.target_vertices / max(norm(f), f_min))
            #rescale = interpolate(op.target_vertices / max_value(abs(f), f_min), P1)
        kernel_str = """
#include <Eigen/Dense>
#include <algorithm>

void metric(double A_[4], const double * B_, const double * scaling, const double * ihmin2, const double * ihmax2, const double * ia2)
{
  Eigen::Map<Eigen::Matrix<double, 2, 2, Eigen::RowMajor> > A((double *)A_);
  Eigen::Map<Eigen::Matrix<double, 2, 2, Eigen::RowMajor> > B((double *)B_);
  A = *scaling * B;
  double mean_diag = 0.5*(A(0,1) + A(1,0));
  A(0,1) = mean_diag;
  A(1,0) = mean_diag;

  Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 2, 2, Eigen::RowMajor>> eigensolver(A);
  Eigen::Matrix<double, 2, 2, Eigen::RowMajor> Q = eigensolver.eigenvectors();
  Eigen::Vector2d D = eigensolver.eigenvalues();
  D(0) = std::min(*ihmin2, std::max(*ihmax2, std::abs(D(0))));
  D(1) = std::min(*ihmin2, std::max(*ihmax2, std::abs(D(1))));
  double max_eig = std::max(D(0), D(1));
  D(0) = std::max(D(0), *ia2 * max_eig);
  D(1) = std::max(D(1), *ia2 * max_eig);

  A = Q * D.asDiagonal() * Q.transpose();
}
"""
        kernel = op2.Kernel(kernel_str, "metric", cpp=True, include_dirs=["%s/include/eigen3" % d for d in PETSC_DIR])
        op2.par_loop(kernel, V.node_set, M.dat(op2.RW), H.dat(op2.READ), rescale.dat(op2.READ), ih_min2.dat(op2.READ), ih_max2.dat(op2.READ), ia2.dat(op2.READ))

    elif op.restrict == 'p_norm':
        detH = Function(P1)
        p = op.norm_order
        kernel_str = """
#include <Eigen/Dense>
#include <algorithm>

void metric1(double A_[4], double * f, const double * B_)
{
  Eigen::Map<Eigen::Matrix<double, 2, 2, Eigen::RowMajor> > A((double *)A_);
  Eigen::Map<Eigen::Matrix<double, 2, 2, Eigen::RowMajor> > B((double *)B_);
  double mean_diag = 0.5*(B(0,1) + B(1,0));
  B(0,1) = mean_diag;
  B(1,0) = mean_diag;

  Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 2, 2, Eigen::RowMajor>> eigensolver(B);
  Eigen::Matrix<double, 2, 2, Eigen::RowMajor> Q = eigensolver.eigenvectors();
  Eigen::Vector2d D = eigensolver.eigenvalues();
  D(0) = std::max(1e-10, std::abs(D(0)));
  D(1) = std::max(1e-10, std::abs(D(1)));
  double det = D(0) * D(1);
  double scaling = pow(det, -1 / (2 * %s + 2));

  A += scaling * Q * D.asDiagonal() * Q.transpose();

  *f += pow(det, %s / (2 * %s + 2));
}
""" % (p, p, p)
        kernel = op2.Kernel(kernel_str, "metric1", cpp=True, include_dirs=["%s/include/eigen3" % d for d in PETSC_DIR])
        op2.par_loop(kernel, V.node_set, M.dat(op2.INC), detH.dat(op2.INC), H.dat(op2.READ))
        rescale = Constant(op.target_vertices / assemble(detH*dx))
        kernel_str = """
#include <Eigen/Dense>
#include <algorithm>

void metric2(double A_[4], const double * scaling, const double * ihmin2, const double * ihmax2, const double * ia2)
{
  Eigen::Map<Eigen::Matrix<double, 2, 2, Eigen::RowMajor> > A((double *)A_);

  A *= *scaling;
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 2, 2, Eigen::RowMajor>> eigensolver(A);
  Eigen::Matrix<double, 2, 2, Eigen::RowMajor> Q = eigensolver.eigenvectors();
  Eigen::Vector2d D = eigensolver.eigenvalues();
  D(0) = std::min(*ihmin2, std::max(*ihmax2, std::abs(D(0))));
  D(1) = std::min(*ihmin2, std::max(*ihmax2, std::abs(D(1))));
  double max_eig = std::max(D(0), D(1));
  D(0) = std::max(D(0), *ia2 * max_eig);
  D(1) = std::max(D(1), *ia2 * max_eig);

  A = Q * D.asDiagonal() * Q.transpose();
}
"""
        kernel = op2.Kernel(kernel_str, "metric2", cpp=True, include_dirs=["%s/include/eigen3" % d for d in PETSC_DIR])
        op2.par_loop(kernel, V.node_set, M.dat(op2.RW), rescale.dat(op2.READ), ih_min2.dat(op2.READ), ih_max2.dat(op2.READ), ia2.dat(op2.READ))

    else:
        raise ValueError("Restriction by {:s} not recognised.".format(op.restrict))

    return M


def isotropic_metric(f, bdy=None, op=DefaultOptions()):
    r"""
    Given a scalar error indicator field `f`, construct an associated isotropic metric field.

    :arg f: function to adapt to.
    :param bdy: specify domain boundary to compute metric on.
    :param op: `Options` class providing min/max cell size values.
    :return: isotropic metric corresponding to `f`.
    """
    assert len(f.ufl_element().value_shape()) == 0
    mesh = f.function_space().mesh()

    # Normalise indicator and project into P1 space
    f_norm = min(max(norm(f), op.min_norm), op.max_norm)
    scaling = 1/op.desired_error if op.restrict == 'error' else op.target_vertices
    g = project(scaling*abs(f)/f_norm, FunctionSpace(mesh, "CG", 1))

    # Establish metric
    V = TensorFunctionSpace(mesh, "CG", 1)
    M = Function(V)
    kernel_str = """
#include <Eigen/Dense>
#include <algorithm>

void isotropic(double A_[4], const double * eps, const double * hmin2, const double * hmax2) {
  Eigen::Map<Eigen::Matrix<double, 2, 2, Eigen::RowMajor> > A((double *)A_);

  double m = std::max(1/(*hmax2), std::min(std::abs(*eps), 1/(*hmin2)));
  A(0,0) = m;
  A(1,1) = m;
}
"""
    kernel = op2.Kernel(kernel_str, "isotropic", cpp=True, include_dirs=["%s/include/eigen3" % d for d in PETSC_DIR])
    op2.par_loop(kernel,
                 V.node_set if bdy is None else DirichletBC(V, 0, bdy).node_set, M.dat(op2.RW), g.dat(op2.READ), Constant(op.h_min**2).dat(op2.READ), Constant(op.h_max**2).dat(op2.READ))

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
    V = M.function_space()
    assert V.ufl_element().family() == 'Lagrange'
    assert V.ufl_element().degree() == 1
    dim = V.mesh().topological_dimension()
    d = str(dim)
    kernel_str = """
#include <Eigen/Dense>

void anisotropic(double A_[%s]) {
  Eigen::Map<Eigen::Matrix<double, %s, %s, Eigen::RowMajor> > A((double *)A_);

  Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, %s, %s, Eigen::RowMajor>> eigensolver(A);
  Eigen::Matrix<double, %s, %s, Eigen::RowMajor> Q = eigensolver.eigenvectors();
  Eigen::Vector%sd D = eigensolver.eigenvalues();
  Eigen::Array%sd D_array = D.array();
  D_array(%s) *= 4;
  A = Q * D_array.matrix().asDiagonal() * Q.transpose();
}
""" % (str(dim*dim), d, d, d, d, d, d, d, d, str(direction))
    kernel = op2.Kernel(kernel_str, "anisotropic", cpp=True, include_dirs=["%s/include/eigen3" % d for d in PETSC_DIR])
    op2.par_loop(kernel, V.node_set, M.dat(op2.RW))
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
        assert M.ufl_element().family() == 'Lagrange'
        assert M.ufl_element().degree() == 1
    except:
        ValueError("Metric field must be P1.")
    try:
        assert(M.ufl_element().value_shape() == (2, 2))
    except:
        NotImplementedError('Only 2x2 metric fields considered so far.')
    ln_beta = np.log(op.max_element_growth)

    def symmetric_product(A, b):
        return b[0] * A[0, 0] * b[0] + 2 * b[0] * A[0, 1] * b[1] + b[1] * A[1, 1] * b[1]

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
    dim = V.mesh().topological_dimension()
    assert dim in (2, 3)
    d = str(dim)
    M = Function(V).assign(M1)
    kernel_str = """
#include <Eigen/Dense>

void intersect(double M_[%s], const double * A_, const double * B_) {
  Eigen::Map<Eigen::Matrix<double, %s, %s, Eigen::RowMajor> > M((double *)M_);
  Eigen::Map<Eigen::Matrix<double, %s, %s, Eigen::RowMajor> > A((double *)A_);
  Eigen::Map<Eigen::Matrix<double, %s, %s, Eigen::RowMajor> > B((double *)B_);

  Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, %s, %s, Eigen::RowMajor>> eigensolver(A);
  Eigen::Matrix<double, %s, %s, Eigen::RowMajor> Q = eigensolver.eigenvectors();
  Eigen::Matrix<double, %s, %s, Eigen::RowMajor> D = eigensolver.eigenvalues().array().sqrt().matrix().asDiagonal();
  Eigen::Matrix<double, %s, %s, Eigen::RowMajor> Sq = Q * D * Q.transpose();
  Eigen::Matrix<double, %s, %s, Eigen::RowMajor> Sqi = Q * D.inverse() * Q.transpose();
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, %s, %s, Eigen::RowMajor>> eigensolver2(Sqi.transpose() * B * Sqi);
  Q = eigensolver2.eigenvectors();
  D = eigensolver2.eigenvalues().array().max(1).matrix().asDiagonal();
  M = Sq.transpose() * Q * D * Q.transpose() * Sq;
}
""" % (str(dim*dim), d, d, d, d, d, d, d, d, d, d, d, d, d, d, d, d, d, d)
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
    dim = V.mesh().topological_dimension()
    assert dim in (2, 3)
    d = str(dim)
    M = Function(V)
    kernel_str = """
#include <Eigen/Dense>

void relax(double M_[%s], const double * A_, const double * B_, const double * alpha) {
  Eigen::Map<Eigen::Matrix<double, %s, %s, Eigen::RowMajor> > M((double *)M_);
  Eigen::Map<Eigen::Matrix<double, %s, %s, Eigen::RowMajor> > A((double *)A_);
  Eigen::Map<Eigen::Matrix<double, %s, %s, Eigen::RowMajor> > B((double *)B_);

  M = (*alpha)*A + (1.0 - *alpha)*B;
}""" % (str(dim*dim), d, d, d, d, d, d)
    kernel = op2.Kernel(kernel_str, "relax", cpp=True, include_dirs=["%s/include/eigen3" % d for d in PETSC_DIR])
    op2.par_loop(kernel, V.node_set, M.dat(op2.RW), M1.dat(op2.READ), M2.dat(op2.READ), alpha.dat(op2.READ))
    return M


def metric_complexity(M):
    r"""
    Compute the complexity of a metric, which approximates the number of vertices in a mesh adapted
    based thereupon.
    """
    return assemble(sqrt(det(M))*dx)

