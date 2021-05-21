"""
**********************************************************************************************
*  NOTE: Much of the code in this file is based on https://github.com/taupalosaurus/darwin.  *
**********************************************************************************************
"""
from firedrake import op2
import os


__all__ = ["eigen_kernel", "get_eigendecomposition", "get_reordered_eigendecomposition",
           "set_eigendecomposition", "set_eigendecomposition_transpose", "intersect",
           "anisotropic_refinement", "metric_from_hessian", "postproc_metric",
           "gemv", "matscale", "poldec_unitary", "poldec_spd", "get_maximum_length_edge"]


# --- Find Eigen include files

try:
    from firedrake.slate.slac.compiler import PETSC_ARCH
except ImportError:
    PETSC_ARCH = os.path.join(os.environ.get('PETSC_DIR'), os.environ.get('PETSC_ARCH'))
if not os.path.isdir(os.path.join(PETSC_ARCH, 'include/eigen3')):
    PETSC_ARCH = '/usr/local'
include_dir = ["%s/include/eigen3" % PETSC_ARCH]


# --- C++ kernels

get_eigendecomposition_str = """
#include <Eigen/Dense>

using namespace Eigen;

void get_eigendecomposition(double EVecs_[%d], double EVals_[%d], const double * M_) {

  // Map inputs and outputs onto Eigen objects
  Map<Matrix<double, %d, %d, RowMajor> > EVecs((double *)EVecs_);
  Map<Vector%dd> EVals((double *)EVals_);
  Map<Matrix<double, %d, %d, RowMajor> > M((double *)M_);

  // Solve eigenvalue problem
  SelfAdjointEigenSolver<Matrix<double, %d, %d, RowMajor>> eigensolver(M);
  EVecs = eigensolver.eigenvectors();
  EVals = eigensolver.eigenvalues();
}
"""

get_reordered_eigendecomp_2d_str = """
#include <Eigen/Dense>

using namespace Eigen;

void get_reordered_eigendecomposition(double EVecs_[4], double EVals_[2], const double * M_) {

  // Map inputs and outputs onto Eigen objects
  Map<Matrix<double, 2, 2, RowMajor> > EVecs((double *)EVecs_);
  Map<Vector2d> EVals((double *)EVals_);
  Map<Matrix<double, 2, 2, RowMajor> > M((double *)M_);

  // Solve eigenvalue problem
  SelfAdjointEigenSolver<Matrix<double, 2, 2, RowMajor>> eigensolver(M);
  Matrix<double, 2, 2, RowMajor> Q = eigensolver.eigenvectors();
  Vector2d D = eigensolver.eigenvalues();

  // Reorder eigenpairs by magnitude of eigenvalue
  if (fabs(D(0)) > fabs(D(1))) {
    EVecs = Q;
    EVals = D;
  } else {
    EVecs(0,0) = Q(0,1);EVecs(0,1) = Q(0,0);
    EVecs(1,0) = Q(1,1);EVecs(1,1) = Q(1,0);
    EVals(0) = D(1);
    EVals(1) = D(0);
  }
}
"""

get_reordered_eigendecomp_3d_str = """
#include <Eigen/Dense>

using namespace Eigen;

void get_reordered_eigendecomposition(double EVecs_[9], double EVals_[3], const double * M_) {

  // Map inputs and outputs onto Eigen objects
  Map<Matrix<double, 3, 3, RowMajor> > EVecs((double *)EVecs_);
  Map<Vector3d> EVals((double *)EVals_);
  Map<Matrix<double, 3, 3, RowMajor> > M((double *)M_);

  // Solve eigenvalue problem
  SelfAdjointEigenSolver<Matrix<double, 3, 3, RowMajor>> eigensolver(M);
  Matrix<double, 3, 3, RowMajor> Q = eigensolver.eigenvectors();
  Vector3d D = eigensolver.eigenvalues();

  // Reorder eigenpairs by magnitude of eigenvalue
  if (fabs(D(0)) > fabs(D(1))) {
    if (fabs(D(1)) > fabs(D(2))) {
      EVecs = Q;
      EVals = D;
    } else if (fabs(D(0)) > fabs(D(2))) {
      EVecs(0,0) = Q(0,0);EVecs(0,1) = Q(0,2);EVecs(0,2) = Q(0,1);
      EVecs(1,0) = Q(1,0);EVecs(1,1) = Q(1,2);EVecs(1,2) = Q(1,1);
      EVecs(2,0) = Q(2,0);EVecs(2,1) = Q(2,2);EVecs(2,2) = Q(2,1);
      EVals(0) = D(0);
      EVals(1) = D(2);
      EVals(2) = D(1);
    } else {
      EVecs(0,0) = Q(0,2);EVecs(0,1) = Q(0,0);EVecs(0,2) = Q(0,1);
      EVecs(1,0) = Q(1,2);EVecs(1,1) = Q(1,0);EVecs(1,2) = Q(1,1);
      EVecs(2,0) = Q(2,2);EVecs(2,1) = Q(2,0);EVecs(2,2) = Q(2,1);
      EVals(0) = D(2);
      EVals(1) = D(0);
      EVals(2) = D(1);
    }
  } else {
    if (fabs(D(0)) > fabs(D(2))) {
      EVecs(0,0) = Q(0,1);EVecs(0,1) = Q(0,0);EVecs(0,2) = Q(0,2);
      EVecs(1,0) = Q(1,1);EVecs(1,1) = Q(1,0);EVecs(1,2) = Q(1,2);
      EVecs(2,0) = Q(2,1);EVecs(2,1) = Q(2,0);EVecs(2,2) = Q(2,2);
      EVals(0) = D(1);
      EVals(1) = D(0);
      EVals(2) = D(2);
    } else if (fabs(D(1)) > fabs(D(2))) {
      EVecs(0,0) = Q(0,1);EVecs(0,1) = Q(0,2);EVecs(0,2) = Q(0,0);
      EVecs(1,0) = Q(1,1);EVecs(1,1) = Q(1,2);EVecs(1,2) = Q(1,0);
      EVecs(2,0) = Q(2,1);EVecs(2,1) = Q(2,2);EVecs(2,2) = Q(2,0);
      EVals(0) = D(1);
      EVals(1) = D(2);
      EVals(2) = D(0);
    } else {
      EVecs(0,0) = Q(0,2);EVecs(0,1) = Q(0,1);EVecs(0,2) = Q(0,0);
      EVecs(1,0) = Q(1,2);EVecs(1,1) = Q(1,1);EVecs(1,2) = Q(1,0);
      EVecs(2,0) = Q(2,2);EVecs(2,1) = Q(2,1);EVecs(2,2) = Q(2,0);
      EVals(0) = D(2);
      EVals(1) = D(1);
      EVals(2) = D(0);
    }
  }
}
"""

set_eigendecomposition_str = """
#include <Eigen/Dense>

using namespace Eigen;

void set_eigendecomposition(double M_[%d], const double * EVecs_, const double * EVals_) {

  // Map inputs and outputs onto Eigen objects
  Map<Matrix<double, %d, %d, RowMajor> > M((double *)M_);
  Map<Matrix<double, %d, %d, RowMajor> > EVecs((double *)EVecs_);
  Map<Vector%dd> EVals((double *)EVals_);

  // Compute metric from eigendecomposition
  M = EVecs * EVals.asDiagonal() * EVecs.transpose();
}
"""

set_eigendecomposition_transpose_str = """
#include <Eigen/Dense>

using namespace Eigen;

void set_eigendecomposition_transpose(double M_[%d], const double * EVecs_, const double * EVals_) {

  // Map inputs and outputs onto Eigen objects
  Map<Matrix<double, %d, %d, RowMajor> > M((double *)M_);
  Map<Matrix<double, %d, %d, RowMajor> > EVecs((double *)EVecs_);
  Map<Vector%dd> EVals((double *)EVals_);

  // Compute metric from transpose eigendecomposition
  M = EVecs.transpose() * EVals.asDiagonal() * EVecs;
}
"""

intersect_str = """
#include <Eigen/Dense>

using namespace Eigen;

void intersect(double M_[%d], const double * A_, const double * B_) {

  // Map inputs and outputs onto Eigen objects
  Map<Matrix<double, %d, %d, RowMajor> > M((double *)M_);
  Map<Matrix<double, %d, %d, RowMajor> > A((double *)A_);
  Map<Matrix<double, %d, %d, RowMajor> > B((double *)B_);

  // Solve eigenvalue problem of first metric, taking square root of eigenvalues
  SelfAdjointEigenSolver<Matrix<double, %d, %d, RowMajor>> eigensolver(A);
  Matrix<double, %d, %d, RowMajor> Q = eigensolver.eigenvectors();
  Matrix<double, %d, %d, RowMajor> D = eigensolver.eigenvalues().array().sqrt().matrix().asDiagonal();

  // Compute square root and inverse square root metrics
  Matrix<double, %d, %d, RowMajor> Sq = Q * D * Q.transpose();
  Matrix<double, %d, %d, RowMajor> Sqi = Q * D.inverse() * Q.transpose();

  // Solve eigenvalue problem for triple product of inverse square root metric and the second metric
  SelfAdjointEigenSolver<Matrix<double, %d, %d, RowMajor>> eigensolver2(Sqi.transpose() * B * Sqi);
  Q = eigensolver2.eigenvectors();
  D = eigensolver2.eigenvalues().array().max(1).matrix().asDiagonal();

  // Compute metric intersection
  M = Sq.transpose() * Q * D * Q.transpose() * Sq;
}
"""

anisotropic_refinement_str = """
#include <Eigen/Dense>

using namespace Eigen;

void anisotropic_refinement(double A_[%d]) {

  // Map input/output metric onto an Eigen object
  Map<Matrix<double, %d, %d, RowMajor> > A((double *)A_);

  // Solve eigenvalue problem
  SelfAdjointEigenSolver<Matrix<double, %d, %d, RowMajor>> eigensolver(A);
  Matrix<double, %d, %d, RowMajor> Q = eigensolver.eigenvectors();
  Vector%dd D = eigensolver.eigenvalues();

  // Scale eigenvalue in appropriate coordinate direction
  Array%dd D_array = D.array();
  D_array(%d) *= %f;

  // Compute metric from eigendecomposition
  A = Q * D_array.matrix().asDiagonal() * Q.transpose();
}
"""

metric_from_hessian_str = """
#include <Eigen/Dense>

using namespace Eigen;

void metric_from_hessian(double A_[%d], const double * B_) {

  // Map inputs and outputs onto Eigen objects
  Map<Matrix<double, %d, %d, RowMajor> > A((double *)A_);
  Map<Matrix<double, %d, %d, RowMajor> > B((double *)B_);

  // Compute mean diagonal and set values appropriately
  double mean_diag;
  int i,j;
  for (i=0; i<%d-1; i++) {
    for (j=i+1; i<%d; i++) {
      mean_diag = 0.5*(B(i,j) + B(j,i));
      B(i,j) = mean_diag;
      B(j,i) = mean_diag;
    }
  }

  // Solve eigenvalue problem
  SelfAdjointEigenSolver<Matrix<double, %d, %d, RowMajor>> eigensolver(B);
  Matrix<double, %d, %d, RowMajor> Q = eigensolver.eigenvectors();
  Vector%dd D = eigensolver.eigenvalues();

  // Take modulus of eigenvalues
  for (i=0; i<%d; i++) D(i) = fmax(1e-10, abs(D(i)));

  // Build metric from eigendecomposition
  A += Q * D.asDiagonal() * Q.transpose();
}
"""

postproc_metric_str = """
#include <Eigen/Dense>

using namespace Eigen;

void postproc_metric(double A_[%d])
{

  // Map input/output metric onto an Eigen object
  Map<Matrix<double, %d, %d, RowMajor> > A((double *)A_);

  // Solve eigenvalue problem
  SelfAdjointEigenSolver<Matrix<double, %d, %d, RowMajor>> eigensolver(A);
  Matrix<double, %d, %d, RowMajor> Q = eigensolver.eigenvectors();
  Vector%dd D = eigensolver.eigenvalues();

  // Scale eigenvalues appropriately
  int i;
  double max_eig = 0.0;
  for (i=0; i<%d; i++) {
    D(i) = fmin(pow(%f, -2), fmax(pow(%f, -2), abs(D(i))));
    max_eig = fmax(max_eig, D(i));
  }
  for (i=0; i<%d; i++) D(i) = fmax(D(i), pow(%f, -2) * max_eig);

  // Build metric from eigendecomposition
  A = Q * D.asDiagonal() * Q.transpose();
}
"""

gemv_str = """
#include <Eigen/Dense>

using namespace Eigen;

void gemv(double y_[%d], const double * A_, const double * x_) {

  // Map inputs and outputs onto Eigen objects
  Map<Vector%dd> y((double *)y_);
  Map<Matrix<double, %d, %d, RowMajor> > A((double *)A_);
  Map<Vector%dd> x((double *)x_);
  double alpha = %f;
  double beta = %f;
  double tol = %f;

  // Apply generalised matrix-vector multiplication
  if (fabs(beta) < tol) y *= beta;
  if (fabs(alpha-1.0) < tol) {
    y += A * x;
  } else {
    y += alpha * A * x;
  }
}
"""

matscale_str = """
#include <Eigen/Dense>

using namespace Eigen;

void matscale(double B_[%d], const double * A_, const double * alpha_) {

  // Map inputs and outputs onto Eigen objects
  Map<Matrix<double, %d, %d, RowMajor> > A((double *)A_);
  Map<Matrix<double, %d, %d, RowMajor> > B((double *)B_);

  // Scale metric
  B += *alpha_ * A;
}
"""

poldec_spd_str = """
#include <Eigen/Dense>

using namespace Eigen;

void poldec_spd(double A_[%d], const double * B_) {

  // Map inputs and outputs onto Eigen objects
  Map<Matrix<double, %d, %d, RowMajor> > A((double *)A_);
  Map<Matrix<double, %d, %d, RowMajor> > B((double *)B_);

  // Compute singular value decomposition
  JacobiSVD<Matrix<double, %d, %d, RowMajor> > svd(B, ComputeFullV);

  // Get SPD part of polar decomposition
  A += svd.matrixV() * svd.singularValues().asDiagonal() * svd.matrixV().transpose();
}"""

poldec_unitary_str = """
#include <Eigen/Dense>

using namespace Eigen;

void poldec_unitary(double A_[%d], const double * B_) {

  // Map inputs and outputs onto Eigen objects
  Map<Matrix<double, %d, %d, RowMajor> > A((double *)A_);
  Map<Matrix<double, %d, %d, RowMajor> > B((double *)B_);

  // Compute singular value decomposition
  JacobiSVD<Matrix<double, %d, %d, RowMajor> > svd(B, ComputeFullU | ComputeFullV);

  // Get unitary part of polar decomposition
  A += svd.matrixU() * svd.matrixV().transpose();
}"""

get_max_length_edge_2d_str = """
for (int i=0; i<max_vector.dofs; i++) {
  int max_index = 0;
  if (edges[1] > edges[max_index]) max_index = 1;
  if (edges[2] > edges[max_index]) max_index = 2;
  max_vector[0] = vectors[2*max_index];
  max_vector[1] = vectors[2*max_index+1];
}
"""


# --- Python interpreters for C++ kernels

def eigen_kernel(kernel, *args, **kwargs):
    """
    Helper function to easily pass Eigen kernels to Firedrake via PyOP2.
    """
    return op2.Kernel(kernel(*args, **kwargs), kernel.__name__, cpp=True, include_dirs=include_dir)


def get_eigendecomposition(d):
    """
    Extract eigenvectors/eigenvalues from a metric field.

    If you care about the order, use `get_reordered_eigendecomposition`.
    """
    return get_eigendecomposition_str % (d*d, d, d, d, d, d, d, d, d)


def get_reordered_eigendecomposition(d):
    """
    Extract eigenvectors/eigenvalues from a metric field, with eigenvalues
    **decreasing** in magnitude.
    """
    assert d in (2, 3)
    return get_reordered_eigendecomp_2d_str if d == 2 else get_reordered_eigendecomp_3d_str


def set_eigendecomposition(d):
    """
    Compute a metric from eigenvectors and eigenvalues as an
    orthogonal eigendecomposition.
    """
    return set_eigendecomposition_str % (d*d, d, d, d, d, d)


def set_eigendecomposition_transpose(d):
    """
    Compute metric from transposed eigenvectors and eigenvalues
    as an orthogonal eigendecomposition.
    """
    return set_eigendecomposition_transpose_str % (d*d, d, d, d, d, d)


def intersect(d):
    """
    Intersect two metric fields.
    """
    return intersect_str % (d*d, d, d, d, d, d, d, d, d, d, d, d, d, d, d, d, d, d, d)


def anisotropic_refinement(d, direction):
    """
    Refine a metric in a single coordinate direction.
    """
    return anisotropic_refinement_str % (d*d, d, d, d, d, d, d, d, d, direction, 4)


def metric_from_hessian(d):
    """
    Build a metric field from a Hessian.
    """
    return metric_from_hessian_str % (d*d, d, d, d, d, d, d, d, d, d, d, d, d)


def postproc_metric(d, **kwargs):
    """
    Post-process a metric field in order to enforce max/min element sizes and anisotropy.
    """
    from adapt_utils.options import Options
    op = kwargs.get('op', Options())
    return postproc_metric_str % (d*d, d, d, d, d, d, d, d, d, op.h_min, op.h_max, d, op.max_anisotropy)


def gemv(d, alpha=1.0, beta=0.0, tol=1e-8):
    """
    Generalised matrix-vector multiplication of matrix A and vector x:

      y = alpha * A * x + beta * y.
    """
    return gemv_str % (d, d, d, d, d, alpha, beta, tol)


def matscale(d):
    """
    Multiply a matrix by a scalar field.
    """
    return matscale_str % (d*d, d, d, d, d)


def poldec_unitary(d):
    """
    Compute the unitary part of the polar decomposition of a matrix.
    """
    return poldec_unitary_str % (d*d, d, d, d, d, d, d)


def poldec_spd(d):
    """
    Compute the SPD part of the polar decomposition of a matrix.
    """
    return poldec_spd_str % (d*d, d, d, d, d, d, d)


def get_maximum_length_edge(d):
    """
    Find the mesh edge with maximum length.
    """
    assert d == 2
    return get_max_length_edge_2d_str
