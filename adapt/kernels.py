try:
    from firedrake.slate.slac.compiler import PETSC_ARCH
except:
    import os

    PETSC_ARCH = os.environ.get('PETSC_ARCH')
    PETSC_DIR = os.environ.get('PETSC_DIR')
    PETSC_ARCH = os.path.join(PETSC_DIR, PETSC_ARCH)
    if not os.path.exists(os.path.join(PETSC_ARCH, 'include/eigen3')):
        PETSC_ARCH = '/usr/local'

from adapt_utils.options import * 


__all__ = ["mykernel", "get_eigendecomposition_kernel", "get_reordered_eigendecomposition_kernel",
           "set_eigendecomposition_kernel", "intersect_kernel", "anisotropic_refinement_kernel",
           "metric_from_hessian_kernel", "scale_metric_kernel", "include_dir",
           "gemv_kernel", "matscale_kernel", "matscale_sum_kernel", "polar_kernel", ]


include_dir = ["%s/include/eigen3" % PETSC_ARCH]


def mykernel(kernel, name):
    return op2.Kernel(kernel, name, cpp=True, include_dirs=include_dir)

def get_eigendecomposition_kernel(d):
    return """
#include <Eigen/Dense>

using namespace Eigen;

void get_eigendecomposition(double EVecs_[%d], double EVals_[%d], const double * M_) {
  Map<Matrix<double, %d, %d, RowMajor> > EVecs((double *)EVecs_);
  Map<Vector%dd> EVals((double *)EVals_);
  Map<Matrix<double, %d, %d, RowMajor> > M((double *)M_);
  SelfAdjointEigenSolver<Matrix<double, %d, %d, RowMajor>> eigensolver(M);
  Matrix<double, %d, %d, RowMajor> Q = eigensolver.eigenvectors();
  Vector%dd D = eigensolver.eigenvalues();
  EVecs = Q;
  EVals = D;
}
""" % (d*d, d, d, d, d, d, d, d, d, d, d, d)

get_reordered_eigendecomposition_kernel_2d = """
#include <Eigen/Dense>

using namespace Eigen;

void get_reordered_eigendecomposition(double EVecs_[4], double EVals_[2], const double * M_) {
  Map<Matrix<double, 2, 2, RowMajor> > EVecs((double *)EVecs_);
  Map<Vector2d> EVals((double *)EVals_);
  Map<Matrix<double, 2, 2, RowMajor> > M((double *)M_);
  SelfAdjointEigenSolver<Matrix<double, 2, 2, RowMajor>> eigensolver(M);
  Matrix<double, 2, 2, RowMajor> Q = eigensolver.eigenvectors().transpose();
  Vector2d D = eigensolver.eigenvalues();
  if (fabs(D(0)) > fabs(D(1))) {
    EVecs(0,0) = Q(1,0);
    EVecs(0,1) = Q(1,1);
    EVecs(1,0) = Q(0,0);
    EVecs(1,1) = Q(0,1);
    EVals(0) = D(0);
    EVals(1) = D(1);
  } else {
    EVecs(0,0) = Q(0,0);
    EVecs(0,1) = Q(0,1);
    EVecs(1,0) = Q(1,0);
    EVecs(1,1) = Q(1,1);
    EVals(0) = D(1);
    EVals(1) = D(0);
  }
}
"""

def get_reordered_eigendecomposition_kernel(d):
    if d == 2:
        return get_reordered_eigendecomposition_kernel_2d
    else:
        raise NotImplementedError  # TODO: 3d case

def set_eigendecomposition_kernel(d):
    return """
#include <Eigen/Dense>

using namespace Eigen;

void set_eigendecomposition(double M_[%d], const double * EVecs_, const double * EVals_) {
  Map<Matrix<double, %d, %d, RowMajor> > M((double *)M_);
  Map<Matrix<double, %d, %d, RowMajor> > EVecs((double *)EVecs_);
  Map<Vector%dd> EVals((double *)EVals_);
  M = EVecs.transpose() * EVals.asDiagonal() * EVecs;
}
""" % (d*d, d, d, d, d, d)

def intersect_kernel(d):
    return """
#include <Eigen/Dense>

using namespace Eigen;

void intersect(double M_[%d], const double * A_, const double * B_) {
  Map<Matrix<double, %d, %d, RowMajor> > M((double *)M_);
  Map<Matrix<double, %d, %d, RowMajor> > A((double *)A_);
  Map<Matrix<double, %d, %d, RowMajor> > B((double *)B_);
  SelfAdjointEigenSolver<Matrix<double, %d, %d, RowMajor>> eigensolver(A);
  Matrix<double, %d, %d, RowMajor> Q = eigensolver.eigenvectors();
  Matrix<double, %d, %d, RowMajor> D = eigensolver.eigenvalues().array().sqrt().matrix().asDiagonal();
  Matrix<double, %d, %d, RowMajor> Sq = Q * D * Q.transpose();
  Matrix<double, %d, %d, RowMajor> Sqi = Q * D.inverse() * Q.transpose();
  SelfAdjointEigenSolver<Matrix<double, %d, %d, RowMajor>> eigensolver2(Sqi.transpose() * B * Sqi);
  Q = eigensolver2.eigenvectors();
  D = eigensolver2.eigenvalues().array().max(1).matrix().asDiagonal();
  M = Sq.transpose() * Q * D * Q.transpose() * Sq;
}
""" % (d*d, d, d, d, d, d, d, d, d, d, d, d, d, d, d, d, d, d, d)

def anisotropic_refinement_kernel(d, direction):
    assert d in (2, 3)
    scale = 4 if d == 2 else 8
    return """
#include <Eigen/Dense>

using namespace Eigen;

void anisotropic(double A_[%d]) {
  Map<Matrix<double, %d, %d, RowMajor> > A((double *)A_);
  SelfAdjointEigenSolver<Matrix<double, %d, %d, RowMajor>> eigensolver(A);
  Matrix<double, %d, %d, RowMajor> Q = eigensolver.eigenvectors();
  Vector%dd D = eigensolver.eigenvalues();
  Array%dd D_array = D.array();
  D_array(%d) *= %f;
  A = Q * D_array.matrix().asDiagonal() * Q.transpose();
}
""" % (d*d, d, d, d, d, d, d, d, d, direction, scale)

def metric_from_hessian_kernel(d, noscale=False, op=Options()):
    p = op.norm_order
    scale = 'false' if noscale or op.normalisation == 'error' else 'true'
    if p is None:
        return linf_metric_from_hessian_kernel(d, scale)
    else:
        return lp_metric_from_hessian_kernel(d, scale, p)

def linf_metric_from_hessian_kernel(d, scale):
    return """
#include <Eigen/Dense>

using namespace Eigen;

void metric_from_hessian(double A_[%d], double * f, const double * B_)
{
  Map<Matrix<double, %d, %d, RowMajor> > A((double *)A_);
  Map<Matrix<double, %d, %d, RowMajor> > B((double *)B_);

  double mean_diag;
  int i,j;
  for (i=0; i<%d-1; i++) {
    for (j=i+1; i<%d; i++) {
      mean_diag = 0.5*(B(i,j) + B(j,i));
      B(i,j) = mean_diag;
      B(j,i) = mean_diag;
    }
  }

  SelfAdjointEigenSolver<Matrix<double, %d, %d, RowMajor>> eigensolver(B);
  Matrix<double, %d, %d, RowMajor> Q = eigensolver.eigenvectors();
  Vector%dd D = eigensolver.eigenvalues();

  /* Normalisation */
  for (i=0; i<%d; i++) D(i) = fmax(1e-10, abs(D(i)));
  if (%s) {
    double det = 1.0;
    for (i=0; i<%d; i++) det *= D(i);
    *f += sqrt(det);
  }
  A += Q * D.asDiagonal() * Q.transpose();
}
""" % (d*d, d, d, d, d, d, d, d, d, d, d, d, d, scale, d)

def lp_metric_from_hessian_kernel(d, scale, p):
    return """
#include <Eigen/Dense>

using namespace Eigen;

void metric_from_hessian(double A_[%d], double * f, const double * B_)
{
  Map<Matrix<double, %d, %d, RowMajor> > A((double *)A_);
  Map<Matrix<double, %d, %d, RowMajor> > B((double *)B_);

  double mean_diag;
  int i,j;
  for (i=0; i<%d-1; i++) {
    for (j=i+1; i<%d; i++) {
      mean_diag = 0.5*(B(i,j) + B(j,i));
      B(i,j) = mean_diag;
      B(j,i) = mean_diag;
    }
  }

  SelfAdjointEigenSolver<Matrix<double, %d, %d, RowMajor>> eigensolver(B);
  Matrix<double, %d, %d, RowMajor> Q = eigensolver.eigenvectors();
  Vector%dd D = eigensolver.eigenvalues();

  /* Normalisation */
  for (i=0; i<%d; i++) D(i) = fmax(1e-10, abs(D(i)));
  double scaling = 1.0;
  if (%s) {
    double det = 1.0;
    for (i=0; i<%d; i++) det *= D(i);
    scaling = pow(det, -1 / (2 * %d + 2));
    *f += pow(det, %d / (2 * %d + 2));
  }
  A += scaling * Q * D.asDiagonal() * Q.transpose();
}
""" % (d*d, d, d, d, d, d, d, d, d, d, d, d, d, scale, d, p, p, p)

def scale_metric_kernel(d, op=Options()):
    return """
#include <Eigen/Dense>

using namespace Eigen;

void scale_metric(double A_[%d])
{
  Map<Matrix<double, %d, %d, RowMajor> > A((double *)A_);
  SelfAdjointEigenSolver<Matrix<double, %d, %d, RowMajor>> eigensolver(A);
  Matrix<double, %d, %d, RowMajor> Q = eigensolver.eigenvectors();
  Vector%dd D = eigensolver.eigenvalues();
  for (int i=0; i<%d; i++) D(i) = fmin(%f, fmax(%f, abs(D(i))));
  double max_eig = fmax(D(0), D(1));
  if (%d == 3) max_eig = fmax(max_eig, D(2));
  for (int i=0; i<%d; i++) D(i) = fmax(D(i), %f * max_eig);
  A = Q * D.asDiagonal() * Q.transpose();
}
""" % (d*d, d, d, d, d, d, d, d, d, pow(op.h_min, -2), pow(op.h_max, -2), d, d, pow(op.max_anisotropy, -2))

def gemv_kernel(d, alpha=1.0, beta=0.0, tol=1e-8):
    return """
#include <Eigen/Dense>

using namespace Eigen;

void gemv(double y_[%d], const double * A_, const double * x_) {
  Map<Vector%dd> y((double *)y_);
  Map<Matrix<double, %d, %d, RowMajor> > A((double *)A_);
  Map<Vector%dd> x((double *)x_);
  double alpha = %f;
  double beta = %f;
  double tol = %f;

  if (fabs(beta) < tol) y *= beta;
  if (fabs(alpha-1.0) < tol) {
    y += A * x;
  } else {
    y += alpha * A * x;
  }
}
""" % (d, d, d, d, d, alpha, beta, tol)

def matscale_kernel(d):
    return """
#include <Eigen/Dense>

using namespace Eigen;

void matscale(double B_[%d], const double * A_, const double * alpha_) {
  Map<Matrix<double, %d, %d, RowMajor> > A((double *)A_);
  Map<Matrix<double, %d, %d, RowMajor> > B((double *)B_);
  B += *alpha_ * A;
}
""" % (d*d, d, d, d, d)

def matscale_sum_kernel(d):
    if d == 2:
        return """
#include <Eigen/Dense>

using namespace Eigen;

void matscale_sum(double B_[4], const double * A1_, const double * A2_, const double * x_) {
  Map<Matrix<double, 2, 2, RowMajor> > A1((double *)A1_);
  Map<Matrix<double, 2, 2, RowMajor> > A2((double *)A2_);
  Map<Matrix<double, 2, 2, RowMajor> > B((double *)B_);
  Map<Vector2d> x((double *)x_);
  B += x[0] * A1;
  B += x[1] * A2;
}
"""
    elif d == 3:
        return """
#include <Eigen/Dense>

using namespace Eigen;

void matscale_sum(double B_[9], const double * A1_, const double * A2_, const double * A3_, const double * x_) {
  Map<Matrix<double, 3, 3, RowMajor> > A1((double *)A1_);
  Map<Matrix<double, 3, 3, RowMajor> > A2((double *)A2_);
  Map<Matrix<double, 3, 3, RowMajor> > A3((double *)A3_);
  Map<Matrix<double, 3, 3, RowMajor> > B((double *)B_);
  Map<Vector3d> x((double *)x_);
  B += x[0] * A1;
  B += x[1] * A2;
  B += x[2] * A3;
}
"""
    else:
        raise NotImplementedError

def polar_kernel(d):
    return """
#include <Eigen/Dense>

void polar(double A_[%d], const double * B_) {
  Eigen::Map<Eigen::Matrix<double, %d, %d, Eigen::RowMajor> > A((double *)A_);
  Eigen::Map<Eigen::Matrix<double, %d, %d, Eigen::RowMajor> > B((double *)B_);
  Eigen::JacobiSVD<Eigen::Matrix<double, %d, %d, Eigen::RowMajor> > svd(B, Eigen::ComputeFullV);

  A += svd.matrixV() * svd.singularValues().asDiagonal() * svd.matrixV().transpose();
}""" % (d*d, d, d, d, d, d, d)
