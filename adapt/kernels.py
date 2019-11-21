from adapt_utils.options import DefaultOptions

__all__ = ["intersect_kernel", "anisotropic_refinement_kernel", "metric_from_hessian_kernel", "scale_metric_kernel"]

# TODO: For plugin, see https://github.com/jwallwork23/adapt_utils/blob/43a291fe3d2fdf7b64d0f7ebaecff7b7d150e641/adapt/metric.py
# NOTE: May need to use std::min and std::max

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
#include <iostream>

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

# TODO: Plugin and test
# TODO: 3d implementation
def metric_from_hessian_kernel(p, noscale=False, op=DefaultOptions()):
    if p is None:
        return """
#include <Eigen/Dense>
#include <algorithm>

using namespace Eigen;

void metric_from_hessian(double A_[4], double * f, const double * B_)
{
  Map<Matrix<double, 2, 2, RowMajor> > A((double *)A_);
  Map<Matrix<double, 2, 2, RowMajor> > B((double *)B_);
  double mean_diag = 0.5*(B(0,1) + B(1,0));
  B(0,1) = mean_diag;
  B(1,0) = mean_diag;
  SelfAdjointEigenSolver<Matrix<double, 2, 2, RowMajor>> eigensolver(B);
  Matrix<double, 2, 2, RowMajor> Q = eigensolver.eigenvectors();
  Vector2d D = eigensolver.eigenvalues();
  D(0) = fmax(1e-10, abs(D(0)));
  D(1) = fmax(1e-10, abs(D(1)));

  /* Normalisation */
  if (%s) {
    double det = D(0) * D(1);
    *f += sqrt(det);
  }
  A += Q * D.asDiagonal() * Q.transpose();
}
""" % ('false' if noscale or op.normalisation == 'complexity' else 'true')
    else:
        return """
#include <Eigen/Dense>
#include <algorithm>

using namespace Eigen;

void metric_from_hessian(double A_[4], double * f, const double * B_)
{
  Map<Matrix<double, 2, 2, RowMajor> > A((double *)A_);
  Map<Matrix<double, 2, 2, RowMajor> > B((double *)B_);
  double mean_diag = 0.5*(B(0,1) + B(1,0));
  B(0,1) = mean_diag;
  B(1,0) = mean_diag;
  SelfAdjointEigenSolver<Matrix<double, 2, 2, RowMajor>> eigensolver(B);
  Matrix<double, 2, 2, RowMajor> Q = eigensolver.eigenvectors();
  Vector2d D = eigensolver.eigenvalues();
  D(0) = fmax(1e-10, abs(D(0)));
  D(1) = fmax(1e-10, abs(D(1)));
  double scaling = 1.0;

  /* Normalisation */
  if (%s) {
    double det = D(0) * D(1);
    scaling = pow(det, -1 / (2 * %d + 2));
    *f += pow(det, %d / (2 * %d + 2));
  }
  A += scaling * Q * D.asDiagonal() * Q.transpose();
}
""" % (p, 'false' if noscale else 'true', p, p)

# TODO: Plugin and test
# TODO: 3d implementation
def scale_metric_kernel(op=DefaultOptions()):
    ia2 = pow(op.max_anisotropy, -2)
    ih_min2 = pow(op.h_min, -2)
    ih_max2 = pow(op.h_max, -2)
    return """
#include <Eigen/Dense>
#include <algorithm>

using namespace Eigen;

void metric2(double A_[4])
{
  Map<Matrix<double, 2, 2, RowMajor> > A((double *)A_);
  SelfAdjointEigenSolver<Matrix<double, 2, 2, RowMajor>> eigensolver(A);
  Matrix<double, 2, 2, RowMajor> Q = eigensolver.eigenvectors();
  Vector2d D = eigensolver.eigenvalues();
  D(0) = fmin(%f, fmax(%f, abs(D(0))));
  D(1) = fmin(%f, fmax(%f, abs(D(1))));
  double max_eig = max(D(0), D(1));
  D(0) = fmax(D(0), %f * max_eig);
  D(1) = fmax(D(1), %f * max_eig);
  A = Q * D.asDiagonal() * Q.transpose();
}
""" % (ih_min2, ih_max2, ih_min2, ih_max2, ia2, ia2)
