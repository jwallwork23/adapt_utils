from firedrake import *
import numpy as np


__all__ = ["index_string", "subdomain_indicator", "bessk0"]


def index_string(index):
    """
    :arg index: integer form of index.
    :return: five-digit string form of index.
    """
    return (5 - len(str(index))) * '0' + str(index)


def subdomain_indicator(mesh, subdomain_id):
    """
    Creates a P0 indicator function relating with `subdomain_id`.
    """
    return assemble(TestFunction(FunctionSpace(mesh, "DG", 0)) * dx(subdomain_id))

def bessi0(x):
    """
    Modified Bessel function of the first kind. Code taken from 'Numerical recipes in C'.
    """
    ax = abs(x)
    y1 = x/3.75
    y1 *= y1
    expr1 = 1.0 + y1*(3.5156229 + y1*(1.2067492 + y1*(0.2659732 + y1*(0.360768e-1 + y1*0.45813e-2))))
    y2 = 3.75/ax
    expr2 = (exp(ax)/sqrt(ax))*(0.39894228 + y2*(0.1328592e-1 + y2*(0.225319e-2 + y2*(-0.157565e-2 + y2*(0.916281e-2 + y2*(-0.2057706e-1 + y2*(0.2635537e-1 + y2*(-0.1647633e-1 + y2*0.392377e-2))))))))
    return conditional(le(ax, 3.75), expr1, expr2)

def bessk0(x):
    """
    Modified Bessel function of the second kind. Code taken from 'Numerical recipes in C'.
    """
    y1 = x*x/4.0
    expr1 = (-ln(x/2.0)*bessi0(x)) + (-0.57721566 + y1*(0.42278420 + y1*(0.23069756 + y1*(0.3488590e-1 + y1*(0.262698e-2 + y1*(0.10750e-3 + y1*0.74e-5))))))
    y2 = 2.0/x
    expr2 = (exp(-x)/sqrt(x))*(1.25331414 + y2*(-0.7832358e-1 + y2*(0.2189568e-1 + y2*(-0.1062446e-1 + y2*(0.587872e-2 + y2*(-0.251540e-2 + y2*0.53208e-3))))))
    return conditional(ge(x, 2), expr2, expr1)


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


def pointwise_max(f, g):
    r"""
    Take the pointwise maximum (in modulus) of Functions `f` and `g`.
    """
    fu = f.ufl_element()
    gu = g.ufl_element()
    try:
        assert fu == gu
    except:
        raise ValueError("Function space mismatch: ", fu, " vs. ", gu)
    h = Function(f.function_space()).assign(np.finfo(0.).min)
    kernel_str = "void maxval(double * z, double const * x, double const * y){*z = fmax(fabs(*x), fabs(*y));}"
    kernel = op2.Kernel(kernel_str, "maxval")
    op2.par_loop(kernel, f.function_space().node_set, h.dat(op2.RW), f.dat(op2.READ), g.dat(op2.READ))

    return h
