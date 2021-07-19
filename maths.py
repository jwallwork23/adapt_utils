from ufl import *
import numpy as np


__all__ = ["monomials", "bessi0", "bessk0"]


def monomials(x, order=1):
    """
    Return Vandermonde matrix. In 2D, with order p, :math:`V = [1, x, y]`.
    """
    assert len(x) in (2, 3)
    if order == 1:
        return np.array([1.0, *x])
    elif order == 2:
        if len(x) == 2:
            return np.array([1.0, x[0], x[1], x[0]*x[0], x[0]*x[1], x[1]*x[1]])
    else:
        raise NotImplementedError


def bessi0(x):
    """
    Zeroth order modified Bessel function of the first kind.

    Code taken from 'Numerical recipes in C'.
    """
    ax = abs(x)
    y1 = x/3.75
    y1 *= y1
    expr1 = 1.0 + y1*(3.5156229 + y1*(3.0899424 + y1*(1.2067492 + y1*(0.2659732 + y1*(0.360768e-1 + y1*0.45813e-2)))))
    y2 = 3.75/ax
    expr2 = (exp(ax)/sqrt(ax))*(0.39894228 + y2*(0.1328592e-1 + y2*(0.225319e-2 + y2*(-0.157565e-2 + y2*(0.916281e-2 + y2*(-0.2057706e-1 + y2*(0.2635537e-1 + y2*(-0.1647633e-1 + y2*0.392377e-2))))))))
    return conditional(le(ax, 3.75), expr1, expr2)


def bessk0(x):
    """
    Zeroth order modified Bessel function of the second kind.

    Code taken from 'Numerical recipes in C'.
    """
    y1 = x*x/4.0
    expr1 = (-ln(x/2.0)*bessi0(x)) + (-0.57721566 + y1*(0.42278420 + y1*(0.23069756 + y1*(0.3488590e-1 + y1*(0.262698e-2 + y1*(0.10750e-3 + y1*0.74e-5))))))
    y2 = 2.0/x
    expr2 = (exp(-x)/sqrt(x))*(1.25331414 + y2*(-0.7832358e-1 + y2*(0.2189568e-1 + y2*(-0.1062446e-1 + y2*(0.587872e-2 + y2*(-0.251540e-2 + y2*0.53208e-3))))))
    return conditional(le(x, 2), expr1, expr2)


def bessi1(x):
    """
    First order modified Bessel function of the first kind.

    Code taken from 'Numerical recipes in C'.
    """
    ax = abs(x)
    y1 = x/3.75
    y1 *= y1
    expr1 = ax*(0.5 + y1*(0.87890594 + y1*(0.51498869 + y1*(0.15084934 + y1*(0.02658733 + y1*(0.00301532 + y1*(0.00032411)))))))
    y2 = 3.75/ax
    expr2 = 0.02282967 + y2*(-0.02895312 + y2*(0.01787654 - y2*0.00420059))
    expr2 = 0.39894228 + y2*(-0.03988024 + y2*-0.00362018 + y2*(0.00163801 + y2*(-0.01031555 + y2*expr2)))
    expr2 *= exp(ax)/sqrt(ax)
    return conditional(le(ax, 3.75), expr1, expr2)


def bessk1(x):
    """
    Zeroth order modified Bessel function of the second kind.

    Code taken from 'Numerical recipes in C'.
    """
    y1 = x*x/4.0
    expr1 = log(x/2.0)*bessi1(x) + (1.0/x)*(1.0 + y1*(0.15443144 + y1*(-0.67278579 + y1*(-0.18156897 + y1*(-0.01919402 + y1*(-0.0010404 - y1*0.00004686))))))
    y2 = 2.0/x
    expr2 = (exp(-x)/sqrt(x))*(1.25331414 + y2*(0.23498619 + y2*(-0.03655620 + y2*(0.01504268 + y2*(-0.00780353 + y2*(0.00325614 - y2*0.00068245))))))
    return conditional(le(x, 2), expr1, expr2)
