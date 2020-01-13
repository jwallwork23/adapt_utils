from firedrake import *
try:
    import firedrake.cython.dmplex as dmplex
except ImportError:
    import firedrake.dmplex as dmplex
from firedrake.petsc import PETSc

import numpy as np
import numpy.linalg as la

from adapt_utils.adapt.kernels import eigen_kernel, get_eigendecomposition


__all__ = ["check_spd", "index_string", "subdomain_indicator", "get_boundary_nodes", "print_doc",
           "bessi0", "bessk0"]


def check_spd(matrix):
    """
    Verify that a tensor field `matrix` is symmetric positive-definite (SPD) and hence a Riemannian
    metric.
    """
    fs = matrix.function_space()

    # Check symmetric
    diff = interpolate(matrix - transpose(matrix), fs)
    try:
        assert norm(diff) < 1e-8
    except AssertionError:
        raise ValueError("Matrix is not symmetric!")

    # Check positive definite
    el = fs.ufl_element()
    evecs = Function(fs)
    evals = Function(VectorFunctionSpace(fs.mesh(), el.family(), el.degree()))
    dim = matrix.function_space().mesh().topological_dimension()
    kernel = eigen_kernel(get_eigendecomposition, dim)
    op2.par_loop(kernel, matrix.function_space().node_set, evecs.dat(op2.RW), evals.dat(op2.RW), matrix.dat(op2.READ))
    try:
        assert evals.vector().gather().min() > 0.0
    except AssertionError:
        raise ValueError("Matrix is not positive definite!")
    PETSc.Sys.Print("Matrix is indeed SPD.\n")

def index_string(index):
    """
    :arg index: integer form of index.
    :return: five-digit string form of index.
    """
    return (5 - len(str(index)))*'0' + str(index)

def subdomain_indicator(mesh, subdomain_id):
    """
    Creates a P0 indicator function relating with `subdomain_id`.
    """
    return assemble(TestFunction(FunctionSpace(mesh, "DG", 0))*dx(subdomain_id))

def get_boundary_nodes(fs, segment='on_boundary'):
    """
    :arg fs: function space to get boundary nodes for.
    :kwarg segment: segment of boundary to get nodes of (default 'on_boundary').
    """
    return fs.boundary_nodes(segment, 'topological')

def print_doc(anything):
    """
    Print the docstring of any class or function.
    """
    print(anything.__doc__)

def bessi0(x):
    """
    Modified Bessel function of the first kind. Code taken from 'Numerical recipes in C'.
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
    Modified Bessel function of the second kind. Code taken from 'Numerical recipes in C'.
    """
    y1 = x*x/4.0
    expr1 = (-ln(x/2.0)*bessi0(x)) + (-0.57721566 + y1*(0.42278420 + y1*(0.23069756 + y1*(0.3488590e-1 + y1*(0.262698e-2 + y1*(0.10750e-3 + y1*0.74e-5))))))
    y2 = 2.0/x
    expr2 = (exp(-x)/sqrt(x))*(1.25331414 + y2*(-0.7832358e-1 + y2*(0.2189568e-1 + y2*(-0.1062446e-1 + y2*(0.587872e-2 + y2*(-0.251540e-2 + y2*0.53208e-3))))))
    return conditional(ge(x, 2), expr2, expr1)
