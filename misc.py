from thetis import *
from firedrake.petsc import PETSc

import os
import fnmatch
import numpy as np
import numpy.linalg as la
import scipy.sparse as sp
import scipy.sparse.linalg as sla

from adapt_utils.adapt.kernels import eigen_kernel, get_eigendecomposition


__all__ = ["doc", "find", "check_spd", "get_boundary_nodes", "index_string",
           "UnnestedConditionCheck", "NestedConditionCheck"]


def doc(anything):
    """
    Print the docstring of any class or function.
    """
    print_output(anything.__doc__)

def find(pattern, path):
    """Find all files with a specified pattern."""
    result = []
    for root, dirs, files in os.walk(path):
        for name in files:
            if fnmatch.fnmatch(name, pattern):
                result.append(os.path.join(root, name))
    return result

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
    print_output("Matrix is indeed SPD.")

def index_string(index):
    """
    :arg index: integer form of index.
    :return: five-digit string form of index.
    """
    return (5 - len(str(index)))*'0' + str(index)

def get_boundary_nodes(fs, segment='on_boundary'):
    """
    :arg fs: function space to get boundary nodes for.
    :kwarg segment: segment of boundary to get nodes of (default 'on_boundary').
    """
    return fs.boundary_nodes(segment, 'topological')

class BaseConditionCheck():

    def __init__(self, bilinear_form, mass_term=None):
        self.a = bilinear_form
        self.A = assemble(bilinear_form).M.handle
        if mass_term is not None:
            self.M = assemble(mass_term).M.handle

    def _get_wrk(self, *args):
        pass

    def _get_csr(self):
        indptr, indices, data = self.wrk.getValuesCSR()
        self.csr = sp.csr_matrix((data, indices, indptr), shape=self.wrk.getSize())

    def condition_number(self, *args, eps=1e-10):
        self._get_wrk(*args)
        try:
            from slepc4py import SLEPc
        except ImportError:
            # If SLEPc is not available, use SciPy
            self._get_csr()
            eigval = sla.eigs(self.csr)[0]
            eigmin = np.min(eigval)
            if eigmin > 0.0:
                print_output("Mass matrix is positive-definite")
            else:
                print_output("Mass matrix is not positive-definite. Minimal eigenvalue: {:.4e}".format(eigmin))
            eigval = np.abs(eigval)
            return np.max(eigval)/max(np.min(eigval), eps)

        if hasattr(self, 'M'):
            raise NotImplementedError  # TODO
        else:
            # Solve eigenvalue problem
            n = self.wrk.getSize()[0]
            es = SLEPc.EPS().create(comm=COMM_WORLD)
            es.setDimensions(n)
            es.setOperators(self.wrk)
            opts = PETSc.Options()
            opts.setValue('eps_type', 'krylovschur')
            opts.setValue('eps_monitor', None)
            # opts.setValue('eps_pc_type', 'lu')
            # opts.setValue('eps_pc_factor_mat_solver_type', 'mumps')
            es.setFromOptions()
            print_output("Solving eigenvalue problem using SLEPc...")
            es.solve()

            # Check convergence
            nconv = es.getConverged()
            if nconv == 0:
                import sys
                warning("Did not converge any eigenvalues")
                sys.exit(0)

            # Compute condition number
            vr, vi = self.wrk.getVecs()
            eigmax = es.getEigenpair(0, vr, vi).real
            eigmin = es.getEigenpair(n-1, vr, vi).real
            eigmin = max(eigmin, eps)
            if eigmin > 0.0:
                print_output("Mass matrix is positive-definite")
            else:
                print_output("Mass matrix is not positive-definite. Minimal eigenvalue: {:.4e}".format(eigmin))
            return np.abs(eigmax/eigmin)


class UnnestedConditionCheck(BaseConditionCheck):

    def __init__(self, bilinear_form):
        super(UnnestedConditionCheck, self).__init__(bilinear_form)
        try:
            assert self.A.getType() != 'nest'
        except AssertionError:
            raise ValueError("Matrix type 'nest' not supported. Use `NestedConditionCheck` instead.")

    def _get_wrk(self):
        self.wrk = self.A


class NestedConditionCheck(BaseConditionCheck):

    def __init__(self, bilinear_form):
        super(NestedConditionCheck, self).__init__(bilinear_form)
        try:
            assert self.A.getType() == 'nest'
        except AssertionError:
            raise ValueError("Matrix type {:} not supported.".format(self.A.getType()))
        m, n = self.A.getNestSize()
        self.submatrices = {}
        for i in range(m):
            self.submatrices[i] = {}
            for j in range(n):
                self.submatrices[i][j] = self.A.getNestSubMatrix(i, j)

    def _get_wrk(self, i, j):
        self.wrk = self.submatrices[i][j]
