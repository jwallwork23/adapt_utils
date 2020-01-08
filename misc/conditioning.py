from firedrake import *
from firedrake.petsc import PETSc

import numpy as np
import numpy.linalg as la
import scipy.sparse as sp
import scipy.sparse.linalg as sla


__all__ = ["UnnestedConditionCheck", "NestedConditionCheck"]


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
                PETSc.Sys.Print("Mass matrix is positive-definite")
            else:
                PETSc.Sys.Print("Mass matrix is not positive-definite. Minimal eigenvalue: {:.4e}".format(eigmin))
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
            PETSc.Sys.Print("Solving eigenvalue problem using SLEPc...")
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
                PETSc.Sys.Print("Mass matrix is positive-definite")
            else:
                PETSc.Sys.Print("Mass matrix is not positive-definite. Minimal eigenvalue: {:.4e}".format(eigmin))
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
