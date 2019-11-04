from firedrake import *

# TODO: use SLEPc
import numpy as np
import numpy.linalg as la
import scipy.sparse as sp
import scipy.sparse.linalg as sla


__all__ = ["UnnestedConditionCheck", "NestedConditionCheck"]


class BaseConditionCheck():

    def __init__(self, bilinear_form):
        self.a = bilinear_form
        self.A = assemble(bilinear_form).M.handle

    def convert_dense(self, *args):
        pass

    def condition_number(self, *args):
        pass


class UnnestedConditionCheck(BaseConditionCheck):

    def __init__(self, bilinear_form):
        super(UnnestedConditionCheck, self).__init__(bilinear_form)
        try:
            assert self.A.getType() != 'nest'
        except:
            raise ValueError("Matrix type 'nest' not supported. Use `NestedConditionCheck` instead.")
        self.m = self.n = 1

    def condition_number(self, eps=1e-10):
        indptr, indices, data = self.A.getValuesCSR()
        A_sparse = sp.csr_matrix((data, indices, indptr), shape=self.A.getSize())
        try:
            eigval = sla.eigs(A_sparse)[0]
        except:
            eigval = la.eig(A_sparse)[0]
        eigval = np.where(np.abs(eigval) < eps, eps, np.abs(eigval))
        return np.max(eigval)/np.min(eigval)


class NestedConditionCheck(BaseConditionCheck):

    def __init__(self, bilinear_form):
        super(NestedConditionCheck, self).__init__(bilinear_form)
        try:
            assert self.A.getType() == 'nest'
        except:
            raise ValueError("Matrix type {:} not supported.".format(self.A.getType()))
        self.m, self.n = self.A.getNestSize()
        self.submatrices = {}
        self.dense_submatrices = {}
        for i in range(self.m):
            self.dense_submatrices[i] = {}
            self.submatrices[i] = {}
            for j in range(self.n):
                self.submatrices[i][j] = self.A.getNestSubMatrix(i, j)

    def condition_number(self, i, j, eps=1e-10):
        indptr, indices, data = self.submatrices[i][j].getValuesCSR()
        size = self.submatrices[i][j].getSize()
        A_sparse = sp.csr_matrix((data, indices, indptr), shape=size)
        try:
            eigval = sla.eigs(A_sparse)[0]
        except:
            eigval = la.eig(A_sparse)[0]
        eigval = np.where(np.abs(eigval) < eps, eps, np.abs(eigval))
        return np.max(eigval)/np.min(eigval)
