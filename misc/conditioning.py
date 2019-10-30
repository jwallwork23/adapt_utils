from firedrake import *

# TODO: use SLEPc
import numpy.linalg as la


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

    # TODO: do not convert to dense! Use getRow(i) and sparse linalg

    def convert_dense(self):
        self.A_dense = self.A.convert('dense')

    def condition_number(self):
        if not hasattr(self, 'A_dense'):
            self.convert_dense()
        return la.cond(self.A_dense)


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

    # TODO: do not convert to dense! Use getRow(i) and sparse linalg

    def convert_dense(self, i, j):
        self.dense_submatrices[i][j] = self.submatrices[i][j].convert('dense').getDenseArray()

    def condition_number(self, i, j):
        if not j in self.dense_submatrices[i]:
            self.convert_dense(i, j)
        return la.cond(self.dense_submatrices[i][j])
