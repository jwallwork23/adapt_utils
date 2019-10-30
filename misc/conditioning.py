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
        assert self.A.getType != 'nest'
        self.m = self.n = 1

    def convert_dense(self):
        self.A_dense = self.A.convert('dense')

    def condition_number(self):
        if not hasattr(self, 'A_dense'):
            self.convert_dense()
        return la.cond(self.A_dense)


class NestedConditionCheck(BaseConditionCheck):

    def __init__(self, bilinear_form):
        super(NestedConditionCheck, self).__init__(bilinear_form)
        assert self.A.getType == 'nest'
        self.m, self.n = self.A.getNestSize()
        self.submatrices = {}
        self.dense_submatrices = {}
        for i in range(self.m):
            self.dense_submatrices[i] = {}
            for j in range(self.n):
                self.submatrices{i} = {j: self.A.getNestSubMatrix(i, j)}

    def convert_dense(self, i, j):
        self.dense_submatrices[i][j] = self.submatrices[i][j].convert('dense').getDenseArray()

    def condition_number(self, i, j):
        if self.dense_submatrices[i][j] == {}:
            self.convert_dense(i, j)
        return la.cond(self.dense_submatrices[i][j])
