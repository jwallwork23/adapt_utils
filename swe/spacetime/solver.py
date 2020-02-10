from firedrake import *

import os
import numpy as np

from adapt_utils.solver import SteadyProblem


__all__ = ["SpaceTimeShallowWaterProblem"]


class SpaceTimeShallowWaterProblem(SteadyProblem):
    """
    Class for solving shallow water problems discretised using space-time FEM.
    """
    def __init__(self, op, mesh=None, discrete_adjoint=True, prev_solution=None, levels=0):
        p = op.degree
        if op.family == 'taylor-hood' and p > 0:
            fe = VectorElement("CG", triangle, p+1)*FiniteElement("CG", triangle, p)
        else:
            raise NotImplementedError  # TODO
        super(SpaceTimeShallowWaterProblem, self).__init__(op, mesh, fe, discrete_adjoint, prev_solution, levels)
        if prev_solution is not None:
            self.interpolate_solution(prev_solution)

        # Parameters for adjoint computation
        z, zeta = self.adjoint_solution.split()
        z.rename("Adjoint fluid velocity")
        zeta.rename("Adjoint elevation")

        # Classification
        self.nonlinear = False

    def set_fields(self):
        self.fields = {}
        # self.fields['viscosity'] = self.op.set_viscosity(self.P1)
        self.fields['bathymetry'] = self.op.set_bathymetry(self.P1DG)
        self.fields['coriolis'] = self.op.set_coriolis(self.P1)

    def set_stabilisation(self):
        self.stabilisation = self.stabilisation or 'no'
        try:
            assert self.stabilisation == 'no'
        except AssertionError:
            raise NotImplementedError  # TODO

    def setup_solver(self):
        raise NotImplementedError  # TODO

    def solve_forward(self):
        raise NotImplementedError  # TODO
