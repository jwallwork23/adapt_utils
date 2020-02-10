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
        # TODO: FunctionSpace is currently hard-coded
        super(SpaceTimeShallowWaterProblem, self).__init__(op, mesh, None, discrete_adjoint, prev_solution, levels)
        try:
            assert self.mesh.topological_dimension() == 3
        except AssertionError:
            raise ValueError("We consider 2 spatial dimensions and 1 time.")

        # Apply initial condition
        self.set_start_condition(adjoint=False)

        # Classification
        self.nonlinear = False

    def create_function_spaces(self):
        """
        Build the finite element space, `V`, for the prognostic solution, along with various other
        useful spaces and test functions and trial functions based upon them.
        """
        self.V = VectorFunctionSpace(self.mesh, "CG", 2, dim=2)*FunctionSpace(self.mesh, "CG", 1)
        self.P0 = FunctionSpace(self.mesh, "DG", 0)
        self.P1 = FunctionSpace(self.mesh, "CG", 1)
        self.P2 = FunctionSpace(self.mesh, "CG", 2)
        self.P1DG = FunctionSpace(self.mesh, "DG", 1)
        self.P1_vec = VectorFunctionSpace(self.mesh, "CG", 1)
        self.P1_ten = TensorFunctionSpace(self.mesh, "CG", 1)
        self.test = TestFunction(self.V)
        self.tests = TestFunctions(self.V)
        self.trial = TrialFunction(self.V)
        self.trials = TrialFunctions(self.V)
        self.p0test = TestFunction(self.P0)
        self.p0trial = TrialFunction(self.P0)


    def set_fields(self):
        self.fields = {}
        # self.fields['viscosity'] = self.op.set_viscosity(self.P1)
        self.fields['bathymetry'] = self.op.set_bathymetry(self.P1)
        self.fields['coriolis'] = self.op.set_coriolis(self.P1)

    def set_stabilisation(self):
        self.stabilisation = self.stabilisation or 'no'
        try:
            assert self.stabilisation == 'no'
        except AssertionError:
            raise NotImplementedError  # TODO

    def set_start_condition(self, adjoint=False):
        self.set_solution(self.op.set_start_condition(self.V, adjoint=adjoint), adjoint)

    def setup_solver_forward(self):
        u, η = self.trials
        v, θ = self.tests

        # Parameters
        g = self.op.g
        f = self.fields['coriolis']
        b = self.fields['bathymetry']

        # Operators
        grad_x = lambda F: as_vector([F.dx(0), F.dx(1)])
        perp = lambda F: as_vector([-F[1], F[0]])
        n = as_vector([self.n[0], self.n[1]])

        # Initial and final time tags
        t0_tag = self.op.t_init_tag
        tf_tag = self.op.t_final_tag

        # Momentum equation
        self.lhs = inner(u.dx(2), v)*dx                 # Time derivative
        self.lhs += inner(g*grad_x(η), v)*dx            # Pressure gradient term
        self.lhs += inner(f*perp(u), v)*dx              # Coriolis term

        # Continuity equation
        self.lhs += inner(η.dx(2), θ)*dx                # Time derivative
        self.lhs += -inner(b*u, grad_x(θ))*dx           # Continuity term
        self.rhs = inner(b*Constant(as_vector([0.0, 0.0])), grad_x(θ))*dx

        # TODO: Enable different BCs

        # Initial conditions
        u0, eta0 = self.solution.copy(deepcopy=True).split()
        self.dbcs = [DirichletBC(self.V.sub(0), u0, t0_tag),
                     DirichletBC(self.V.sub(1), eta0, t0_tag)]

    def plot_solution(self, adjoint=False):
        if adjoint:
            z, zeta = self.adjoint_solution.split()
            zeta.rename("Adjoint elevation")
            spd = interpolate(sqrt(dot(u, u)), self.P1)
            spd.rename("Adjoint fluid speed")
            self.adjoint_solution_file.write(spd, zeta)
        else:
            u, eta = self.solution.split()
            eta.rename("Elevation")
            spd = interpolate(sqrt(dot(u, u)), self.P1)
            spd.rename("Fluid speed")
            self.solution_file.write(spd, eta)
