from thetis import *

import numpy as np

from adapt_utils.adapt.adaptation import AdaptiveMesh
from adapt_utils.solver import UnsteadyProblem


__all__ = ["AdaptiveProblem"]


class AdaptiveProblem():
    # TODO: doc
    def __init__(self, op, meshes, finite_element, discrete_adjoint=True, levels=0, hierarchies=None):
        op.print_debug(op.indent + "{:s} initialisation begin".format(self.__class__.__name__))

        # Read args and kwargs
        self.op = op
        self.finite_element = finite_element
        self.stabilisation = op.stabilisation
        self.discrete_adjoint = discrete_adjoint
        self.approach = op.approach
        self.levels = levels
        if levels > 0:
            raise NotImplementedError  # TODO

        # Setup problem
        self.num_meshes = op.num_meshes
        op.print_debug(op.indent + "SETUP: Building meshes...")
        self.set_meshes(meshes)
        op.print_debug(op.indent + "SETUP: Building function spaces...")
        self.create_function_spaces()
        op.print_debug(op.indent + "SETUP: Building solutions...")
        self.create_solutions()
        op.print_debug(op.indent + "SETUP: Building fields...")
        self.set_fields()
        self.set_stabilisation()
        op.print_debug(op.indent + "SETUP: Setting boundary conditions...")
        self.boundary_conditions = [op.set_boundary_conditions(V) for V in self.V]

        raise NotImplementedError  # TODO

    def set_meshes(self, meshes):  # TODO: levels > 0
        """
        Build an class:`AdaptiveMesh` object associated with each mesh.
        """
        self.meshes = meshes or [self.op.default_mesh for i in range(self.num_meshes)]
        self.ams = [AdaptiveMesh(self.meshes[i], levels=self.levels) for i in range(self.num_meshes)]
        msg = self.op.indent + "SETUP: Mesh {:d} has {:d} elements"
        for i in range(self.num_meshes):
            self.op.print_debug(msg.format(i, self.meshes[i].num_cells()))

    def create_function_spaces(self):
        """
        Build the finite element space, `V`, for the prognostic solution, along with various other
        useful spaces and test functions and trial functions based upon them.
        """
        fe = self.finite_element
        self.V = [FunctionSpace(mesh, fe) for mesh in self.meshes]
        self.P0 = [FunctionSpace(mesh, "DG", 0) for mesh in self.meshes]
        self.P1 = [FunctionSpace(mesh, "CG", 1) for mesh in self.meshes]
        # self.P2 = [FunctionSpace(mesh, "CG", 2) for mesh in self.meshes]
        # self.P1DG = [FunctionSpace(mesh, "DG", 1) for mesh in self.meshes]
        # self.P1_vec = [VectorFunctionSpace(mesh, "CG", 1) for mesh in self.meshes]
        # self.P1DG_vec = [VectorFunctionSpace(mesh, "DG", 1) for mesh in self.meshes]
        self.P1_ten = [TensorFunctionSpace(mesh, "CG", 1) for mesh in self.meshes]
        # self.test = [TestFunction(V) for V in self.V]
        # self.tests = [TestFunctions(V) for V in self.V]
        # self.trial = [TrialFunction(V) for V in self.V]
        # self.trials = [TrialFunctions(V) for V in self.V]
        # self.p0test = [TestFunction(P0) for P0 in self.P0]
        # self.p0trial = [TrialFunction(P0) for P0 in self.P0]

    def create_solutions(self):
        """
        Set up `Function`s in the prognostic space to hold the forward and adjoint solutions.
        """
        self.solutions = [Function(V, name='Solution') for V in self.V]
        self.adjoint_solutions = [Function(V, name='Adjoint solution') for V in self.V]

    def set_fields(self, adapted=False):
        """
        Set velocity field, viscosity, etc.
        """
        raise NotImplementedError("Should be implemented in derived class.")

    def set_stabilisation(self):
        """
        Set stabilisation mode and parameter.
        """
        raise NotImplementedError("Should be implemented in derived class.")

    def solve_step(self, adjoint=False):
        """
        Solve forward PDE on a particular mesh.
        """
        raise NotImplementedError("Should be implemented in derived class.")

    def solve_forward(self):
        raise NotImplementedError("Should be implemented in derived class.")

    def solve_adjoint(self):
        raise NotImplementedError("Should be implemented in derived class.")

    def dwr_indication(self, adjoint=False):
        """
        Indicate errors in the quantity of interest by the Dual Weighted Residual method. This is
        inherently problem-dependent.
        """
        raise NotImplementedError

    def quantity_of_interest(self):
        """
        Functional of interest which takes the PDE solution as input.
        """
        raise NotImplementedError("Should be implemented in derived class.")
