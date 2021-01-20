from thetis import *
from firedrake_adjoint import *
from firedrake.adjoint.blocks import GenericSolveBlock, ProjectBlock
import pyadjoint

import os

from adapt_utils.steady.solver import AdaptiveSteadyProblem


__all__ = ["AdaptiveSteadyProblem"]


class AdaptiveDiscreteAdjointSteadyProblem(AdaptiveSteadyProblem):
    """
    Subclass for :class:`AdaptiveSteadyProblem` which uses the discrete adjoint functionality
    built into Firedrake to solve adjoint problems.
    """
    def __init__(self, *args, **kwargs):
        self.tape = get_working_tape()
        super(AdaptiveDiscreteAdjointSteadyProblem, self).__init__(*args, **kwargs)

        # Default control
        if self.equation_set == 'shallow_water':
            self.control_field = self.fields[0].horizontal_viscosity
        elif self.equation_set == 'tracer':
            self.control_field = self.fields[0].horizontal_diffusivity
        else:
            raise NotImplementedError
        self.control = Control(self.control_field)
        self.step = Function(self.control_field.function_space()).assign(1.0)
        self.initial_control = self.control_field.copy(deepcopy=True)

    def clear_tape(self):
        self.tape.clear_tape()

    def adapt_meshes(self, **kwargs):
        """
        Fully reset the problem - including the tape - when the mesh is adapted.
        """
        self.print("\nStarting mesh adaptation for iteration {:d}...".format(self.outer_iteration+1))
        self.meshes[0] = adapt(self.meshes[0], self.metrics[0])
        self.set_meshes(self.meshes)
        self.clear_tape()
        self.__init__(self.op, meshes=self.meshes, nonlinear=self.nonlinear)

        # Logging
        adapt_field = self.op.adapt_field
        if self.op.adapt_field not in ('tracer', 'sediment', 'bathymetry'):
            adapt_field = 'shallow_water'
        self.log_entities(adapt_field=adapt_field)

    def get_solve_blocks(self):
        """
        Extract all tape blocks which are subclasses of :class:`GenericSolveBlock`, but not
        :class:`ProjectBlock`.
        """
        blocks = self.tape.get_blocks()
        if len(blocks) == 0:
            raise ValueError("Tape is empty!")
        solve_blocks = [block for block in blocks if isinstance(block, GenericSolveBlock)]
        solve_blocks = [block for block in solve_blocks if not isinstance(block, ProjectBlock)]
        return [block for block in solve_blocks if block.adj_sol is not None]

    @property
    def solve_blocks(self):
        return self.get_solve_blocks()

    def solve_adjoint(self, **kwargs):
        """
        Solve discrete adjoint as a by-product when computing the gradient w.r.t. the default
        control.
        """
        J = self.quantity_of_interest()
        dJdm = compute_gradient(J, self.control)
        adj_sol = self.get_solutions(self.equation_set, adjoint=True)[0]
        adj_sol.assign(-self.solve_blocks[-1].adj_sol)  # TODO: Why minus sign?
        return J
