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

    def set_controls(self):
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

    def setup_all(self, **kwargs):
        """
        Clear the tape and reset controls.
        """
        self.clear_tape()
        super(AdaptiveDiscreteAdjointSteadyProblem, self).setup_all(**kwargs)
        self.set_controls()

    def get_metric(self, *args, **kwargs):
        """
        Do not annotate the metric computation process.
        """
        with stop_annotating():
            return super(AdaptiveDiscreteAdjointSteadyProblem, self).get_metric(*args, **kwargs)

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
