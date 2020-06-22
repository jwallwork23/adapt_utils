from thetis import *
from firedrake_adjoint import *
from firedrake.adjoint.blocks import GenericSolveBlock

import numpy as np

from adapt_utils.adapt.solver import AdaptiveProblem


__all__ = ["AdaptiveDiscreteAdjointProblem"]


class AdaptiveDiscreteAdjointProblem(AdaptiveProblem):
    # TODO: doc

    def __init__(self, *args, **kwargs):
        super(AdaptiveDiscreteAdjointProblem, self).__init__(*args, **kwargs)
        self.tape = get_working_tape()
        if self.num_meshes > 1:
            raise NotImplementedError  # TODO

    def clear_tape(self):
        self.tape.clear_tape()

    def compute_gradient(self, controls):
        J = self.quantity_of_interest()
        return compute_gradient(J, controls)

    def get_solve_blocks(self):
        blocks = self.tape.get_blocks()
        if len(blocks) == 0:
            raise ValueError("Tape is empty!")
        self.solve_blocks = [block for block in blocks if isinstance(block, GenericSolveBlock)]

    def extract_adjoint_solution(self, solve_step):
        i = 0  # TODO: Allow multiple meshes
        if not hasattr(self, 'solve_blocks'):
            self.get_solve_blocks()
        if self.op.solve_swe:
            if self.op.solve_tracer:
                raise NotImplementedError("Haven't accounted for coupled model yet.")  # TODO
            else:
                self.adj_solutions[i].assign(self.solve_blocks[solve_step].adj_sol)
        elif self.op.solve_tracer:
            self.adj_solutions_tracer[i].assign(self.solve_blocks[solve_step].adj_sol)
        else:
            raise ValueError
