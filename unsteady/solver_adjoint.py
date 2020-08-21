from thetis import *
from firedrake_adjoint import *
from firedrake.adjoint.blocks import GenericSolveBlock

from adapt_utils.unsteady.solver import AdaptiveProblem


__all__ = ["AdaptiveDiscreteAdjointProblem"]


class AdaptiveDiscreteAdjointProblem(AdaptiveProblem):
    """
    Subclass for :class:`AdaptiveProblem` which uses the discrete adjoint functionality built into
    Firedrake to solve adjoint problems.
    """
    def __init__(self, *args, **kwargs):
        super(AdaptiveDiscreteAdjointProblem, self).__init__(*args, **kwargs)
        self.tape = get_working_tape()
        if self.num_meshes > 1:
            raise NotImplementedError  # TODO: Allow multiple meshes

    def clear_tape(self):
        self.tape.clear_tape()

    # TODO: just call `solve_adjoint` (once merged)

    def compute_gradient(self, controls, scaling=1.0):
        """Compute the gradient of the quantity of interest with respect to a list of controls."""
        J = self.quantity_of_interest()
        return compute_gradient(J, controls, adj_value=scaling)

    def get_solve_blocks(self):
        """Extract all tape blocks which are subclasses of :class:`GenericSolveBlock`."""
        blocks = self.tape.get_blocks()
        if len(blocks) == 0:
            raise ValueError("Tape is empty!")
        self.solve_blocks = [block for block in blocks if isinstance(block, GenericSolveBlock) and block.adj_sol is not None]

    def extract_adjoint_solution(self, solve_step):
        """
        Extract the adjoint solution from solve block `solve_step`.

        NOTE: Only currently supported for either shallow water *or* tracer, *not* coupled mode.
        """
        i = 0  # TODO: Allow multiple meshes
        if solve_step % self.op.dt_per_export == 0:
            msg = "{:2d} {:s}  ADJOINT EXTRACT mesh {:2d}/{:2d}  time {:8.2f}"
            time = self.op.dt*solve_step
            self.print(msg.format(self.outer_iteration, '  '*i, i+1, self.num_meshes, time))
        if not hasattr(self, 'solve_blocks'):
            self.get_solve_blocks()
        adj_sol = self.solve_blocks[solve_step].adj_sol
        if self.op.solve_swe:
            if self.op.solve_tracer:
                raise NotImplementedError("Haven't accounted for coupled model yet.")  # TODO
            else:
                self.adj_solutions[i].assign(adj_sol)
        elif self.op.solve_tracer:
            self.adj_solutions_tracer[i].assign(adj_sol)
        else:
            raise ValueError

    def save_adjoint_trajectory(self):
        """Save the entire adjoint solution trajectory to .vtu, backwards in time."""
        self.get_solve_blocks()
        if self.op.solve_swe:
            self._save_adjoint_trajectory_shallow_water()
        if self.op.solve_tracer:
            self._save_adjoint_trajectory_tracer()

    def _save_adjoint_trajectory_shallow_water(self):
        i = 0  # TODO: Allow multiple meshes
        proj = Function(self.P1_vec[i]*self.P1[i])
        proj_u, proj_eta = proj.split()
        proj_u.rename("Projected discrete adjoint velocity")
        proj_eta.rename("Projected discrete adjoint elevation")
        iterator = list(range(len(self.solve_blocks)-1, -1, -self.op.dt_per_export))
        if 0 not in iterator:
            iterator.extend([0, ])
        for j in iterator:
            self.extract_adjoint_solution(j)
            if self.op.plot_pvd:
                proj.project(self.adj_solutions[i])
                self.adjoint_solution_file.write(proj_u, proj_eta)

    def _save_adjoint_trajectory_tracer(self):
        i = 0  # TODO: Allow multiple meshes
        proj = Function(self.P1[i], name="Projected discrete adjoint tracer")
        iterator = list(range(len(self.solve_blocks)-1, -1, -self.op.dt_per_export))
        if 0 not in iterator:
            iterator.extend([0, ])
        for j in iterator:
            self.extract_adjoint_solution(j)
            if self.op.plot_pvd:
                proj.project(self.adj_solutions_tracer[i])
                self.adjoint_tracer_file.write(proj)
