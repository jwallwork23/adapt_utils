from thetis import *
from firedrake_adjoint import *
from firedrake.adjoint.blocks import GenericSolveBlock
import pyadjoint

from adapt_utils.unsteady.solver import AdaptiveProblem


__all__ = ["AdaptiveDiscreteAdjointProblem"]


class AdaptiveDiscreteAdjointProblem(AdaptiveProblem):
    """
    Subclass for :class:`AdaptiveProblem` which uses the discrete adjoint functionality built into
    Firedrake to solve adjoint problems.
    """
    def __init__(self, *args, **kwargs):
        super(AdaptiveDiscreteAdjointProblem, self).__init__(*args, **kwargs)
        op = self.op
        self.tape = get_working_tape()
        if self.num_meshes > 1:
            raise NotImplementedError  # TODO: Allow multiple meshes

        # Check that only one set of equations has been solved  # TODO
        op = self.op
        equations = []
        for to_solve in (op.solve_swe, op.solve_tracer, op.solve_sediment, op.solve_exner):
            if to_solve:
                equations.append(to_solve)
        if len(equations) == 0:
            raise ValueError("No equations solved for!")
        elif len(equations) > 1:
            raise NotImplementedError("Haven't accounted for coupled model yet.")

    def clear_tape(self):
        """
        Strip all blocks from pyadjoint's tape, along with their associated data.
        """
        print_output("Clearing tape...")
        self.tape.clear_tape()

    def solve_adjoint(self, scaling=1.0):
        """
        Solve the discrete adjoint problem for some quantity of interest.
        """
        J = self.quantity_of_interest()
        return pyadjoint.solve_adjoint(J, adj_value=scaling)

    def compute_gradient(self, controls, scaling=1.0):
        """
        Compute the gradient of the quantity of interest with respect to a list of controls.
        """
        J = self.quantity_of_interest()
        return compute_gradient(J, controls, adj_value=scaling)

    def check_solve_block(self, block):
        """
        Check that `block` corresponds to a finite element/nonlinear/linear solve.
        """
        out = True
        if not isinstance(block, GenericSolveBlock):
            out = False
        elif not hasattr(block, 'adj_sol'):
            out = False
        elif block.adj_sol is None:
            out = False
        return out

    def get_solve_blocks(self):
        """
        Extract all tape blocks which are subclasses of :class:`GenericSolveBlock`.
        """
        blocks = self.tape.get_blocks()
        if len(blocks) == 0:
            raise ValueError("Tape is empty!")
        self._solve_blocks = [block for block in blocks if self.check_solve_block(block)]

    @property
    def solve_blocks(self):
        if not hasattr(self, '_solve_blocks'):
            self.get_solve_blocks()
        return self._solve_blocks

    def extract_adjoint_solution(self, solve_step):
        """
        Extract the adjoint solution from solve block `solve_step`.

        NOTE: Only currently supported for either shallow water *or* tracer, *not* coupled mode.
        """
        op = self.op
        i = 0  # TODO: Allow multiple meshes
        if solve_step % op.dt_per_export == 0:
            time = op.dt*solve_step
            if self.num_meshes == 1:
                self.print("ADJOINT EXTRACT  time {:8.2f}".format(time))
            else:
                msg = "{:2d} {:s}  ADJOINT EXTRACT mesh {:2d}/{:2d}  time {:8.2f}"
                self.print(msg.format(self.outer_iteration, '  '*i, i+1, self.num_meshes, time))
        adj_sol = self.solve_blocks[solve_step].adj_sol

        # Extract adjoint solution and insert it into the appropriate solution field
        if self.op.solve_swe:
            self.adj_solutions[i].assign(adj_sol)
        elif self.op.solve_tracer:
            self.adj_solutions_tracer[i].assign(adj_sol)
        elif self.op.solve_sediment:
            self.adj_solutions_sediment[i].assign(adj_sol)
        else:
            self.adj_solutions_bathymetry[i].assign(adj_sol)

    def save_adjoint_trajectory(self):
        """
        Save the entire adjoint solution trajectory to .vtu, backwards in time.
        """
        if self.op.solve_swe:
            self._save_adjoint_trajectory_shallow_water()
        elif self.op.solve_tracer:
            self._save_adjoint_trajectory_tracer()
        elif self.op.solve_sediment:
            self._save_adjoint_trajectory_sediment()
        else:
            self._save_adjoint_trajectory_bathymetry()

    def _save_adjoint_trajectory_shallow_water(self):
        i = 0  # TODO: Allow multiple meshes
        proj = Function(self.P1_vec[i]*self.P1[i])
        proj_u, proj_eta = proj.split()
        proj_u.rename("Projected discrete adjoint velocity")
        proj_eta.rename("Projected discrete adjoint elevation")
        iterator = list(range(len(self.solve_blocks)-1, -1, -self.op.dt_per_export))
        if 0 not in iterator:
            iterator.extend([0])
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
            iterator.extend([0])
        for j in iterator:
            self.extract_adjoint_solution(j)
            if self.op.plot_pvd:
                proj.project(self.adj_solutions_tracer[i])
                self.adjoint_tracer_file.write(proj)

    def _save_adjoint_trajectory_sediment(self):
        raise NotImplementedError

    def _save_adjoint_trajectory_bathymetry(self):
        raise NotImplementedError
