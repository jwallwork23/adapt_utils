from firedrake import *
from firedrake_adjoint import *
from fenics_adjoint.solving import SolveBlock       # For extracting adjoint solutions
from fenics_adjoint.projection import ProjectBlock  # Exclude projections from tape reading

from adapt_utils.adapt.options import DefaultOptions
from adapt_utils.adapt.metric import isotropic_metric


class BaseProblem():
    """
    Base class for solving PDE problems using mesh adaptivity.

    There are three main functionalities:
        * solve PDE;
        * solve adjoint PDE using pyadjoint;
        * adapt mesh based on some error estimator of choice.
    """
    def __init__(self, mesh, approach, issteady=True, op=DefaultOptions()):
        self.mesh = mesh
        self.approach = approach
        self.issteady = issteady
        self.op = op
        self.op.approach = approach

        # function spaces
        self.P0 = FunctionSpace(self.mesh, "DG", 0)
        self.P1 = FunctionSpace(self.mesh, "CG", 1)
        self.P1_vec = VectorFunctionSpace(self.mesh, "CG", 1)

        # TODO: initialise solution and adjoint in order to name adjoint nicely

    def solve(self):
        """
        Solve forward PDE.
        """
        pass


    def objective_functional(self):
        """
        Functional of interest which takes the PDE solution as input.
        """
        pass

    def solve_continuous_adjoint(self):
        """
        Solve the adjoint PDE using a hand-coded continuous adjoint.
        """
        pass

    def solve_discrete_adjoint(self):
        """
        Solve the adjoint PDE in the discrete sense, using pyadjoint.
        """
        if self.issteady:

            # compute some gradient in order to get adjoint solutions
            J = self.objective_functional()
            compute_gradient(J, Control(self.gradient_field))
            tape = get_working_tape()
            solve_blocks = [block for block in tape._blocks if isinstance(block, SolveBlock)
                                                            and not isinstance(block, ProjectBlock)
                                                            and block.adj_sol is not None]
            try:
                assert len(solve_blocks) == 1
            except:
                ValueError("Expected one SolveBlock, but encountered {:d}".format(len(solve_blocks)))

            # extract adjoint solution
            self.adjoint_solution = Function(self.solution.function_space())
            self.adjoint_solution.assign(solve_blocks[0].adj_sol)
            # TODO: initialise solution and adjoint in order to name adjoint nicely
        else:
            raise NotImplementedError  # TODO

    def dwp_indication(self):
        """
        Indicate significance by the product of forward and adjoint solutions. This approach was
        used for mesh adaptive tsunami modelling in [Davis and LeVeque, 2016]. Here 'DWP' is used
        to stand for Dual Weighted Primal.
        """
        self.indicator = Function(self.P1)
        self.indicator.project(inner(self.solution, self.adjoint_solution))
        self.indicator.rename('dwp')

    def normalise_indicator(self):
        """
        Given a scalar indicator f and a target number of vertices N, rescale f by
            f := abs(f) * N / norm(f),
        subject to the imposition of minimum and maximum tolerated norms.
        """
        scale_factor = min(max(norm(self.indicator), self.op.min_norm), self.op.max_norm)
        if scale_factor < 1.00001*self.op.min_norm:
            print("WARNING: minimum norm attained")
        elif scale_factor > 0.99999*self.op.max_norm:
            print("WARNING: maximum norm attained")
        self.indicator.interpolate(Constant(self.op.target_vertices/scale_factor)*abs(self.indicator))

    def plot(self):
        """
        Plot current mesh and indicator field, if available.
        """
        di = self.op.directory()
        File(di + 'mesh.pvd').write(self.mesh.coordinates)
        if hasattr(self, 'indicator'):
            name = self.indicator.dat.name
            self.indicator.rename(name + ' indicator')
            File(di + 'indicator.pvd').write(self.indicator)

    def dwr_estimation(self):
        """
        Indicate errors in the objective functional by the Dual Weighted Residual method. This is
        inherently problem-dependent.

        The resulting P0 field should be stored as `self.indicator`.
        """
        pass

    def get_hessian_metric(self, adjoint=False):
        """
        Compute an appropriate Hessian metric for the problem at hand. This is inherently
        problem-dependent, since the choice of field for adaptation is not universal.

        Hessian metric should be computed and stored as `self.M`.
        """
        pass

    def get_isotropic_metric(self):
        """
        Scale an identity matrix by the indicator field `self.indicator` in order to drive
        isotropic mesh refinement.
        """
        el = self.indicator.ufl_element()
        if (el.family(), el.degree()) != ('Lagrange', 1):
            self.indicator = project(self.indicator, self.P1)
        self.normalise_indicator()
        self.M = isotropic_metric(self.indicator, op=self.op)

    def get_anisotropic_metric(self):
        """
        Apply the approach of [Loseille, Dervieux, Alauzet, 2009] to extract an anisotropic mesh 
        from the Dual Weighted Residual method.
        """
        pass

    def adapt_mesh(self):
        """
        Adapt mesh according to error estimation strategy of choice.
        """
        if not hasattr(self, 'solution'):
            self.solve()

        # Get metric (if appropriate)
        if self.approach == 'fixed_mesh':
            return
        elif self.approach == 'uniform':
            self.mesh = MeshHierarchy(self.mesh, 1)[0]
            return
        elif self.approach == 'hessian':
            self.get_hessian_metric()
        else:
            if not hasattr(self, 'adjoint_solution'):
                self.solve_adjoint()

            if self.approach == 'dwp':
                self.dwp_indication()
                self.get_isotropic_metric()
            elif self.approach == 'dwr':
                self.dwr_estimation()
                self.get_isotropic_metric()
            else:
                raise NotImplementedError  # TODO

        # Adapt mesh
        self.mesh = adapt(self.mesh, self.M)

    def interpolate_fields(self):
        """
        Interpolate fields onto the new mesh after a mesh adaptation.
        """
        raise NotImplementedError  # TODO


class MeshOptimisation():
    """
    Loop over all mesh optimisation steps in order to obtain a mesh which is optimal w.r.t. the
    given error estimator for the given PDE problem.
    """
    def __init__(self, approach='uniform'):
        self.approach = approach

    def iterate(self):
        raise NotImplementedError  # TODO
