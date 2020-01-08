from firedrake import *
from firedrake.petsc import PETSc
from thetis import create_directory

import os
import numpy as np

from adapt_utils.misc.misc import index_string
from adapt_utils.misc.conditioning import *
from adapt_utils.adapt.adaptation import *
from adapt_utils.adapt.metric import *
from adapt_utils.adapt.p0_metric import *
from adapt_utils.adapt.kernels import eigen_kernel, matscale


__all__ = ["SteadyProblem", "UnsteadyProblem"]


class SteadyProblem():
    """
    Base class for solving steady-state PDE problems using mesh adaptivity.

    There are three main functionalities:
        * solve PDE;
        * solve adjoint PDE;
        * adapt mesh based on some error estimator of choice.
    """
    def __init__(self, op, mesh, finite_element, discrete_adjoint=False, prev_solution=None, levels=1):

        # Read args and kwargs
        self.op = op
        self.finite_element = finite_element
        self.stab = op.stabilisation
        self.discrete_adjoint = discrete_adjoint
        self.prev_solution = prev_solution
        self.approach = op.approach
        self.levels = levels

        # Setup problem
        self.set_mesh(mesh)
        if levels > 0:
            self.create_enriched_problem()
        self.create_function_spaces()
        self.create_solutions()
        self.boundary_conditions = op.set_boundary_conditions(self.V)

        # Outputs
        self.di = create_directory(self.op.di)
        self.solution_file = File(os.path.join(self.di, 'solution.pvd'))
        self.adjoint_solution_file = File(os.path.join(self.di, 'adjoint_solution.pvd'))
        self.indicator_file = File(os.path.join(self.di, 'indicator.pvd'))

        # Error estimator/indicator storage
        self.estimators = {}
        self.indicators = {}

    def set_mesh(self, mesh):
        """
        Build `AdaptiveMesh` object.
        """
        mesh = self.op.default_mesh if mesh is None else mesh
        self.am = AdaptiveMesh(mesh, levels=self.levels)
        self.mesh = self.am.mesh

    def create_enriched_problem(self, degree_increase=1):  # TODO: degree_increase=0 for SW
        """
        Create equivalent problem in iso-P2 refined space.
        """
        fe = FiniteElement(self.finite_element.family(),
                           self.finite_element.cell(),
                           self.finite_element.degree() + degree_increase)
        self.tp_enriched = type(self)(self.op, self.am.refined_mesh, fe,self. discrete_adjoint, self.prev_solution, self.levels-1)

    def create_function_spaces(self):
        """
        Build the finite element space, `V`, for the prognostic solution, along with various other
        useful spaces and test functions and trial functions based upon them.
        """
        self.V = FunctionSpace(self.mesh, self.finite_element)
        self.P0 = FunctionSpace(self.mesh, "DG", 0)
        self.P1 = FunctionSpace(self.mesh, "CG", 1)
        self.P2 = FunctionSpace(self.mesh, "CG", 2)
        self.P1DG = FunctionSpace(self.mesh, "DG", 1)
        self.P1_vec = VectorFunctionSpace(self.mesh, "CG", 1)
        self.P1_ten = TensorFunctionSpace(self.mesh, "CG", 1)
        self.test = TestFunction(self.V)
        self.trial = TrialFunction(self.V)
        self.p0test = TestFunction(self.P0)
        self.p0trial = TrialFunction(self.P0)
        self.n = FacetNormal(self.mesh)
        self.h = CellSize(self.mesh)

    def create_solutions(self):
        """
        Set up `Function`s in the prognostic space to hold the forward and adjoint solutions.
        """
        self.solution = Function(self.V, name='Solution')
        self.adjoint_solution = Function(self.V, name='Adjoint solution')

    def solve(self, adjoint=False):
        """
        Solve the forward or adjoint PDE, as specified by the boolean kwarg, `adjoint`.
        """
        if adjoint:
            self.solve_adjoint()
        else:
            self.solve_forward()

    def solve_forward(self):
        raise NotImplementedError("Should be implemented in derived class.")

    def solve_adjoint(self):
        """
        Solve adjoint problem using method specified by `discrete_adjoint` boolean kwarg.
        """
        if self.discrete_adjoint:
            PETSc.Sys.Print("Solving discrete adjoint problem...")
            self.solve_discrete_adjoint()
        else:
            PETSc.Sys.Print("Solving continuous adjoint problem...")
            self.solve_continuous_adjoint()

    def solve_continuous_adjoint(self):
        raise NotImplementedError("Should be implemented in derived class.")

    def solve_discrete_adjoint(self):
        try:
            assert hasattr(self, 'lhs')
        except AssertionError:
            raise ValueError("Cannot compute discrete adjoint since LHS unknown.")
        dFdu = derivative(self.lhs, self.solution, TrialFunction(self.V))
        dFdu_form = adjoint(dFdu)
        dJdu = derivative(self.quantity_of_interest_form(), self.solution, TestFunction(self.V))
        solve(dFdu_form == dJdu, self.adjoint_solution, solver_parameters=self.op.adjoint_params)
        self.plot()

    def solve_high_order(self, adjoint=True):  # TODO: Account for nonlinear case
        """
        Solve the problem using linear and quadratic approximations on a refined mesh, take the
        difference and project back into the original space.
        """
        # Solve adjoint problem on fine mesh using quadratic elements
        self.tp_enriched.solve(adjoint=adjoint)
        # TODO: For adjoint case, what if nonlinear?
        sol_p2 = self.tp_enriched.get_solution(adjoint=adjoint)

        # Project into P1 to get linear approximation, too
        sol_p1 = self.project_solution(self.tp_enriched.P1, adjoint=adjoint)
        # sol_p1 = Function(self.tp_enriched.P1)
        # prolong(sol, sol_p1)  # FIXME: Maybe the hierarchy isn't recognised?

        # Evaluate difference in enriched space
        self.set_error(interpolate(sol_p2 - sol_p1, self.tp_enriched.P2), adjoint=adjoint)

    def get_solution(self, adjoint=False):
        """
        Retrieve forward or adjoint solution, as specified by boolean kwarg `adjoint`.
        """
        return self.adjoint_solution if adjoint else self.solution

    def set_solution(self, val, adjoint=False):
        """
        Set forward or adjoint solution, as specified by boolean kwarg `adjoint`.
        """
        if adjoint:
            self.adjoint_solution = val
        else:
            self.solution = val

    def set_error(self, val, adjoint=False):
        """
        Set forward or adjoint error, as specified by boolean kwarg `adjoint`.
        """
        if adjoint:
            self.adjoint_error = val
        else:
            self.error = val

    def get_solution_label(self, adjoint=False):
        return 'adjoint_solution' if adjoint else 'solution'

    def project_solution(self, space, adjoint=False):
        """
        Project forward or adjoint solution into `space` space, as specified by the boolean kwarg
        `adjoint`.
        """
        if not hasattr(self, self.get_solution_label(adjoint)):
            self.solve(adjoint)
        return project(self.get_solution(adjoint), space)

    def get_qoi_kernel(self):
        """
        Derivative `g` of functional of interest `J`. i.e. For solution `u` we have
            J(u) = g . u
        """
        raise NotImplementedError("Should be implemented in derived class.")

    def quantity_of_interest(self):
        """
        Functional of interest which takes the PDE solution as input.
        """
        if not hasattr(self, 'kernel'):
            self.get_qoi_kernel()
        return assemble(inner(self.solution, self.kernel)*dx(degree=12))

    def get_strong_residual(self, adjoint=False):
        """
        Compute the strong residual for the forward or adjoint PDE, as specified by the `adjoint`
        boolean kwarg.
        """
        if adjoint:
            self.get_strong_residual_forward()
        else:
            self.get_strong_residual_adjoint()

    def get_strong_residual_forward(self):
        raise NotImplementedError("Should be implemented in derived class.")

    def get_strong_residual_adjoint(self):
        raise NotImplementedError("Should be implemented in derived class.")

    def get_dwr_residual(self, sol, adjoint_sol, adjoint=False):
        """
        Evaluate the cellwise component of the forward or adjoint Dual Weighted Residual (DWR) error
        estimator (see [Becker and Rannacher, 2001]), as specified by the boolean kwarg `adjoint`.
        """
        if adjoint:
            self.get_dwr_residual_adjoint(sol, adjoint_sol)
        else:
            self.get_dwr_residual_forward(sol, adjoint_sol)

    def get_dwr_residual_forward(self, sol, adjoint_sol):
        raise NotImplementedError("Should be implemented in derived class.")

    def get_dwr_residual_adjoint(self, sol, adjoint_sol):
        raise NotImplementedError("Should be implemented in derived class.")

    def get_dwr_flux(self, sol, adjoint_sol, adjoint=False):
        """
        Evaluate the edgewise component of the forward or adjoint Dual Weighted Residual (DWR) error
        estimator (see [Becker and Rannacher, 2001]), as specified by the boolean kwarg `adjoint`.
        """
        if adjoint:
            self.get_dwr_flux_adjoint(sol, adjoint_sol)
        else:
            self.get_dwr_flux_forward(sol, adjoint_sol)

    def get_dwr_flux_forward(self, sol, adjoint_sol):
        raise NotImplementedError("Should be implemented in derived class.")

    def get_dwr_flux_adjoint(self, sol, adjoint_sol):
        raise NotImplementedError("Should be implemented in derived class.")

    def dwr_indication(self, adjoint=False):  # TODO: Change inputs for consistency
        """
        Indicate errors in the quantity of interest by the Dual Weighted Residual (DWR) method of
        [Becker and Rannacher, 2001].

        A P1 field to be used for isotropic mesh adaptation is stored as `self.indicator`.
        """
        label = 'dwr'
        cell_label = 'dwr_cell'
        flux_label = 'dwr_flux'
        if adjoint:
            label += '_adjoint'
            cell_label += '_adjoint'
            flux_label += '_adjoint'
        self.get_dwr_residual(self.solution, self.adjoint_solution, adjoint=adjoint)
        self.get_dwr_flux(self.solution, self.adjoint_solution, adjoint=adjoint)
        self.indicator = Function(self.P1, name=label)
        self.indicator.interpolate(abs(self.indicators[cell_label] + self.indicators[flux_label]))
        self.estimators[label] = self.estimators[cell_label] + self.estimators[flux_label]
        self.indicators[label] = self.indicator

    def dwp_indication(self):
        """
        Indicate significance by the product of forward and adjoint solutions. This approach was
        used for mesh adaptive tsunami modelling in [Davis and LeVeque, 2016]. Here 'DWP' is used
        to stand for Dual Weighted Primal.
        """
        prod = inner(self.solution, self.adjoint_solution)
        self.indicators['dwp'] = assemble(self.p0test*prod*dx)
        self.indicator = interpolate(prod, self.P1)
        self.indicator.rename('dwp')

    def get_hessian_metric(self, adjoint=False):
        """
        Compute an appropriate Hessian metric for the problem at hand. This is inherently
        problem-dependent, since the choice of field for adaptation is not universal.

        Hessian metric should be computed and stored as `self.M`.
        """
        raise NotImplementedError("Should be implemented in derived class.")

    def get_hessian(self, adjoint=False):
        """
        Compute an appropriate Hessian for the problem at hand. This is inherently
        problem-dependent, since the choice of field for adaptation is not universal.
        """
        self.get_hessian_metric(adjoint=adjoint, noscale=True)
        return self.M

    def get_isotropic_metric(self):
        """
        Scale an identity matrix by the indicator field in order to drive
        isotropic mesh refinement.
        """
        el = self.indicator.ufl_element()
        name = self.indicator.dat.name
        if (el.family(), el.degree()) != ('Lagrange', 1):
            self.indicator = project(self.indicator, self.P1)
            self.indicator.rename(name)
        self.M = isotropic_metric(self.indicator, op=self.op)

    def get_loseille_metric(self, adjoint=False, relax=True):
        """
        Construct an anisotropic metric using an approach inspired by [Loseille et al. 2009].
        """
        raise NotImplementedError("Should be implemented in derived class.")

    def get_power_metric(self, adjoint=False):
        """
        Construct an anisotropic metric using an approach inspired by [Power et al. 2006].

        If `adjoint` mode is turned off, weight the Hessian of the adjoint solution with a residual
        for the forward PDE. Otherwise, weight the Hessian of the forward solution with a residual
        for the adjoint PDE.
        """
        self.get_strong_residual(adjoint=adjoint)
        H = self.get_hessian(adjoint=not adjoint)
        H_scaled = Function(self.P1_ten).assign(np.finfo(0.0).min)
        dim = self.mesh.topological_dimension()
        kernel = eigen_kernel(matscale, dim)
        op2.par_loop(kernel, self.P1.node_set, H_scaled.dat(op2.RW), H.dat(op2.READ), self.indicator.dat(op2.READ))
        self.M = steady_metric(self.solution if adjoint else self.adjoint_solution, H=H, op=self.op)

    def plot(self):
        """
        Plot current mesh and indicator field, if available.
        """
        File(os.path.join(self.di, 'mesh.pvd')).write(self.mesh.coordinates)
        if hasattr(self, 'indicator'):
            name = self.indicator.dat.name
            self.indicator.rename(' '.join([name, 'indicator']))
            File(os.path.join(self.di, 'indicator.pvd')).write(self.indicator)

    def indicate_error(self):  # TODO: Change 'relax' to 'average'
        """
        Evaluate error estimation strategy of choice in order to obtain a metric field for mesh
        adaptation.

        NOTE: User-provided metrics may be applied by defining a `custom_adapt` method.
        """
        if self.approach == 'fixed_mesh':
            return
        elif self.approach == 'uniform':
            return
        elif 'hessian' in self.approach:
            if self.approach in ('hessian', 'hessian_adjoint'):
                self.get_hessian_metric(adjoint='adjoint' in self.approach)
            else:
                self.get_hessian_metric(adjoint=False)
                M = self.M.copy()
                self.get_hessian_metric(adjoint=True)
                self.M = combine_metrics(M, self.M, average='relax' in self.approach)
        elif self.approach == 'dwp':
            self.dwp_indication()
            self.get_isotropic_metric()
        elif 'dwr' in self.approach:
            self.dwr_indication(adjoint='adjoint' in self.approach)
            self.get_isotropic_metric()
            if not self.approach in ('dwr', 'dwr_adjoint'):
                i = self.indicator.copy()
                M = self.M.copy()
                self.dwr_indication(adjoint=not 'adjoint' in self.approach)
                self.get_isotropic_metric()
                if self.approach in ('dwr_both', 'dwr_averaged'):
                    self.indicator.interpolate(Constant(0.5)*(i+self.indicator))
                    self.get_isotropic_metric()
                else:
                    self.M = combine_metrics(M, self.M, average='relax' in self.approach)
        elif 'loseille' in self.approach:
            self.get_loseille_metric(adjoint='adjoint' in self.approach)
            if not self.approach in ('loseille', 'loseille_adjoint'):
                M = self.M.copy()
                self.get_loseille_metric(adjoint=not 'adjoint' in self.approach)
                self.M = combine_metrics(M, self.M, average='relax' in self.approach)
        elif 'power' in self.approach:
            self.get_power_metric(adjoint='adjoint' in self.approach)
            if not self.approach in ('power', 'power_adjoint'):
                M = self.M.copy()
                self.get_power_metric(adjoint=not 'adjoint' in self.approach)
                self.M = combine_metrics(M, self.M, average='relax' in self.approach)
        elif 'carpio_isotropic' in self.approach:
            self.dwr_indication(adjoint='adjoint' in self.approach)
            eta = self.indicator.copy()
            if self.approach == 'carpio_isotropic_both':
                self.dwr_indication(adjoint=not 'adjoint' in self.approach)
                eta = Function(self.P0).interpolate(eta + self.indicator)
            amd = AnisotropicMetricDriver(self.am, indicator=eta, op=self.op)
            amd.get_isotropic_metric()
            self.M = amd.p1metric
        elif self.approach == 'carpio_both':
            self.dwr_indication()
            i = self.indicator.copy()
            self.get_hessian_metric(noscale=False, degree=1)  # NOTE: degree 0 doesn't work
            M = self.M.copy()
            self.dwr_indication(adjoint=True)
            self.get_hessian_metric(noscale=False, degree=1, adjoint=True)
            self.indicator.interpolate(i + self.indicator)
            self.M = metric_intersection(self.M, M)
            amd = AnisotropicMetricDriver(self.am, hessian=self.M, indicator=self.indicator, op=self.op)
            amd.get_anisotropic_metric()
            self.M = amd.p1metric
        elif 'carpio' in self.approach:
            self.dwr_indication(adjoint='adjoint' in self.approach)
            self.get_hessian_metric(noscale=True, degree=1, adjoint=adjoint)
            amd = AnisotropicMetricDriver(self.am, hessian=self.M, indicator=self.indicator, op=self.op)
            amd.get_anisotropic_metric()
            self.M = amd.p1metric
        else:
            try:
                assert hasattr(self, 'custom_adapt')
            except AssertionError:
                raise ValueError("Adaptivity mode {:s} not regcognised.".format(self.approach))
            PETSc.Sys.Print("Using custom metric '{:s}'".format(self.approach))
            self.custom_adapt()

        # Assemble global error estimator
        if hasattr(self, 'indicator'):
            self.estimator = self.indicator.vector().gather().sum()

    def adapt_mesh(self):
        """
        Adapt mesh using metric constructed in error estimation step.

        NOTE: User-provided metrics may be applied by defining a `custom_adapt` method.
        """
        if self.approach == 'fixed_mesh':
            return
        elif self.approach == 'uniform':
            if self.am.levels > 1:
                raise ValueError("Cannot perform uniform refinement because `AdaptiveMesh` object is not hierarchical.")
            self.mesh = self.am.hierarchy[1]
            return
        else:
            if not hasattr(self, 'M'):
                PETSc.Sys.Print("Metric not found. Computing it now.")
                self.indicate_error()
            self.am.adapt(self.M)
            self.mesh = self.am.mesh
        PETSc.Sys.Print("Done adapting. Number of elements: {:d}".format(self.mesh.num_cells()))
        self.plot()

        # Re-initialise problem
        self.create_function_spaces()
        self.create_solutions()

    def check_conditioning(self, submatrices=None):
        """
        Check condition number of LHS matrix, or submatrices thereof.

        :kwarg submatries: indices for desired submatrices (default all with `None`).
        """
        try:
            assert hasattr(self, 'lhs')
        except AssertionError:
            msg = "Cannot determine condition number since {:s} does not know the LHS."
            raise ValueError(msg.format(self.__class__.__name__))
        if hasattr(self.V, 'num_sub_spaces'):
            n = self.V.num_sub_spaces()
            cc = NestedConditionCheck(self.lhs)
            if submatrices is None:
                submatrices = []
                for i in range(n):
                    for j in range(n):
                        submatrices.append((i, j))
            self.condition_number = {}
            for s in submatrices:
                kappa = cc.condition_number(s[0], s[1])
                self.condition_number[s] = kappa
                PETSc.Sys.Print("Condition number %1d,%1d: %.4e" % (s[0], s[1], kappa))
        else:
            cc = UnnestedConditionCheck(self.lhs)
            self.condition_number = cc.condition_number()
            PETSc.Sys.Print("Condition number: %.4e" % self.condition_number)


class UnsteadyProblem(SteadyProblem):
    """
    Base class for solving time-dependent PDE problems using mesh adaptivity.

    There are three main functionalities:
        * solve PDE;
        * solve adjoint PDE;
        * adapt mesh based on some error estimator of choice.
    """
    def __init__(self, op, mesh, finite_element, discrete_adjoint=False, levels=1):
        super(UnsteadyProblem, self).__init__(op, mesh, finite_element, discrete_adjoint, None, levels)
        self.set_start_condition()
        self.step_end = op.end_time if self.approach == 'fixed_mesh' else op.dt*op.dt_per_remesh
        self.estimators = {}
        self.indicators = {}
        self.num_exports = int(np.floor((op.end_time - op.dt)/op.dt/op.dt_per_export))

    def create_solutions(self):
        self.solution = Function(self.V, name='Solution')
        self.adjoint_solution = Function(self.V, name='Adjoint solution')
        self.solution_old = Function(self.V, name='Old solution')
        self.adjoint_solution_old = Function(self.V, name='Old adjoint solution')

    def set_start_condition(self, adjoint=False):
        self.set_solution(self.op.set_final_condition(self.V), adjoint)

    def solve_step(self, adjoint=False, **kwargs):
        """
        Solve forward PDE on a particular mesh.
        """
        raise NotImplementedError("Should be implemented in derived class.")

    def set_fields(self):
        """
        Set velocity field, viscosity, QoI kernel, etc.
        """
        raise NotImplementedError("Should be implemented in derived class.")

    def solve(self, adjoint=False):
        """
        Solve PDE using mesh adaptivity.
        """
        self.remesh_step = 0

        # Adapt w.r.t. initial conditions a few times before the solver loop
        if self.approach != 'fixed_mesh':
            for i in range(max(self.op.num_adapt, 2)):
                self.get_adjoint_state()
                self.adapt_mesh()
                self.set_start_condition(adjoint)
                self.set_fields()
        elif adjoint:
            self.set_start_condition(adjoint)

        # Solve/adapt loop
        while self.step_end <= self.op.end_time:

            # Fixed mesh case
            if self.approach == 'fixed_mesh':
                self.solve_step(adjoint=adjoint)
                break

            # Adaptive mesh case
            for i in range(self.op.num_adapt):
                self.adapt_mesh()

                # Interpolate value from previous step onto new mesh
                if self.remesh_step == 0:
                    self.set_start_condition(adjoint)
                elif i == 0:
                    self.solution.project(solution)
                else:
                    self.solution.project(solution_old)

                # Solve PDE on new mesh
                self.op.plot_pvd = True if i == 0 else False
                time = None if i == 0 else self.step_end - self.op.dt
                self.solve_step(adjoint=adjoint, time=time)

                # Store solutions from last two steps on first mesh in sequence
                if i == 0:
                    solution = Function(self.V)
                    solution.assign(self.solution)
                    solution_old = Function(self.V)
                    solution_old.assign(self.solution_old)
                    if self.step_end + self.op.dt*self.op.dt_per_remesh > self.op.end_time:
                        break  # No need to do adapt for final timestep
            self.plot()

            self.step_end += self.op.dt*self.op.dt_per_remesh
            self.remesh_step += 1

        # Evaluate QoI
        if self.approach != 'fixed_mesh':
            self.solution.project(solution)
        self.get_qoi_kernel()

    def quantity_of_interest(self):
        """
        Functional of interest which takes the PDE solution as input.
        """
        if not hasattr(self, 'kernel'):
            self.get_qoi_kernel()
        raise NotImplementedError  # TODO: account for time integral forms

    def get_adjoint_state(self, variable='Tracer2d'):
        """
        Get adjoint solution at timestep i.
        """
        if self.approach in ('uniform', 'hessian', 'vorticity'):
            return
        if not hasattr(self, 'V_orig'):
            self.V_orig = FunctionSpace(self.mesh, self.finite_element)
        op = self.op
        names = {'Tracer2d': 'tracer_2d', 'Velocity2d': 'uv_2d', 'Elevation2d': 'elev_2d'}
        i = self.remesh_step*int(self.op.dt_per_export/self.op.dt_per_remesh)

        # FIXME for continuous adjoint
        filename = 'Adjoint2d_{:5s}'.format(index_string(i))

        # filename = '{:s}_{:5s}'.format(variable, index_string(i))
        to_load = Function(self.V_orig, name=names[variable])
        to_load_old = Function(self.V_orig, name=names[variable])
        with DumbCheckpoint(os.path.join('outputs/adjoint/hdf5', filename), mode=FILE_READ) as la:
            la.load(to_load)
            la.load(to_load_old)
            la.close()
        self.adjoint_solution.project(to_load)
        self.adjoint_solution_old.project(to_load_old)
        self.adjoint_solution_file.write(self.adjoint_solution, t=self.op.dt*i)

    def dwr_indication(self, adjoint=False):
        """
        Indicate errors in the quantity of interest by the Dual Weighted Residual method. This is
        inherently problem-dependent.

        The resulting P0 field should be stored as `self.indicator`.
        """
        raise NotImplementedError  # TODO: Use steady format, but need to get adjoint sol
