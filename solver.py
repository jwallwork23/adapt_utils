from firedrake import *
from thetis import create_directory, print_output

import os
import numpy as np
import matplotlib.pyplot as plt

from adapt_utils.adapt.adaptation import *
from adapt_utils.adapt.metric import *
from adapt_utils.adapt.p0_metric import *
from adapt_utils.adapt.r import *
from adapt_utils.adapt.kernels import eigen_kernel, matscale
from adapt_utils.misc import *


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
        print_output(op.indent + "{:s} initialisation begin".format(self.__class__.__name__))

        # Read args and kwargs
        self.op = op
        self.finite_element = finite_element
        self.stabilisation = op.stabilisation
        self.discrete_adjoint = discrete_adjoint
        self.prev_solution = prev_solution
        self.approach = op.approach
        self.levels = levels

        # Setup problem
        print_output(op.indent+"Building mesh...")
        self.set_mesh(mesh)
        self.am_init = self.am.copy()
        print_output(op.indent+"Building function spaces...")
        self.create_function_spaces()
        print_output(op.indent+"Building solutions...")
        self.create_solutions()
        print_output(op.indent+"Building fields...")
        self.set_fields()
        self.set_stabilisation()
        print_output(op.indent+"Setting boundary conditions...")
        self.boundary_conditions = op.set_boundary_conditions(self.V)

        # Outputs
        self.di = create_directory(self.op.di)
        self.solution_file = File(os.path.join(self.di, 'solution.pvd'))
        self.adjoint_solution_file = File(os.path.join(self.di, 'adjoint_solution.pvd'))
        self.indicator_file = File(os.path.join(self.di, 'indicator.pvd'))
        self.monitor_file = File(os.path.join(self.di, 'monitor.pvd'))

        # Storage during inner mesh adaptation loop
        self.indicators = {}
        self.estimators = {}
        self.num_cells = [self.mesh.num_cells()]
        self.num_vertices = [self.mesh.num_vertices()]
        self.qois = []

        # Storage during outer mesh adaptation loop
        self.outer_estimators = []
        self.outer_num_cells = []
        self.outer_num_vertices = []
        self.outer_qois = []
        print_output(op.indent + "{:s} initialisation complete!\n".format(self.__class__.__name__))

    def set_mesh(self, mesh):
        """
        Build `AdaptiveMesh` object.
        """
        mesh = mesh or self.op.default_mesh
        self.am = AdaptiveMesh(mesh, levels=self.levels)
        self.mesh = self.am.mesh
        if self.levels > 0:
            self.create_enriched_problem()
        self.n = FacetNormal(self.mesh)  # TODO: use version in AdaptiveMesh
        self.h = CellSize(self.mesh)     # TODO: use version in AdaptiveMesh
        self.op.print_debug("Number of mesh elements: {:d}".format(mesh.num_cells()))

    def create_enriched_problem(self):
        """
        Create equivalent problem in iso-P2 refined space.
        """
        self.op_enriched = self.op.copy()
        self.op_enriched.degree += self.op.degree_increase
        self.op_enriched.indent += '  '
        print_output("\nCreating enriched finite element space of degree {:d}...".format(self.op_enriched.degree))
        self.tp_enriched = type(self)(self.op_enriched, mesh=self.am.refined_mesh,
                                      discrete_adjoint=self.discrete_adjoint,
                                      prev_solution=self.prev_solution, levels=self.levels-1)

    def create_function_spaces(self):
        """
        Build the finite element space, `V`, for the prognostic solution, along with various other
        useful spaces and test functions and trial functions based upon them.
        """
        fe = self.finite_element
        self.V = FunctionSpace(self.mesh, fe)
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

    def create_solutions(self):
        """
        Set up `Function`s in the prognostic space to hold the forward and adjoint solutions.
        """
        self.solution = Function(self.V, name='Solution')
        self.adjoint_solution = Function(self.V, name='Adjoint solution')

    def set_fields(self):
        """
        Set velocity field, viscosity, etc.
        """
        raise NotImplementedError("Should be implemented in derived class.")

    def project_fields(self, prob):
        """
        Project velocity field, viscosity, etc. from another Problem.
        """
        raise NotImplementedError("Should be implemented in derived class.")

    def set_stabilisation(self):
        """
        Set stabilisation mode and parameter.
        """
        raise NotImplementedError("Should be implemented in derived class.")

    def setup_solver(self, adjoint=False):
        """
        Setup solver for forward or adjoint PDE, as specified by the boolean kwarg, `adjoint`.
        """
        if adjoint:
            self.setup_solver_adjoint()
        else:
            self.setup_solver_forward()

    def setup_solver_forward(self):
        raise NotImplementedError("Should be implemented in derived class.")

    def setup_solver_adjoint(self):
        raise NotImplementedError("Should be implemented in derived class.")

    def solve(self, adjoint=False):
        """
        Solve the forward or adjoint PDE, as specified by the boolean kwarg, `adjoint`.
        """
        if adjoint:
            self.solve_adjoint()
        else:
            self.solve_forward()

    def solve_forward(self):
        self.setup_solver_forward()
        if self.nonlinear:
            self.rhs = 0
        solve(self.lhs == self.rhs, self.solution, bcs=self.dbcs, solver_parameters=self.op.params)
        self.plot_solution(adjoint=False)

    def solve_adjoint(self):
        """
        Solve adjoint problem using method specified by `discrete_adjoint` boolean kwarg.
        """
        if self.discrete_adjoint:
            print_output("Solving discrete adjoint problem on mesh with {:d} elements".format(self.mesh.num_cells()))
            self.solve_discrete_adjoint()
        else:
            print_output("Solving continuous adjoint problem on mesh with {:d} elements".format(self.mesh.num_cells()))
            self.solve_continuous_adjoint()
        self.plot_solution(adjoint=True)

    def solve_continuous_adjoint(self):
        self.setup_solver_adjoint()
        if self.nonlinear:
            self.rhs_adjoint = 0
        solve(self.lhs_adjoint == self.rhs_adjoint, self.adjoint_solution, bcs=self.dbcs_adjoint, solver_parameters=self.op.adjoint_params)

    def solve_discrete_adjoint(self):
        try:
            assert hasattr(self, 'lhs')
        except AssertionError:
            raise ValueError("Cannot compute discrete adjoint since LHS unknown.")
        dFdu = derivative(self.lhs, self.solution, TrialFunction(self.V))
        dFdu_form = adjoint(dFdu)
        dJdu = derivative(self.quantity_of_interest_form(), self.solution, TestFunction(self.V))
        solve(dFdu_form == dJdu, self.adjoint_solution, solver_parameters=self.op.adjoint_params)

    def solve_high_order(self, adjoint=True, solve_forward=False):
        """
        Solve the problem using linear and quadratic approximations on a refined mesh, take the
        difference and project back into the original space.
        """
        tpe = self.tp_enriched
        solve_forward &= self.nonlinear  # (Adjoint of linear PDE independent of forward)

        # Solve on a fine mesh using elements of higher order
        if adjoint:
            if solve_forward:
                tpe.solve_forward()
            elif self.nonlinear:
                tpe.project_solution(self.solution, adjoint=False)  # FIXME: prolong
                if hasattr(tpe, 'setup_solver'):
                    tpe.setup_solver()
        tpe.solve(adjoint=adjoint)
        sol_p2 = tpe.get_solution(adjoint=adjoint)

        # Project into P1 to get linear approximation, too
        sol_p1 = tpe.project(self.get_solution(adjoint=adjoint))
        # sol_p1 = Function(tpe.P1)
        # prolong(sol, sol_p1)  # FIXME: Maybe the hierarchy isn't recognised?

        # Evaluate difference in enriched space
        self.set_error(tpe.interpolate(tpe.difference(sol_p2, sol_p1)), adjoint=adjoint)

    def get_solution(self, adjoint=False):
        """
        Retrieve forward or adjoint solution, as specified by boolean kwarg `adjoint`.
        """
        return self.adjoint_solution if adjoint else self.solution

    def get_error(self, adjoint=False):
        """
        Retrieve forward or adjoint error, as specified by boolean kwarg `adjoint`.
        """
        return self.adjoint_error if adjoint else self.error

    def set_solution(self, val, adjoint=False):
        """
        Set forward or adjoint solution, as specified by boolean kwarg `adjoint`.
        """
        # TODO: Update names in mixed space
        # name = self.get_solution(adjoint).dat.name
        if adjoint:
            self.adjoint_solution = val
        else:
            self.solution = val
        # self.get_solution(adjoint).rename(name)

    def set_error(self, val, adjoint=False):
        """
        Set forward or adjoint error, as specified by boolean kwarg `adjoint`.
        """
        if adjoint:
            self.adjoint_error = val
        else:
            self.error = val

    def difference(self, u, v, out=None):
        """
        Take the difference of two functions `u` and `v` defined on `self.mesh`.
        """
        assert u.function_space() == v.function_space()
        out = out or Function(u.function_space())
        assert out.function_space() == u.function_space()
        out.assign(u)
        out -= v
        return out

    def interpolate(self, val, out=None):
        """
        Interpolate a function in `self.V`.
        """
        if isinstance(val, Constant) or val is None:
            return val
        out = out or Function(self.V)
        assert out.function_space() == self.V
        for outi, vi in zip(out.split(), val.split()):
            outi.interpolate(vi)
        return out

    def project(self, val, out=None):
        """
        Project a function in `V`.
        """
        if isinstance(val, Constant) or val is None:
            return val
        out = out or Function(self.V)
        for outi, vi in zip(out.split(), val.split()):
            outi.project(vi)
        return out

    def interpolate_solution(self, val, adjoint=False):
        """
        Interpolate forward or adjoint solution, as specified by the boolean kwarg
        `adjoint`.
        """
        self.interpolate(val, out=self.get_solution(adjoint=adjoint))

    def project_solution(self, val, adjoint=False):
        """
        Project forward or adjoint solution, as specified by the boolean kwarg
        `adjoint`.
        """
        self.project(val, out=self.get_solution(adjoint=adjoint))

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

    def get_dwr_residual(self, adjoint=False):
        """
        Evaluate the cellwise component of the forward or adjoint Dual Weighted Residual (DWR) error
        estimator (see [Becker and Rannacher, 2001]), as specified by the boolean kwarg `adjoint`.
        """
        if adjoint:
            self.get_dwr_residual_adjoint()
        else:
            self.get_dwr_residual_forward()

    def get_dwr_residual_forward(self):
        raise NotImplementedError("Should be implemented in derived class.")

    def get_dwr_residual_adjoint(self):
        raise NotImplementedError("Should be implemented in derived class.")

    def get_dwr_flux(self, adjoint=False):
        """
        Evaluate the edgewise component of the forward or adjoint Dual Weighted Residual (DWR) error
        estimator (see [Becker and Rannacher, 2001]), as specified by the boolean kwarg `adjoint`.
        """
        if adjoint:
            self.get_dwr_flux_adjoint()
        else:
            self.get_dwr_flux_forward()

    def get_dwr_flux_forward(self):
        raise NotImplementedError("Should be implemented in derived class.")

    def get_dwr_flux_adjoint(self):
        raise NotImplementedError("Should be implemented in derived class.")

    def dwr_indication(self, adjoint=False):
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

        # Compute DWR residual and flux terms
        self.solve_high_order(adjoint=not adjoint)
        self.get_dwr_residual(adjoint=adjoint)
        self.get_dwr_flux(adjoint=adjoint)

        # Indicate error in P1 space
        self.indicator = Function(self.P1, name=label)
        self.indicator.interpolate(abs(self.indicators[cell_label] + self.indicators[flux_label]))
        self.indicators[label] = Function(self.P0, name=label)
        self.indicators[label].interpolate(abs(self.indicators[cell_label] + self.indicators[flux_label]))

        # Estimate error
        if not label in self.estimators:
            self.estimators[label] = []
        self.estimators[label].append(self.estimators[cell_label][-1] + self.estimators[flux_label][-1])

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
        self.estimate_error('dwp')

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
        H_scaled = Function(self.P1_ten).assign(0.0)
        kernel = eigen_kernel(matscale, self.mesh.topological_dimension())
        op2.par_loop(kernel, self.P1.node_set, H_scaled.dat(op2.RW), H.dat(op2.READ), self.indicator.dat(op2.READ))
        self.M = steady_metric(self.solution if adjoint else self.adjoint_solution, H=H, op=self.op)

    def plot(self):
        """
        Plot current mesh and indicator fields, if available.
        """
        File(os.path.join(self.di, 'mesh.pvd')).write(self.mesh.coordinates)
        for key in self.indicators:
            File(os.path.join(self.di, key + '.pvd')).write(self.indicators[key])

    def plot_solution(self, adjoint=False):
        """
        Plot solution of forward or adjoint PDE to .vtu, as specified by the boolean kwarg
        `adjoint`.
        """
        if adjoint:
            self.adjoint_solution_file.write(self.adjoint_solution)
        else:
            self.solution_file.write(self.solution)

    def indicate_error(self):
        """
        Evaluate error estimation strategy of choice in order to obtain a metric field for mesh
        adaptation.

        NOTE: User-provided metrics may be applied by defining a `custom_adapt` method.
        """
        approach = self.approach
        adjoint = 'adjoint' in approach
        average = 'relax' in approach
        # TODO:
        #  * Change 'relax' to 'average'
        #  * Remove all author names
        #  * Integrate AnisotropicMetricDriver

        # No adaptation
        if approach == 'fixed_mesh':  # TODO: Special case estimator convergence
            return

        # Global AMR
        elif approach == 'uniform':  # TODO: Special case estimator convergence
            return

        # Anisotropic Hessian-based adaptation
        elif 'hessian' in approach:  # TODO: Special case estimator convergence
            if approach in ('hessian', 'hessian_adjoint'):
                self.get_hessian_metric(adjoint=adjoint)
            else:
                self.get_hessian_metric(adjoint=False)
                M = self.M.copy()
                self.get_hessian_metric(adjoint=True)
                self.M = combine_metrics(M, self.M, average=average)

        # Isotropic 'Dual Weighted Primal' (see [Davis & LeVeque, 2016])
        elif approach == 'dwp':
            self.dwp_indication()
            self.get_isotropic_metric()

        # Isotropic Dual Weighted Residual (see [Becker & Rannacher, 2001])
        elif 'dwr' in approach:
            self.dwr_indication(adjoint=adjoint)
            self.get_isotropic_metric()
            if not self.approach in ('dwr', 'dwr_adjoint'):
                i = self.indicator.copy()
                M = self.M.copy()
                self.dwr_indication(adjoint=not adjoint)
                if not approach in self.estimators:
                    self.estimators[approach] = []
                self.get_isotropic_metric()
                if approach in ('dwr_both', 'dwr_averaged'):
                    self.indicator.interpolate(Constant(0.5)*(i+self.indicator))
                    self.get_isotropic_metric()
                else:
                    self.M = combine_metrics(M, self.M, average=average)
                estimator = self.estimators['dwr'][-1] + self.estimators['dwr_adjoint'][-1]
                self.estimators[approach].append(estimator)

        # Anisotropic a priori (see [Loseille et al., 2010])
        elif 'loseille' in approach:  # TODO: Special case estimator convergence
            self.get_loseille_metric(adjoint=adjoint)
            if not approach in ('loseille', 'loseille_adjoint'):
                M = self.M.copy()
                self.get_loseille_metric(adjoint=not adjoint)
                self.M = combine_metrics(M, self.M, average=average)

        # Anisotropic a posteriori (see [Power et al., 2006])
        elif 'power' in approach:
            self.get_power_metric(adjoint=adjoint)
            self.indicators[approach] = self.indicators['_'.join(['cell_residual', 'adjoint' if adjoint else 'forward'])].copy()
            if not approach in ('power', 'power_adjoint'):
                M = self.M.copy()
                self.get_power_metric(adjoint=not adjoint)
                self.indicators[approach] += self.indicators['_'.join(['cell_residual', 'adjoint' if not adjoint else 'forward'])].copy()
                self.M = combine_metrics(M, self.M, average=average)
            self.estimate_error()

        # Isotropic a posteriori (see Carpio et al., 2013])
        elif 'carpio_isotropic' in approach:
            self.dwr_indication(adjoint=adjoint)
            self.indicators[approach] = Function(self.P0)
            self.indicators[approach] += self.indicators['dwr_adjoint' if adjoint else 'dwr']
            if approach == 'carpio_isotropic_both':
                self.dwr_indication(adjoint=not adjoint)
                self.indicators[approach] += self.indicators['dwr_adjoint' if not adjoint else 'dwr']
            amd = AnisotropicMetricDriver(self.am, indicator=self.indicators[approach], op=self.op)
            amd.get_isotropic_metric()
            self.M = amd.p1metric
            self.estimate_error()

        # Aniotropic a posteriori (see Carpio et al., 2013])
        elif approach == 'carpio_both':
            self.dwr_indication(adjoint=False)
            self.indicators[approach] = self.indicators['dwr'].copy()
            self.get_hessian_metric(noscale=False, degree=1)  # NOTE: degree 0 doesn't work
            M = self.M.copy()
            self.dwr_indication(adjoint=True)
            self.indicators[approach] += self.indicators['dwr_adjoint']
            self.get_hessian_metric(noscale=False, degree=1, adjoint=True)
            self.M = metric_intersection(self.M, M)
            amd = AnisotropicMetricDriver(self.am, hessian=self.M, indicator=self.indicators[approach], op=self.op)
            amd.get_anisotropic_metric()
            self.M = amd.p1metric
            self.estimate_error()
        elif 'carpio' in approach:
            self.dwr_indication(adjoint=adjoint)
            self.indicators[approach] = self.indicators['dwr_adjoint' if adjoint else 'dwr']
            self.get_hessian_metric(noscale=True, degree=1, adjoint=adjoint)
            amd = AnisotropicMetricDriver(self.am, hessian=self.M, indicator=self.indicators[approach], op=self.op)
            amd.get_anisotropic_metric()
            self.M = amd.p1metric
            self.estimate_error()

        elif self.approach == 'monge_ampere':
            return

        # User specified adaptation methods
        else:
            try:
                assert hasattr(self, 'custom_adapt')
            except AssertionError:
                raise ValueError("Adaptivity mode {:s} not regcognised.".format(approach))
            print_output("Using custom metric '{:s}'".format(approach))
            self.custom_adapt()

    def estimate_error(self, approach=None):
        """
        Compute error estimator associated with `approach` by summing the corresponding error
        indicator over all elements.
        """
        approach = approach or self.approach
        assert approach in self.indicators
        if not approach in self.estimators:
            self.estimators[approach] = []
        self.estimators[approach].append(self.indicators[approach].vector().gather().sum())

    def plot_error_estimate(self, approach):
        raise NotImplementedError  # TODO

    def adapt_mesh(self):  # TODO: option for r-adaptation every *timestep*, rather than every export
        """
        Adapt mesh using metric constructed in error estimation step.

        NOTE: User-provided metrics may be applied by defining a `custom_adapt` method.
        """
        if 'fixed_mesh' in self.approach:
            return
        elif self.approach == 'uniform':
            if self.am.levels == 0:
                raise ValueError("Cannot perform uniform refinement because `AdaptiveMesh` object is not hierarchical.")
            self.mesh = self.am.hierarchy[1]
            return
        elif self.approach == 'monge_ampere':  # TODO: Integrate into AdaptiveMesh
            try:
                assert hasattr(self, 'monitor_function')
            except AssertionError:
                raise ValueError("Please supply a monitor function.")

            # Create MeshMover object and establish coordinate transformation
            mesh_mover = MeshMover(self.am_init.mesh, self.monitor_function, op=self.op)
            mesh_mover.adapt()

            # Create a temporary Problem based on the new mesh
            am_copy = self.am.copy()
            tmp = type(self)(self.op, mesh=am_copy.mesh, discrete_adjoint=self.discrete_adjoint,
                             prev_solution=self.prev_solution, levels=self.levels)
            x = Function(tmp.mesh.coordinates)
            x.dat.data[:] = mesh_mover.x.dat.data  # FIXME: Not parallel
            tmp.mesh.coordinates.assign(x)  # TODO: Need to modify coords of hierarchy, too

            # Project fields and solutions onto temporary Problem
            tmp.project_fields(self)
            tmp.project_solution(self.solution)
            tmp.project_solution(self.adjoint_solution, adjoint=True)

            # Update self.mesh and function spaces, etc.
            self.mesh.coordinates.dat.data[:] = x.dat.data  # FIXME: Not parallel
            self.create_function_spaces()
            self.create_solutions()
            self.boundary_conditions = self.op.set_boundary_conditions(self.V)
            self.project_fields(tmp)
            self.project_solution(tmp.solution)
            self.project_solution(tmp.adjoint_solution, adjoint=True)

            # Plot monitor function
            m = interpolate(self.monitor_function(self.mesh), self.P1)
            m.rename("Monitor function")
            self.monitor_file.write(m)
        else:
            try:
                assert hasattr(self, 'M')
            except AssertionError:
                raise ValueError("Please supply a metric.")
            self.am.pragmatic_adapt(self.M)
            self.set_mesh(self.am.mesh)

        if self.approach != 'monge_ampere':
            print_output("Done adapting. Number of elements: {:d}".format(self.mesh.num_cells()))
            self.num_cells.append(self.mesh.num_cells())
            self.num_vertices.append(self.mesh.num_vertices())
            self.plot()

            # Re-initialise problem
            self.create_function_spaces()
            self.create_solutions()
            self.set_fields()
            self.boundary_conditions = self.op.set_boundary_conditions(self.V)

    def initialise_mesh(self):
        """
        Repeatedly apply mesh adaptation in order to give a suitable initial mesh. A common usage
        is when bathymetry is interpolated from raw data and we want its anisotropy to align with
        that of the mesh.

        NOTE: `self.set_fields` will be called after each adaptation step.
        """
        for i in range(self.op.num_adapt):
            self.indicate_error()
            self.adapt_mesh()

    def adaptation_loop(self, outer_iteration=None):
        """
        Run mesh adaptation loop to convergence, with the following convergence criteria:
          * Relative difference in quantity of interest < `self.op.qoi_rtol`;
          * Relative difference in number of mesh elements < `self.op.element_rtol`;
          * Relative difference in error estimator < `self.op.estimator_rtol`;
          * Maximum iterations `self.op.num_adapt`.

        Error estimator, QoI, element count and vertex count are stored, unless the maximum
        iteration count is reached, or the element count goes below 200.
        """
        op = self.op
        qoi_old = np.finfo(float).min
        num_cells_old = np.iinfo(int).min
        estimator_old = np.finfo(float).min
        print_output("Number of mesh elements: %d" % self.mesh.num_cells())
        for i in range(op.num_adapt):
            if outer_iteration is None:
                print_output("\n  Adaptation loop, iteration %d." % (i+1))
            else:
                print_output("\n  Adaptation loop %d, iteration %d." % (outer_iteration, i+1))
            print_output("====================================\n")
            self.solve_forward()
            qoi = self.quantity_of_interest()
            self.qois.append(qoi)
            print_output("Quantity of interest: %.4e" % qoi)
            if i > 0 and np.abs(qoi - qoi_old) < op.qoi_rtol*qoi_old:
                print_output("Converged quantity of interest!")
                break
            self.solve_adjoint()
            self.indicate_error()
            estimator = self.estimators[self.approach][-1]
            print_output("Error estimator '%s': %.4e" % (self.approach, estimator))
            if i > 0 and np.abs(estimator - estimator_old) < op.estimator_rtol*estimator_old:
                print_output("Converged error estimator!")
                break
            self.adapt_mesh()
            num_cells = self.mesh.num_cells()
            print_output("Number of mesh elements: %d" % num_cells)
            if i > 0 and np.abs(num_cells - num_cells_old) < op.element_rtol*num_cells_old:
                print_output("Converged number of mesh elements!")
                break
            if i == op.num_adapt-1 or num_cells < 200:
                print_output("Adaptation loop failed to converge in {:d} iterations".format(i+1))
                return
            qoi_old = qoi
            num_cells_old = num_cells
            estimator_old = estimator
            dofs = self.V.dof_count
            dofs = dofs if isinstance(dofs, int) else sum(dofs)  # TODO: parallelise
        if outer_iteration is None:
            print_output('\n' + 80*'#' + '\n' + 37*' ' + 'SUMMARY\n' + 80*'#')
            print_output("Approach:             '{:s}'".format(self.approach))
            print_output("Target:               {:.2e}".format(op.target))
            print_output("Number of elements:   {:d}".format(num_cells))
            print_output("DOF count:            {:d}".format(dofs))
            print_output("Quantity of interest: {:.4e}".format(qoi))
            print_output('\n' + 80*'#')
        self.outer_estimators.append(self.estimators[self.approach][-1])
        self.outer_num_cells.append(self.num_cells[-1])
        self.outer_num_vertices.append(self.num_vertices[-1])
        self.outer_qois.append(self.qois[-1])

    def outer_adaptation_loop(self):
        """
        Perform multiple adaptation loops, with the target complexity/error increasing/decaying by
        a constant factor.
        """
        op = self.op
        initial_target = op.target
        for i in range(op.outer_iterations):
            op.target = initial_target*op.target_base**i
            op.set_default_mesh()  # TODO: Temporary
            self.set_mesh(op.default_mesh)
            self.create_function_spaces()
            self.create_solutions()
            self.set_fields()
            self.boundary_conditions = op.set_boundary_conditions(self.V)
            self.adaptation_loop(outer_iteration=i+1)
            if i < op.outer_iterations-1:
                print_output('\n************************************\n')

    def plot_qoi_convergence(self):
        """
        Convergence plot of QoI against number of elements, taken over an outer adaptation loop.
        """
        # TODO: Nicer formatting
        plt.semilogx(self.outer_num_cells, self.outer_qois)
        plt.xlabel("Number of elements")
        plt.ylabel("Quantity of interest")

    def check_conditioning(self, submatrices=None):  # TODO: Account for RHS
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
                print_output("Condition number {:1d},{:1d}: {:.4e}".format(s[0], s[1], kappa))
        else:
            cc = UnnestedConditionCheck(self.lhs)
            self.condition_number = cc.condition_number()
            print_output("Condition number: {:.4e}".format(self.condition_number))


class UnsteadyProblem(SteadyProblem):
    """
    Base class for solving time-dependent PDE problems using mesh adaptivity.

    There are three main functionalities:
        * solve PDE;
        * solve adjoint PDE;
        * adapt mesh based on some error estimator of choice.
    """
    def __init__(self, op, mesh, finite_element, discrete_adjoint=False, prev_solution=None, levels=1):
        super(UnsteadyProblem, self).__init__(op, mesh, finite_element, discrete_adjoint, prev_solution, levels)
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
        self.set_solution(self.op.set_start_condition(self.V, adjoint=adjoint), adjoint)

    def solve_step(self, adjoint=False):
        """
        Solve forward PDE on a particular mesh.
        """
        raise NotImplementedError("Should be implemented in derived class.")

    def solve(self, adjoint=False, uses_adjoint=True):
        """
        Solve PDE using mesh adaptivity.
        """
        self.remesh_step = 0
        uses_adjoint &= not 'fixed_mesh' in self.approach
        uses_adjoint &= self.approach != 'hessian'

        # Adapt w.r.t. initial conditions a few times before the solver loop
        if uses_adjoint:
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
            # NOTE:
            #  * Use 'fixed_mesh_plot' to call `plot` at each export.
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
                    self.project_solution(solution, adjoint=adjoint)
                else:
                    self.project_solution(solution_old, adjoint=adjoint)

                # Solve PDE on new mesh
                self.op.plot_pvd = i == 0
                # time = None if i == 0 else self.step_end - self.op.dt
                # self.solve_step(adjoint=adjoint, time=time)
                self.solve_step(adjoint=adjoint)

                # Store solutions from last two steps on first mesh in sequence
                if i == 0:
                    solution = Function(self.solution)
                    solution_old = Function(self.solution_old)
                    if self.step_end + self.op.dt*self.op.dt_per_remesh > self.op.end_time:
                        break  # No need to do adapt for final timestep
            # self.plot()  # TODO: Temporary. It is called at end of adapt_mesh

            self.step_end += self.op.dt*self.op.dt_per_remesh
            self.remesh_step += 1

        # Evaluate QoI
        if uses_adjoint:
            self.project_solution(solution)
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
