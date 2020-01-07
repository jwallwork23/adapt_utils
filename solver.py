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
from adapt_utils.adapt.kernels import matscale_kernel, include_dir


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

        # Build AdaptiveMesh object
        mesh = op.default_mesh if mesh is None else mesh
        self.am = AdaptiveMesh(mesh, levels=levels)
        self.mesh = self.am.mesh

        # Create equivalent problem in enriched space
        if levels > 0:
            fe = FiniteElement(finite_element.family(),
                               finite_element.cell(),
                               finite_element.degree()+1)
            self.tp_enriched = type(self)(op, self.am.refined_mesh, fe, discrete_adjoint, prev_solution, levels-1)

        # Function spaces and mesh quantities
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

        # Prognostic fields
        self.solution = Function(self.V, name='Solution')
        self.adjoint_solution = Function(self.V, name='Adjoint solution')

        # Outputs
        self.di = create_directory(self.op.di)
        self.solution_file = File(os.path.join(self.di, 'solution.pvd'))
        self.adjoint_solution_file = File(os.path.join(self.di, 'adjoint_solution.pvd'))

        # Error estimator/indicator storage
        self.estimators = {}
        self.indicators = {}

    def set_target_vertices(self, num_vertices=None):
        """
        Set target number of vertices for adapted mesh by scaling the current number of vertices.
        """
        if num_vertices is None:
            num_vertices = self.mesh.num_vertices()
        self.op.target = num_vertices*self.op.rescaling

    def solve(self):
        """
        Solve forward PDE.
        """
        pass

    def get_qoi_kernel(self):
        """
        Derivative `g` of functional of interest `J`. i.e. For solution `u` we have
            J(u) = g . u
        """
        pass

    def quantity_of_interest(self):
        """
        Functional of interest which takes the PDE solution as input.
        """
        if not hasattr(self, 'kernel'):
            self.get_qoi_kernel()
        return assemble(inner(self.solution, self.kernel)*dx(degree=12))

    def solve_continuous_adjoint(self):
        """
        Solve the adjoint PDE using a hand-coded continuous adjoint.
        """
        pass

    def solve_discrete_adjoint(self):
        """
        Solve the adjoint PDE in the discrete sense.
        """
        pass

    def solve_adjoint(self):
        """
        Solve adjoint problem using specified method.
        """
        if self.discrete_adjoint:
            PETSc.Sys.Print("Solving discrete adjoint problem...")
            self.solve_discrete_adjoint()
        else:
            PETSc.Sys.Print("Solving continuous adjoint problem...")
            self.solve_continuous_adjoint()

    def dwp_indication(self):
        """
        Indicate significance by the product of forward and adjoint solutions. This approach was
        used for mesh adaptive tsunami modelling in [Davis and LeVeque, 2016]. Here 'DWP' is used
        to stand for Dual Weighted Primal.
        """
        self.indicator = Function(self.P1)
        self.indicator.project(inner(self.solution, self.adjoint_solution))
        self.indicator.rename('dwp')

    def get_strong_residual(self, adjoint=False):
        if adjoint:
            self.get_strong_residual_forward()
        else:
            self.get_strong_residual_adjoint()

    def get_strong_residual_forward(self):
        pass

    def get_strong_residual_adjoint(self):
        pass

    def get_dwr_residual(self, sol, adjoint_sol, adjoint=False):
        pass

    def get_dwr_flux(self, sol, adjoint_sol, adjoint=False):
        pass

    def plot(self):
        """
        Plot current mesh and indicator field, if available.
        """
        File(os.path.join(self.di, 'mesh.pvd')).write(self.mesh.coordinates)
        if hasattr(self, 'indicator'):
            name = self.indicator.dat.name
            self.indicator.rename(' '.join([name, 'indicator']))
            File(os.path.join(self.di, 'indicator.pvd')).write(self.indicator)

    def dwr_indication(self, adjoint=False):
        """
        Indicate errors in the quantity of interest by the Dual Weighted Residual method. This is
        inherently problem-dependent.

        The resulting P0 field should be stored as `self.indicator`.
        """
        label = 'dwr'
        if adjoint:
            label += '_adjoint'
        self.get_dwr_residual(self.solution, self.adjoint_solution, adjoint=adjoint)
        self.get_dwr_flux(self.solution, self.adjoint_solution, adjoint=adjoint)
        self.indicator = Function(self.P1, name=label)
        self.indicator.interpolate(abs(self.indicators['dwr_cell'] + self.indicators['dwr_flux']))
        self.estimators[label] = self.estimators['dwr_cell'] + self.estimators['dwr_flux']
        self.indicators[label] = self.indicator

    def get_hessian(self, adjoint=False):
        """
        Compute an appropriate Hessian for the problem at hand. This is inherently
        problem-dependent, since the choice of field for adaptation is not universal.
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
        Scale an identity matrix by the indicator field in order to drive
        isotropic mesh refinement.
        """
        el = self.indicator.ufl_element()
        name = self.indicator.dat.name
        if (el.family(), el.degree()) != ('Lagrange', 1):
            self.indicator = project(self.indicator, self.P1)
            self.indicator.rename(name)
        self.M = isotropic_metric(self.indicator, op=self.op)

    def get_loseille_metric(self, adjoint=False, relax=False):
        """
        Construct an anisotropic metric using an approach inspired by [Loseille et al. 2009].
        """
        pass

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
        kernel = op2.Kernel(matscale_kernel(dim), "matscale", cpp=True, include_dirs=include_dir)
        op2.par_loop(kernel, self.P1.node_set, H_scaled.dat(op2.RW), H.dat(op2.READ), self.indicator.dat(op2.READ))
        self.M = steady_metric(self.solution if adjoint else self.adjoint_solution, H=H, op=self.op)

    def indicate_error(self):
        """
        Evaluate error estimation strategy of choice in order to obtain a metric field for mesh
        adaptation.

        NOTE: User-provided metrics may be applied by defining a `custom_adapt` method.
        """
        if self.approach == 'fixed_mesh':
            return
        elif self.approach == 'uniform':
            try:
                assert self.am.levels > 1
            except ValueError:
                raise ValueError("Cannot perform uniform refinement because `AdaptiveMesh` object is not hierarchical.")
            self.mesh = self.am.hierarchy[1]
        elif self.approach == 'hessian':
            self.get_hessian_metric()
        elif self.approach == 'hessian_adjoint':
            self.get_hessian_metric(adjoint=True)
        elif self.approach == 'hessian_relaxed':
            self.get_hessian_metric(adjoint=False)
            M = self.M.copy()
            self.get_hessian_metric(adjoint=True)
            self.M = metric_relaxation(M, self.M)
        elif self.approach == 'hessian_superposed':
            self.get_hessian_metric(adjoint=False)
            M = self.M.copy()
            self.get_hessian_metric(adjoint=True)
            self.M = metric_intersection(M, self.M)
        elif self.approach == 'dwp':
            self.dwp_indication()
            self.get_isotropic_metric()
        elif self.approach == 'dwr':
            self.dwr_indication()
            self.get_isotropic_metric()
        elif self.approach == 'dwr_adjoint':
            self.dwr_indication(adjoint=True)
            self.get_isotropic_metric()
        elif self.approach == 'dwr_both':
            self.dwr_indication()
            self.get_isotropic_metric()
            i = self.indicator.copy()
            self.dwr_indication(adjoint=True)
            self.indicator.interpolate(Constant(0.5)*(i+self.indicator))
            self.get_isotropic_metric()
        elif self.approach == 'dwr_averaged':
            self.dwr_indication()
            self.get_isotropic_metric()
            i = self.indicator.copy()
            self.dwr_indication(adjoint=True)
            self.indicator.interpolate(Constant(0.5)*(abs(i)+abs(self.indicator)))
            self.get_isotropic_metric()
        elif self.approach == 'dwr_relaxed':
            self.dwr_indication()
            self.get_isotropic_metric()
            M = self.M.copy()
            self.dwr_indication(adjoint=True)
            self.get_isotropic_metric()
            self.M = metric_relaxation(M, self.M)
        elif self.approach == 'dwr_superposed':
            self.dwr_indication()
            self.get_isotropic_metric()
            M = self.M.copy()
            self.dwr_indication(adjoint=True)
            self.get_isotropic_metric()
            self.M = metric_intersection(M, self.M)
        elif self.approach == 'loseille':
            self.get_loseille_metric(adjoint=False)
        elif self.approach == 'loseille_adjoint':
            self.get_loseille_metric(adjoint=True)
        elif self.approach == 'loseille_relaxed':
            self.get_loseille_metric(adjoint=False)
            M = self.M.copy()
            self.get_loseille_metric(adjoint=True)
            self.M = metric_relaxation(M, self.M)
        elif self.approach == 'loseille_superposed':
            self.get_loseille_metric(adjoint=False)
            M = self.M.copy()
            self.get_loseille_metric(adjoint=True)
            self.M = metric_intersection(M, self.M)
        elif self.approach == 'power':
            self.get_power_metric(adjoint=False)
        elif self.approach == 'power_adjoint':
            self.get_power_metric(adjoint=True)
        elif self.approach == 'power_relaxed':
            self.get_power_metric(adjoint=False)
            M = self.M.copy()
            self.get_power_metric(adjoint=True)
            self.M = metric_relaxation(M, self.M)
        elif self.approach == 'power_superposed':
            self.get_power_metric(adjoint=False)
            M = self.M.copy()
            self.get_power_metric(adjoint=True)
            # self.M = metric_intersection(self.M, M)
            self.M = metric_intersection(M, self.M)
        elif self.approach == 'carpio_isotropic':
            self.dwr_indication()
            amd = AnisotropicMetricDriver(self.mesh, indicator=self.indicator, op=self.op)
            amd.get_isotropic_metric()
            self.M = amd.p1metric
        elif self.approach == 'carpio_isotropic_adjoint':
            self.dwr_indication(adjoint=True)
            amd = AnisotropicMetricDriver(self.mesh, indicator=self.indicator, op=self.op)
            amd.get_isotropic_metric()
            self.M = amd.p1metric
        elif self.approach == 'carpio_isotropic_both':
            self.dwr_indication()
            i = self.indicator.copy()
            self.dwr_indication(adjoint=True)
            eta = Function(self.P0).interpolate(i + self.indicator)
            amd = AnisotropicMetricDriver(self.mesh, indicator=eta, op=self.op)
            amd.get_isotropic_metric()
            self.M = amd.p1metric
        elif self.approach == 'carpio':
            self.dwr_indication()
            self.get_hessian_metric(noscale=True, degree=1)  # NOTE: degree 0 doesn't work
            amd = AnisotropicMetricDriver(self.mesh, hessian=self.M, indicator=self.indicator, op=self.op)
            amd.get_anisotropic_metric()
            self.M = amd.p1metric
        elif self.approach == 'carpio_adjoint':
            self.dwr_indication(adjoint=True)
            self.get_hessian_metric(noscale=True, degree=1, adjoint=True)
            amd = AnisotropicMetricDriver(self.mesh, hessian=self.M, indicator=self.indicator, op=self.op)
            amd.get_anisotropic_metric()
            self.M = amd.p1metric
        elif self.approach == 'carpio_both':
            self.dwr_indication()
            i = self.indicator.copy()
            self.get_hessian_metric(noscale=False, degree=1)
            M = self.M.copy()
            self.dwr_indication(adjoint=True)
            self.get_hessian_metric(noscale=False, degree=1, adjoint=True)
            self.indicator.interpolate(i + self.indicator)
            self.M = metric_intersection(self.M, M)
            amd = AnisotropicMetricDriver(self.mesh, hessian=self.M, indicator=self.indicator, op=self.op)
            amd.get_anisotropic_metric()
            self.M = amd.p1metric
        else:
            try:
                assert hasattr(self, 'custom_adapt')
            except:
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
            self.mesh = MeshHierarchy(self.mesh, 1)[1]
            return
        else:
            if not hasattr(self, 'M'):
                PETSc.Sys.Print("Metric not found. Computing it now.")
                self.indicate_error()
            self.am.adapt(self.M)
            self.mesh = self.am.mesh
            # self.mesh = Mesh(adapt(self.mesh, self.M).coordinates)
        PETSc.Sys.Print("Done adapting. Number of elements: {:d}".format(self.mesh.num_cells()))
        self.plot()

    def check_conditioning(self, submatrices=None):
        """
        Check condition number of LHS matrix, or submatrices thereof.

        :kwarg submatries: indices for desired submatrices (default all with `None`).
        """
        try:
            assert hasattr(self, 'lhs')
        except:
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


class UnsteadyProblem():
    """
    Base class for solving time-dependent PDE problems using mesh adaptivity.

    There are three main functionalities:
        * solve PDE;
        * solve adjoint PDE;
        * adapt mesh based on some error estimator of choice.
    """
    def __init__(self, op, mesh, finite_element, discrete_adjoint=False, levels=1):

        # Read args and kwargs
        self.finite_element = finite_element
        self.discrete_adjoint = discrete_adjoint
        self.op = op
        self.stab = op.stabilisation
        self.approach = op.approach

        # Construct AdaptiveMesh object
        mesh = op.default_mesh if mesh is None else mesh
        self.am = AdaptiveMesh(mesh, levels=levels)
        self.mesh = self.am.mesh

        # Function spaces and mesh quantities
        self.V = FunctionSpace(self.mesh, self.finite_element)
        self.P0 = FunctionSpace(self.mesh, "DG", 0)
        self.P1 = FunctionSpace(self.mesh, "CG", 1)
        self.P1DG = FunctionSpace(self.mesh, "DG", 1)
        self.P1_vec = VectorFunctionSpace(self.mesh, "CG", 1)
        self.P1_ten = TensorFunctionSpace(self.mesh, "CG", 1)
        self.test = TestFunction(self.V)
        self.trial = TrialFunction(self.V)
        self.p0test = TestFunction(self.P0)
        self.p0trial = TrialFunction(self.P0)
        self.n = FacetNormal(self.mesh)
        self.h = CellSize(self.mesh)

        # Prognostic fields
        self.solution = Function(self.V, name='Solution')
        self.solution_old = Function(self.V, name='Old solution')
        self.adjoint_solution = Function(self.V, name='Adjoint solution')
        self.adjoint_solution_old = Function(self.V, name='Old adjoint solution')
        self.set_start_condition()

        # Outputs
        self.di = create_directory(self.op.di)
        self.solution_file = File(os.path.join(self.di, 'solution.pvd'))
        self.adjoint_solution_file = File(os.path.join(self.di, 'adjoint_solution.pvd'))
        self.indicator_file = File(os.path.join(self.di, 'indicator.pvd'))

        # Adaptivity
        self.step_end = op.end_time if self.approach == 'fixed_mesh' else op.dt*op.dt_per_remesh
        self.estimators = {}
        self.indicators = {}
        self.num_exports = int(np.floor((op.end_time - op.dt)/op.dt/op.dt_per_export))

    def set_target_vertices(self, rescaling=0.85, num_vertices=None):
        """
        Set target number of vertices for adapted mesh by scaling the current number of vertices.
        """
        if num_vertices is None:
            num_vertices = self.mesh.num_vertices()
        self.op.target = num_vertices * rescaling

    def solve_step(self, adjoint=False, **kwargs):
        """
        Solve forward PDE on a particular mesh.
        """
        pass

    def set_start_condition(self, adjoint=False):
        if adjoint:
            self.adjoint_solution = self.op.set_final_condition(self.V)
        else:
            self.solution = self.op.set_initial_condition(self.V)

    def set_fields(self):
        """
        Set velocity field, viscosity, QoI kernel, etc.
        """
        pass

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

    def get_qoi_kernel(self):
        """
        Derivative `g` of functional of interest `J`. i.e. For solution `u` we have
            J(u) = g . u
        """
        pass

    def quantity_of_interest(self):  # TODO: account for time integral forms
        """
        Functional of interest which takes the PDE solution as input.
        """
        if not hasattr(self, 'kernel'):
            self.get_qoi_kernel()
        return assemble(inner(self.solution, self.kernel)*dx(degree=12))

    def solve_continuous_adjoint(self):
        """
        Solve the adjoint PDE using a hand-coded continuous adjoint.
        """
        self.solve(adjoint=True)

    def solve_adjoint(self):
        """
        Solve adjoint problem using specified method.
        """
        PETSc.Sys.Print("Solving adjoint problem...")
        self.solve_continuous_adjoint()

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

    def dwp_indication(self):
        """
        Indicate significance by the product of forward and adjoint solutions. This approach was
        used for mesh adaptive tsunami modelling in [Davis and LeVeque, 2016]. Here 'DWP' is used
        to stand for Dual Weighted Primal.
        """
        self.indicator = Function(self.P1)
        self.indicator.project(inner(self.solution, self.adjoint_solution))
        self.indicator.rename('dwp')

    def get_strong_residual(self, adjoint=False):
        if adjoint:
            self.get_strong_residual_forward()
        else:
            self.get_strong_residual_adjoint()

    def get_strong_residual_forward(self):
        pass

    def get_strong_residual_adjoint(self):
        pass

    def plot(self):
        """
        Plot current mesh and indicator field, if available.
        """
        if hasattr(self, 'indicator'):
            self.indicator.rename(' '.join([self.approach, 'indicator']))
            self.indicator_file.write(self.indicator, t=self.remesh_step*self.op.dt)

    def dwr_indication(self, adjoint=False):
        """
        Indicate errors in the quantity of interest by the Dual Weighted Residual method. This is
        inherently problem-dependent.

        The resulting P0 field should be stored as `self.indicator`.
        """
        pass  # TODO: Use steady format, but need to get adjoint sol

    def get_hessian(self, adjoint=False):
        """
        Compute an appropriate Hessian for the problem at hand. This is inherently
        problem-dependent, since the choice of field for adaptation is not universal.
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
        name = self.indicator.dat.name
        if (el.family(), el.degree()) != ('Lagrange', 1):
            self.indicator = project(self.indicator, self.P1)
            self.indicator.rename(name)
        self.M = isotropic_metric(self.indicator, op=self.op)

    def get_loseille_metric(self, adjoint=False, relax=False):
        """
        Construct an anisotropic metric using an approach inspired by [Loseille et al. 2009].
        """
        pass

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
        kernel = op2.Kernel(matscale_kernel(dim), "matscale", cpp=True, include_dirs=include_dir)
        op2.par_loop(kernel, self.P1.node_set, H_scaled.dat(op2.RW), H.dat(op2.READ), self.indicator.dat(op2.READ))
        self.M = steady_metric(self.solution if adjoint else self.adjoint_solution, H=H, op=self.op)


    def adapt_mesh(self):
        """
        Adapt mesh according to error estimation strategy of choice.

        NOTE: User-provided metrics may be applied by defining a `custom_adapt` method.

        between metrics at the current and previous steps.
        :kwarg estimate_error: Toggle computation of global error estimate.
        """
        if self.approach == 'fixed_mesh':
            return
        elif self.approach == 'uniform':
            try:
                assert self.am.levels > 1
            except ValueError:
                raise ValueError("Cannot perform uniform refinement because `AdaptiveMesh` object is not hierarchical.")
            self.mesh = self.am.hierarchy[1]
            return
        elif self.approach == 'hessian':
            self.get_hessian_metric()
        elif self.approach == 'hessian_adjoint':
            self.get_hessian_metric(adjoint=True)
        elif self.approach == 'hessian_relaxed':
            self.get_hessian_metric(adjoint=False)
            M = self.M.copy()
            self.get_hessian_metric(adjoint=True)
            self.M = metric_relaxation(M, self.M)
        elif self.approach == 'hessian_superposed':
            self.get_hessian_metric(adjoint=False)
            M = self.M.copy()
            self.get_hessian_metric(adjoint=True)
            self.M = metric_intersection(M, self.M)
        elif self.approach == 'dwp':
            self.dwp_indication()
            self.get_isotropic_metric()
        elif self.approach == 'dwr':
            self.dwr_indication()
            self.get_isotropic_metric()
        elif self.approach == 'dwr_adjoint':
            self.dwr_indication(adjoint=True)
            self.get_isotropic_metric()
        elif self.approach == 'dwr_both':
            self.dwr_indication()
            self.get_isotropic_metric()
            i = self.indicator.copy()
            self.dwr_indication(adjoint=True)
            self.indicator.interpolate(Constant(0.5)*(i+self.indicator))
            self.get_isotropic_metric()
        elif self.approach == 'dwr_averaged':
            self.dwr_indication()
            self.get_isotropic_metric()
            i = self.indicator.copy()
            self.dwr_indication(adjoint=True)
            self.indicator.interpolate(Constant(0.5)*(abs(i)+abs(self.indicator)))
            self.get_isotropic_metric()
        elif self.approach == 'dwr_relaxed':
            self.dwr_indication()
            self.get_isotropic_metric()
            M = self.M.copy()
            self.dwr_indication(adjoint=True)
            self.get_isotropic_metric()
            self.M = metric_relaxation(M, self.M)
        elif self.approach == 'dwr_superposed':
            self.dwr_indication()
            self.get_isotropic_metric()
            M = self.M.copy()
            self.dwr_indication(adjoint=True)
            self.get_isotropic_metric()
            self.M = metric_intersection(M, self.M)
        elif self.approach == 'loseille':
            self.get_loseille_metric(adjoint=False)
        elif self.approach == 'loseille_adjoint':
            self.get_loseille_metric(adjoint=True)
        elif self.approach == 'loseille_relaxed':
            self.get_loseille_metric(adjoint=False)
            M = self.M.copy()
            self.get_loseille_metric(adjoint=True)
            self.M = metric_relaxation(M, self.M)
        elif self.approach == 'loseille_superposed':
            self.get_loseille_metric(adjoint=False)
            M = self.M.copy()
            self.get_loseille_metric(adjoint=True)
            self.M = metric_intersection(M, self.M)
        elif self.approach == 'power':
            self.get_power_metric(adjoint=False)
        elif self.approach == 'power_adjoint':
            self.get_power_metric(adjoint=True)
        elif self.approach == 'power_relaxed':
            self.get_power_metric(adjoint=False)
            M = self.M.copy()
            self.get_power_metric(adjoint=True)
            self.M = metric_relaxation(M, self.M)
        elif self.approach == 'power_superposed':
            self.get_power_metric(adjoint=False)
            M = self.M.copy()
            self.get_power_metric(adjoint=True)
            # self.M = metric_intersection(self.M, M)
            self.M = metric_intersection(M, self.M)
        else:
            try:
                assert hasattr(self, 'custom_adapt')
                self.custom_adapt()
            except:
                raise ValueError("Adaptivity mode {:s} not regcognised.".format(self.approach))

        # Adapt mesh  # FIXME: The 0.1 factor seems pretty arbitrary
        if self.M is not None: # and norm(self.M) > 0.1*norm(Constant(1, domain=self.mesh)):
            self.am.adapt(self.M)
            self.mesh = self.am.mesh
            # self.mesh = Mesh(adapt(self.mesh, self.M).coordinates)
            PETSc.Sys.Print("Number of elements: %d" % self.mesh.num_cells())

            # Re-establish function spaces
            self.V = FunctionSpace(self.mesh, self.finite_element)
            self.P0 = FunctionSpace(self.mesh, "DG", 0)
            self.P1 = FunctionSpace(self.mesh, "CG", 1)
            self.P1DG = FunctionSpace(self.mesh, "DG", 1)
            self.P1_vec = VectorFunctionSpace(self.mesh, "CG", 1)
            self.P1_ten = TensorFunctionSpace(self.mesh, "CG", 1)
            self.test = TestFunction(self.V)
            self.trial = TrialFunction(self.V)
            self.p0test = TestFunction(self.P0)
            self.p0trial = TrialFunction(self.P0)
            self.n = FacetNormal(self.mesh)
            self.h = CellSize(self.mesh)

            self.solution = Function(self.V, name='Solution')
            self.solution_old = Function(self.V, name='Old solution')
            self.adjoint_solution = Function(self.V, name='Adjoint solution')
            self.adjoint_solution_old = Function(self.V, name='Old adjoint solution')
        else:
            PETSc.Sys.Print("******** WARNING: Adaptation not used ********")
