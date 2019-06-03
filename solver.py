from firedrake import *
from firedrake_adjoint import *
from firedrake.petsc import PETSc
from thetis import create_directory
from fenics_adjoint.solving import SolveBlock       # For extracting adjoint solutions
from fenics_adjoint.projection import ProjectBlock  # Exclude projections from tape reading
import pyadjoint

import datetime
from time import clock
import numpy as np

from adapt_utils.options import DefaultOptions
from adapt_utils.misc import index_string
from adapt_utils.adapt.adaptation import *
from adapt_utils.adapt.metric import *


__all__ = ["SteadyProblem", "UnsteadyProblem", "MeshOptimisation", "OuterLoop"]


now = datetime.datetime.now()
date = str(now.day) + '-' + str(now.month) + '-' + str(now.year % 2000)


class SteadyProblem():
    """
    Base class for solving PDE problems using mesh adaptivity.

    There are three main functionalities:
        * solve PDE;
        * solve adjoint PDE using pyadjoint;
        * adapt mesh based on some error estimator of choice.
    """
    def __init__(self, mesh, op, finite_element, discrete_adjoint=False, prev_solution=None):
        self.mesh = op.default_mesh if mesh is None else mesh
        self.op = op
        self.finite_element = finite_element
        self.stab = op.stabilisation
        self.discrete_adjoint = discrete_adjoint
        self.prev_solution = prev_solution
        self.approach = op.approach

        # function spaces and mesh quantities
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

        # prognostic fields
        self.solution = Function(self.V)
        self.adjoint_solution = Function(self.V)

        # outputs
        self.di = create_directory(self.op.di)
        self.solution_file = File(self.di + 'solution.pvd')
        self.adjoint_solution_file = File(self.di + 'adjoint_solution.pvd')

    def set_target_vertices(self, num_vertices=None):
        """
        Set target number of vertices for adapted mesh by scaling the current number of vertices.
        """
        if num_vertices is None:
            num_vertices = self.mesh.num_vertices()
        self.op.target = num_vertices * self.op.rescaling

    def solve(self):
        """
        Solve forward PDE.
        """
        pass

    def get_objective_kernel(self):
        """
        Derivative `g` of functional of interest `J`. i.e. For solution `u` we have
            J(u) = g . u
        """
        pass

    def objective_functional(self):
        """
        Functional of interest which takes the PDE solution as input.
        """
        if not hasattr(self, 'kernel'):
            self.get_objective_kernel()
        return assemble(inner(self.solution, self.kernel)*dx)

    def solve_continuous_adjoint(self):
        """
        Solve the adjoint PDE using a hand-coded continuous adjoint.
        """
        pass

    def solve_discrete_adjoint(self):
        """
        Solve the adjoint PDE in the discrete sense, using pyadjoint.
        """
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
        self.adjoint_solution.assign(solve_blocks[0].adj_sol)
        tape.clear_tape()

    def solve_adjoint(self):
        """
        Solve adjoint problem using specified method.
        """
        PETSc.Sys.Print("Solving adjoint problem...")
        if self.discrete_adjoint:
            self.solve_discrete_adjoint()
        else:
            self.solve_continuous_adjoint()

    def dwp_indication(self):
        """
        Indicate significance by the product of forward and adjoint solutions. This approach was
        used for mesh adaptive tsunami modelling in [Davis and LeVeque, 2016]. Here 'DWP' is used
        to stand for Dual Weighted Primal.
        """
        self.p1indicator = Function(self.P1)
        self.p1indicator.project(inner(self.solution, self.adjoint_solution))
        self.p1indicator.rename('dwp')

    def explicit_estimation(self, space=None, square=True):
        pass

    def explicit_estimation_adjoint(self, space=None, square=True):
        pass

    def plot(self):
        """
        Plot current mesh and indicator field, if available.
        """
        File(self.di + 'mesh.pvd').write(self.mesh.coordinates)
        if hasattr(self, 'p1indicator'):
            name = self.p1indicator.dat.name
            self.p1indicator.rename(name + ' p1indicator')
            File(self.di + 'p1indicator.pvd').write(self.p1indicator)
        if hasattr(self, 'p0indicator'):
            name = self.p0indicator.dat.name
            self.p0indicator.rename(name + ' p0indicator')
            File(self.di + 'p0indicator.pvd').write(self.p0indicator)

    def dwr_indication(self):
        """
        Indicate errors in the objective functional by the Dual Weighted Residual method. This is
        inherently problem-dependent.

        The resulting P0 field should be stored as `self.p0indicator`.
        """
        pass

    def dwr_indication_adjoint(self):
        pass

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
        #self.p0indicator.interpolate(abs(self.p0indicator))
        if not hasattr(self, 'p1indicator'):
            self.p1indicator = project(self.p0indicator, self.P1)
            self.p1indicator.interpolate(abs(self.p1indicator))  # ensure non-negativity
        self.M = isotropic_metric(self.p1indicator, op=self.op)

    def get_anisotropic_metric(self, adjoint=False, relax=False):
        """
        Apply the approach of [Loseille, Dervieux, Alauzet, 2009] to extract an anisotropic mesh 
        from the Dual Weighted Residual method.
        """
        pass

    def get_power_metric(self, adjoint=False):
        """
        TO DO
        """
        # TODO: doc
        if adjoint:
            self.explicit_estimation_adjoint(square=False)
            self.p1indicator.interpolate(abs(self.p1cell_res_adjoint))
        else:
            self.explicit_estimation(square=False)
            self.p1indicator.interpolate(abs(self.p1cell_res))
        H = self.get_hessian(adjoint=not adjoint)
        for i in range(self.mesh.num_vertices()):
            H.dat.data[i][:,:] *= self.p1indicator.dat.data[i]  # TODO: use pyop2
        if adjoint:
            self.M = steady_metric(self.solution, H=H, op=self.op)
        else:
            self.M = steady_metric(self.adjoint_solution, H=H, op=self.op)

    def estimate_error(self, relaxation_parameter=0.9, prev_metric=None, custom_adapt=None):
        """
        Evaluate error estimation strategy of choice.
        """
        with pyadjoint.stop_annotating():
            if self.approach == 'fixed_mesh':
                return
            elif self.approach == 'uniform':
                self.mesh = MeshHierarchy(self.mesh, 1)[1]
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
            elif self.approach == 'explicit':
                self.explicit_estimation()
                self.get_isotropic_metric()
            elif self.approach == 'explicit_adjoint':
                self.explicit_estimation_adjoint()
                self.get_isotropic_metric()
            elif self.approach == 'explicit_relaxed':
                self.explicit_estimation()
                self.get_isotropic_metric()
                M = self.M.copy()
                self.explicit_estimation_adjoint()
                self.get_isotropic_metric()
                self.M = metric_relaxation(M, self.M)
            elif self.approach == 'explicit_superposed':
                self.explicit_estimation()
                self.get_isotropic_metric()
                M = self.M.copy()
                self.explicit_estimation_adjoint()
                self.get_isotropic_metric()
                self.M = metric_intersection(M, self.M)
            elif self.approach == 'dwp':
                self.dwp_indication()
                self.get_isotropic_metric()
            elif self.approach == 'dwr':
                self.dwr_indication()
                self.get_isotropic_metric()
            elif self.approach == 'dwr_adjoint':
                self.dwr_indication_adjoint()
                self.get_isotropic_metric()
            elif self.approach == 'dwr_both':
                self.dwr_indication()
                self.get_isotropic_metric()
                i = self.p1indicator.copy()
                self.dwr_indication_adjoint()
                self.p1indicator.interpolate(Constant(0.5)*(i+self.p1indicator))
                self.get_isotropic_metric()
            elif self.approach == 'dwr_averaged':
                self.dwr_indication()
                self.get_isotropic_metric()
                i = self.p1indicator.copy()
                self.dwr_indication_adjoint()
                self.p1indicator.interpolate(Constant(0.5)*(abs(i)+abs(self.p1indicator)))
                self.get_isotropic_metric()
            elif self.approach == 'dwr_relaxed':
                self.dwr_indication()
                self.get_isotropic_metric()
                M = self.M.copy()
                self.dwr_indication_adjoint()
                self.get_isotropic_metric()
                self.M = metric_relaxation(M, self.M)
            elif self.approach == 'dwr_superposed':
                self.dwr_indication()
                self.get_isotropic_metric()
                M = self.M.copy()
                self.dwr_indication_adjoint()
                self.get_isotropic_metric()
                self.M = metric_intersection(M, self.M)
            elif self.approach == 'loseille':
                self.get_anisotropic_metric(adjoint=False)
            elif self.approach == 'loseille_adjoint':
                self.get_anisotropic_metric(adjoint=True)
            elif self.approach == 'loseille_relaxed':
                self.get_anisotropic_metric(adjoint=False)
                M = self.M.copy()
                self.get_anisotropic_metric(adjoint=True)
                self.M = metric_relaxation(M, self.M)
            elif self.approach == 'loseille_superposed':
                self.get_anisotropic_metric(adjoint=False)
                M = self.M.copy()
                self.get_anisotropic_metric(adjoint=True)
                self.M = metric_intersection(M, self.M)
            elif self.approach == 'hybrid':
                self.dwr_indication()
                self.get_isotropic_metric()
                M = self.M.copy()
                self.get_anisotropic_metric(adjoint=False)
                self.M = metric_intersection(M, self.M)
            elif self.approach == 'power':
                self.get_power_metric(adjoint=False)
            elif self.approach == 'power_adjoint':
                self.get_power_metric(adjoint=True)
            elif self.approach == 'power_relaxed':
                self.explicit_estimation(square=False)
                self.p1indicator.interpolate(abs(self.p1indicator))
                H = self.get_hessian(adjoint=True)
                for i in range(self.mesh.num_vertices()):
                    H.dat.data[i][:,:] *= self.p1indicator.dat.data[i]  # TODO: use pyop2
                indicator = self.p1indicator.copy()
                self.explicit_estimation_adjoint(square=False)
                self.p1indicator.interpolate(abs(self.p1indicator))
                H2 = self.get_hessian(adjoint=False)
                for i in range(self.mesh.num_vertices()):
                    H.dat.data[i][:,:] += H2.dat.data[i]*self.p1indicator.dat.data[i]  # TODO: use pyop2
                    H.dat.data[i][:,:] /= indicator.dat.data[i] + self.p1indicator.dat.data[i]
                self.M = steady_metric(None, mesh=self.mesh, H=H, op=self.op)
            elif self.approach == 'power_superposed':
                self.get_power_metric(adjoint=False)
                M = self.M.copy()
                self.get_power_metric(adjoint=True)
                #self.M = metric_intersection(self.M, M)
                self.M = metric_intersection(M, self.M)
            else:
                try:
                    assert hasattr(self, 'custom_adapt')
                    self.custom_adapt()
                except:
                    raise ValueError("Adaptivity mode {:s} not regcognised.".format(self.approach))

            # Apply metric relaxation, if requested
            self.M_unrelaxed = self.M.copy()
            if prev_metric is not None:
                self.M.project(metric_relaxation(self.M, project(prev_metric, self.P1_ten), relaxation_parameter))
            # (Default relaxation of 0.9 following [Power et al 2006])

        # FIXME!
        if hasattr(self, 'p0indicator'):
            self.estimator = sum(self.p0indicator.dat.data)

    def adapt_mesh(self, relaxation_parameter=0.9, prev_metric=None, custom_adapt=None):
        if not hasattr(self, 'M'):
            self.estimate_error(relaxation_parameter=relaxation_parameter, prev_metric=prev_metric, custom_adapt=custom_adapt)
        self.mesh = multi_adapt(self.M, op=self.op)
        self.plot()


class MeshOptimisation():
    """
    Loop over all mesh optimisation steps in order to obtain a mesh which is optimal w.r.t. the
    given error estimator for the given PDE problem.
    """
    def __init__(self, problem, op, mesh=None):
        self.problem = problem
        self.mesh = mesh
        assert op is not None
        self.op = op
        self.di = create_directory(op.di)

        # Default tolerances etc
        self.msg = "Mesh %2d: %7d cells, objective %.4e"
        self.conv_msg = "Converged after %d iterations due to %s"
        self.startit = 0
        self.maxit = 35
        self.element_rtol = 0.001    # Following [Power et al 2006]
        self.objective_rtol = 0.001  # TODO: experiment with these tighter tolerances
        self.estimator_atol = 1e-8

        # Logging
        self.logmsg = ''
        self.log = True

        # Data storage
        self.dat = {'elements': [],
                    'vertices': [],
                    'objective': [],
                    'estimator': [],
                    'effectivity': [],
                    'approach': self.op.approach}

    def iterate(self):
        M_ = None
        M = None

        # Create a log file and spit out parameters
        if self.log:
            self.logfile = open('{:s}/optimisation_log'.format(self.di), 'a+')
            self.logfile.write('\n{:s}{:s}\n\n'.format(date, self.logmsg))
            self.logfile.write('stabilisation: {:s}\n'.format(self.op.stabilisation))
            self.logfile.write('dwr_approach: {:s}\n'.format(self.op.dwr_approach))
            self.logfile.write('high_order: {:b}\n'.format(self.op.order_increase))
            self.logfile.write('relax: {:b}\n'.format(self.op.relax))
            self.logfile.write('maxit: {:d}\n'.format(self.maxit))
            self.logfile.write('element_rtol: {:f}\n'.format(self.element_rtol))
            self.logfile.write('objective_rtol: {:f}\n'.format(self.objective_rtol))
            self.logfile.write('estimator_atol: {:f}\n\n'.format(self.estimator_atol))
            # TODO: parallelise using Thetis format

        prev_sol = None
        tstart = clock()
        for i in range(self.startit, self.maxit):
            j = i - self.startit
            PETSc.Sys.Print('Solving on mesh %d' % i)
            tp = self.problem(mesh=self.mesh if j == 0 else tp.mesh,
                              op=self.op,
                              prev_solution=prev_sol)

            # Solve
            tp.solve()
            self.solution = tp.solution

            # Extract data
            self.dat['elements'].append(tp.mesh.num_cells())
            self.dat['vertices'].append(tp.mesh.num_vertices())
            self.dat['objective'].append(tp.objective_functional())
            PETSc.Sys.Print(self.msg % (i, self.dat['elements'][i], self.dat['objective'][i]))
            if self.log:  # TODO: parallelise
                self.logfile.write('Mesh  {:2d}: elements = {:10d}\n'.format(i, self.dat['elements'][j]))
                self.logfile.write('Mesh  {:2d}: vertices = {:10d}\n'.format(i, self.dat['vertices'][j]))
                self.logfile.write('Mesh  {:2d}:        J = {:.4e}\n'.format(i, self.dat['objective'][j]))

            # Solve adjoint
            if not self.op.approach in ('fixed_mesh', 'uniform', 'hessian', 'explicit', 'vorticity'):
                tp.solve_adjoint()  # TODO: This is not always necessary

            ## Estimate and record error  # FIXME
            #tp.estimate_error()
            #self.dat['estimator'].append(tp.estimator)
            #PETSc.Sys.Print('error estimator : %.4e' % tp.estimator)
            #if self.log:  # TODO: parallelise
            #    self.logfile.write('Mesh  {:2d}: estimator = {:.4e}\n'.format(i, tp.estimator))

            # Stopping criteria
            if i > self.startit:
                out = None
                obj_diff = abs(self.dat['objective'][j] - self.dat['objective'][j-1])
                el_diff = abs(self.dat['elements'][j] - self.dat['elements'][j-1])
                if obj_diff < self.objective_rtol*self.dat['objective'][j-1]:
                    out = self.conv_msg % (i+1, 'convergence in objective functional.')
                #elif self.dat['estimator'][j] < self.estimator_atol:  # FIXME
                #    out = self.conv_msg % (i+1, 'convergence in error estimator.')
                elif el_diff < self.element_rtol*self.dat['elements'][j-1]:
                    out = self.conv_msg % (i+1, 'convergence in mesh element count.')
                elif i >= self.maxit-1:
                    out = self.conv_msg % (i+1, 'maximum mesh adaptation count reached.')
                if out is not None:
                    PETSc.Sys.Print(out)
                    if self.log:
                        self.logfile.write(out+'\n')
                        tp.plot()
                    break

            # Adapt mesh
            #tp.set_target_vertices(num_vertices=self.dat['vertices'][0])  # FIXME
            tp.adapt_mesh(prev_metric=M_)
            tp.plot()
            if tp.nonlinear:
                prev_sol = tp.solution
            if self.op.relax:
                M_ = tp.M_unrelaxed
        self.dat['time'] = clock() - tstart
        PETSc.Sys.Print('Time to solution: %.1fs' % (self.dat['time']))
        if self.log:
            self.logfile.close()


class OuterLoop():
    def __init__(self, problem, op, mesh=None):
        self.problem = problem
        self.op = op
        self.mesh = op.default_mesh if mesh is None else mesh
        self.di = create_directory(self.op.di)

        # Default tolerances etc
        self.msg = "{:s} {:.2e} elements {:7d} iter {:2d} time {:6.1f} objective {:.4e} estimator {:.4e} ei {}\n"
        self.maxit = 35
        self.element_rtol = 0.005    # Following [Power et al 2006]
        self.objective_rtol = 0.005
        self.outer_startit = 0
        self.outer_maxit = 4
        #self.log = False
        self.log = True
        self.base = 10
        self.start_error = 1

    def desired_error_loop(self):
        mode = 'desired_error'

        # Create log file
        logfile = open(self.di + 'desired_error_test.log', 'a+')
        logfile.write('\n' + date + '\n\n')
        logfile.write('maxit: {:d}\n'.format(self.maxit))
        logfile.write('element_rtol: {:.4f}\n'.format(self.element_rtol))
        logfile.write('objective_rtol: {:.4f}\n'.format(self.objective_rtol))
        logfile.write('outer_maxit: {:d}\n\n'.format(self.outer_maxit))

        for i in range(self.outer_startit, self.outer_maxit):

            # Iterate over increasing target vertex counts
            PETSc.Sys.Print("\nOuter loop %d for approach '%s'" % (i+1, self.op.approach))
            self.op.target = self.start_target*pow(self.base, i)
            opt = MeshOptimisation(self.problem, mesh=self.mesh, op=self.op)
            opt.maxit = self.maxit
            opt.element_rtol = self.element_rtol
            opt.objective_rtol = self.objective_rtol
            opt.iterate()
            self.final_mesh = opt.mesh
            self.solution = opt.solution
            self.final_J = opt.dat['objective'][-1]

            # Logging
            logfile.write(self.msg.format(mode,
                                          1/self.op.target,
                                          opt.dat['elements'][-1],
                                          len(opt.dat['objective']),
                                          opt.dat['time'],
                                          opt.dat['objective'][-1],
                                          opt.dat['estimator'][-1],
                                          opt.dat['effectivity']))
            PETSc.Sys.Print("%d %.4e %.4e %a" % (opt.dat['elements'][-1],
                                                 opt.dat['objective'][-1],
                                                 opt.dat['estimator'][-1],
                                                 opt.dat['effectivity'][-1]))

            # Convergence criterion: relative tolerance for objective functional
            if i > self.outer_startit:
                obj_diff = abs(opt.dat['objective'][-1] - J_)
                if obj_diff < self.objective_rtol*J_:
                    PETSc.Sys.Print(opt.conv_msg % (i+1, 'convergence in objective functional.'))
                    break
            J_ = opt.dat['objective'][-1]
        logfile.close()


class UnsteadyProblem():
    def __init__(self, mesh, op, finite_element, discrete_adjoint=False):
        self.finite_element = finite_element
        self.discrete_adjoint = discrete_adjoint
        self.op = op
        self.mesh = op.default_mesh if mesh is None else mesh
        self.stab = op.stabilisation
        self.approach = op.approach

        # function spaces and mesh quantities
        self.V = FunctionSpace(self.mesh, self.finite_element)
        self.P0 = FunctionSpace(self.mesh, "DG", 0)
        self.P1 = FunctionSpace(self.mesh, "CG", 1)
        self.P1DG = FunctionSpace(self.mesh, "DG", 1)
        self.P1_vec = VectorFunctionSpace(self.mesh, "CG", 1)
        self.P1_ten = TensorFunctionSpace(self.mesh, "CG", 1)
        self.test = TestFunction(self.V)
        self.trial = TrialFunction(self.V)
        self.n = FacetNormal(self.mesh)
        self.h = CellSize(self.mesh)

        # prognostic fields
        self.solution = Function(self.V)
        self.adjoint_solution = Function(self.V)

        # outputs
        self.di = create_directory(self.op.di)
        self.solution_file = File(self.di + 'solution.pvd')
        self.adjoint_solution_file = File(self.di + 'adjoint_solution.pvd')
        self.indicator_file = File(self.di + 'indicator.pvd')

        # Adaptivity
        self.step_end = op.end_time if self.approach == 'fixed_mesh' else op.dt*op.dt_per_remesh

    def set_target_vertices(self, rescaling=0.85, num_vertices=None):
        """
        Set target number of vertices for adapted mesh by scaling the current number of vertices.
        """
        if num_vertices is None:
            num_vertices = self.mesh.num_vertices()
        self.op.target = num_vertices * rescaling

    def solve_step(self):
        """
        Solve forward PDE on a particular mesh.
        """
        pass

    def solve(self):
        """
        Solve PDE using mesh adaptivity.
        """
        self.remesh_step = 0
        while self.step_end <= self.op.end_time:
            if self.approach != 'fixed_mesh':
                if not self.approach in ('uniform', 'hessian', 'explicit', 'vorticity'):
                    self.get_adjoint_state()
                    self.interpolate_adjoint_solution()
                self.adapt_mesh()
                if self.remesh_step != 0:
                    self.interpolate_solution()
                else:
                    self.solution = self.op.set_initial_condition(self.V)
                #    if not self.approach in ('uniform', 'hessian', 'explicit'):
                #        self.interpolate_adjoint_solution()
                #    self.adapt_mesh()  # adapt again for the first iteration
                #    self.solution = self.op.set_initial_condition(self.V)
            self.solve_step()
            self.step_end += self.op.dt*self.op.dt_per_remesh
            self.remesh_step += 1

    def get_objective_kernel(self):
        """
        Derivative `g` of functional of interest `J`. i.e. For solution `u` we have
            J(u) = g . u
        """
        pass

    def objective_functional(self):  # TODO: account for time integral forms
        """
        Functional of interest which takes the PDE solution as input.
        """
        if not hasattr(self, 'kernel'):
            self.get_objective_kernel()
        return assemble(inner(self.solution, self.kernel)*dx)

    def solve_continuous_adjoint(self):  # NOTE: this should save to HDF5
        """
        Solve the adjoint PDE using a hand-coded continuous adjoint.
        """
        pass

    def solve_discrete_adjoint(self):
        """
        Solve the adjoint PDE in the discrete sense, using pyadjoint.
        """
        create_directory('outputs/hdf5/')
        J = self.objective_functional()
        compute_gradient(J, Control(self.gradient_field))
        tape = get_working_tape()
        solve_blocks = [block for block in tape._blocks if isinstance(block, SolveBlock)
                                                        and not isinstance(block, ProjectBlock)
                                                        and block.adj_sol is not None]

        N = len(solve_blocks)
        try:
            assert N == int(self.end_time/self.dt)
        except:
            ValueError("Expected one SolveBlock, but encountered {:d}".format(N))
        for i in range(0, N, self.op.dt_per_remesh):
            self.adjoint_solution.assign(solve_blocks[i].adj_sol)  # TODO: annotate in pyadjoint to track progress
            with DumbCheckpoint('outputs/hdf5/Adjoint2d_' + index_string(i), mode=FILE_CREATE) as sa:
                sa.store(self.adjoint_solution)
                sa.close()
            PETSc.Sys.Print("Storing adjoint solution %d" % i)
            self.adjoint_solution_file.write(self.adjoint_solution, t=self.op.dt*i)
        tape.clear_tape()

    def solve_adjoint(self):
        """
        Solve adjoint problem using specified method.
        """
        PETSc.Sys.Print("Solving adjoint problem...")
        if self.discrete_adjoint:
            self.solve_discrete_adjoint()
        else:
            self.solve_continuous_adjoint()

    def get_adjoint_state(self):
        """
        Get adjoint solution at timestep i.
        """
        i = self.remesh_step * self.op.dt_per_remesh
        with DumbCheckpoint('outputs/hdf5/Adjoint2d_' + index_string(i), mode=FILE_READ) as la:
            la.load(self.adjoint_solution)
            la.close()
        self.adjoint_solution_file.write(self.adjoint_solution, t=self.op.dt*i)

    def dwp_indication(self):
        """
        Indicate significance by the product of forward and adjoint solutions. This approach was
        used for mesh adaptive tsunami modelling in [Davis and LeVeque, 2016]. Here 'DWP' is used
        to stand for Dual Weighted Primal.
        """
        self.indicator = Function(self.P1)
        self.indicator.project(inner(self.solution, self.interpolated_adjoint_solution))
        self.indicator.rename('dwp')

    def explicit_estimation(self, space=None, square=True):  # TODO: change notation to indication
        pass

    def explicit_estimation_adjoint(self, space=None, square=True):
        pass

    def plot(self):
        """
        Plot current mesh and indicator field, if available.
        """
        if hasattr(self, 'indicator'):
            self.indicator.rename(self.approach + ' indicator')
            self.indicator_file.write(self.indicator, t=self.remesh_step*self.op.dt)

    def dwr_indication(self):
        """
        Indicate errors in the objective functional by the Dual Weighted Residual method. This is
        inherently problem-dependent.

        The resulting P0 field should be stored as `self.indicator`.
        """
        pass

    def dwr_indication_adjoint(self):
        pass

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

    def get_anisotropic_metric(self, adjoint=False, relax=False):
        """
        Apply the approach of [Loseille, Dervieux, Alauzet, 2009] to extract an anisotropic mesh 
        from the Dual Weighted Residual method.
        """
        pass

    def adapt_mesh(self, relaxation_parameter=Constant(0.9), prev_metric=None):
        """
        Adapt mesh according to error estimation strategy of choice.
        """
        with pyadjoint.stop_annotating():
            if self.approach == 'fixed_mesh':
                return
            elif self.approach == 'uniform':
                self.mesh = MeshHierarchy(self.mesh, 1)[1]
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
            elif self.approach == 'explicit':
                self.explicit_estimation()
                self.get_isotropic_metric()
            elif self.approach == 'explicit_adjoint':
                self.explicit_estimation_adjoint()
                self.get_isotropic_metric()
            elif self.approach == 'explicit_relaxed':
                self.explicit_estimation()
                self.get_isotropic_metric()
                M = self.M.copy()
                self.explicit_estimation_adjoint()
                self.get_isotropic_metric()
                self.M = metric_relaxation(M, self.M)
            elif self.approach == 'explicit_superposed':
                self.explicit_estimation()
                self.get_isotropic_metric()
                M = self.M.copy()
                self.explicit_estimation_adjoint()
                self.get_isotropic_metric()
                self.M = metric_intersection(M, self.M)
            elif self.approach == 'dwp':
                self.dwp_indication()
                self.get_isotropic_metric()
            elif self.approach == 'dwr':
                self.dwr_indication()
                self.get_isotropic_metric()
            elif self.approach == 'dwr_adjoint':
                self.dwr_indication_adjoint()
                self.get_isotropic_metric()
            elif self.approach == 'dwr_both':
                self.dwr_indication()
                self.get_isotropic_metric()
                i = self.indicator.copy()
                self.dwr_indication_adjoint()
                self.indicator.interpolate(Constant(0.5)*(i+self.indicator))
                self.get_isotropic_metric()
            elif self.approach == 'dwr_averaged':
                self.dwr_indication()
                self.get_isotropic_metric()
                i = self.indicator.copy()
                self.dwr_indication_adjoint()
                self.indicator.interpolate(Constant(0.5)*(abs(i)+abs(self.indicator)))
                self.get_isotropic_metric()
            elif self.approach == 'dwr_relaxed':
                self.dwr_indication()
                self.get_isotropic_metric()
                M = self.M.copy()
                self.dwr_indication_adjoint()
                self.get_isotropic_metric()
                self.M = metric_relaxation(M, self.M)
            elif self.approach == 'dwr_superposed':
                self.dwr_indication()
                self.get_isotropic_metric()
                M = self.M.copy()
                self.dwr_indication_adjoint()
                self.get_isotropic_metric()
                self.M = metric_intersection(M, self.M)
            elif self.approach == 'loseille':
                self.get_anisotropic_metric(adjoint=False)
            elif self.approach == 'loseille_adjoint':
                self.get_anisotropic_metric(adjoint=True)
            elif self.approach == 'loseille_relaxed':
                self.get_anisotropic_metric(adjoint=False)
                M = self.M.copy()
                self.get_anisotropic_metric(adjoint=True)
                self.M = metric_relaxation(M, self.M)
            elif self.approach == 'loseille_superposed':
                self.get_anisotropic_metric(adjoint=False)
                M = self.M.copy()
                self.get_anisotropic_metric(adjoint=True)
                self.M = metric_intersection(M, self.M)
            elif self.approach == 'power':
                self.explicit_estimation_adjoint(square=False)
                H = self.get_hessian(adjoint=False)
                for i in range(len(self.indicator.dat.data)):
                    H.dat.data[i][:,:] *= np.abs(self.indicator.dat.data[i])  # TODO: use pyop2
                self.M = steady_metric(self.adjoint_solution, H=H, op=self.op)
            elif self.approach == 'power_adjoint':
                self.explicit_estimation(square=False)
                H = self.get_hessian(adjoint=True)
                for i in range(len(self.indicator.dat.data)):
                    H.dat.data[i][:,:] *= np.abs(self.indicator.dat.data[i])  # TODO: use pyop2
                self.M = steady_metric(self.solution, H=H, op=self.op)
            elif self.approach == 'power_relaxed':
                self.explicit_estimation(square=False)
                H = self.get_hessian(adjoint=True)
                for i in range(len(self.indicator.dat.data)):
                    H.dat.data[i][:,:] *= np.abs(self.indicator.dat.data[i])  # TODO: use pyop2
                indicator = self.indicator.copy()
                self.explicit_estimation_adjoint(square=False)
                H2 = self.get_hessian(adjoint=False)
                for i in range(len(self.indicator.dat.data)):
                    H.dat.data[i][:,:] += H2.dat.data[i]*np.abs(self.indicator.dat.data[i])  # TODO: use pyop2
                    H.dat.data[i][:,:] /= np.abs(indicator.dat.data[i]) + np.abs(self.indicator.dat.data[i])
                self.M = steady_metric(self.solution+self.adjoint_solution, mesh=self.mesh, H=H, op=self.op)
            elif self.approach == 'power_superposed':
                self.explicit_estimation(square=False)
                H = self.get_hessian(adjoint=True)
                for i in range(len(self.indicator.dat.data)):                 # TODO: use pyop2
                    H.dat.data[i][:,:] *= np.abs(self.indicator.dat.data[i])
                M = steady_metric(self.solution, H=H, op=self.op)
                self.explicit_estimation_adjoint(square=False)
                H = self.get_hessian(adjoint=False)
                for i in range(len(self.indicator.dat.data)):
                    H.dat.data[i][:,:] *= np.abs(self.indicator.dat.data[i])
                self.M = steady_metric(self.adjoint_solution, H=H, op=self.op)
                self.M = metric_intersection(M, self.M)
            else:
                try:
                    assert hasattr(self, 'custom_adapt')
                    self.custom_adapt()
                except:
                    raise ValueError("Adaptivity mode {:s} not regcognised.".format(self.approach))

            # Apply metric relaxation, if requested
            if prev_metric is not None:
                if self.M is None:
                    self.M.project(project(prev_metric, self.P1_ten))
                else:
                    self.M_unrelaxed = self.M.copy()
                    self.M.project(metric_relaxation(project(prev_metric, self.P1_ten), self.M, relaxation_parameter))
            # (Default relaxation of 0.9 following [Power et al 2006])

            # Adapt mesh
            if self.M is not None and norm(self.M) > 0.1*norm(Constant(1, domain=self.mesh)):
                # FIXME: The 0.1 factor seems pretty arbitrary
                #self.mesh = adapt(self.mesh, self.M)
                self.mesh = multi_adapt(self.M, op=self.op)

                # Re-establish function spaces
                self.V = FunctionSpace(self.mesh, self.finite_element)
                self.P0 = FunctionSpace(self.mesh, "DG", 0)
                self.P1 = FunctionSpace(self.mesh, "CG", 1)
                self.P1DG = FunctionSpace(self.mesh, "DG", 1)
                self.P1_vec = VectorFunctionSpace(self.mesh, "CG", 1)
                self.P1_ten = TensorFunctionSpace(self.mesh, "CG", 1)
                self.test = TestFunction(self.V)
                self.trial = TrialFunction(self.V)
                self.n = FacetNormal(self.mesh)
                self.h = CellSize(self.mesh)

            # Plot results
            self.plot()

    def interpolate_solution(self):
        """
        Interpolate solution onto the new mesh after a mesh adaptation.
        """
        with pyadjoint.stop_annotating():
            interpolated_solution = Function(FunctionSpace(self.mesh, self.V.ufl_element()))
            interpolated_solution.project(self.solution)
            name = self.solution.dat.name
            self.solution = interpolated_solution
            self.solution.rename(name)

    def interpolate_adjoint_solution(self):
        """
        Interpolate adjoint solution onto the new mesh after a mesh adaptation.
        """
        with pyadjoint.stop_annotating():
            self.interpolated_adjoint_solution = Function(FunctionSpace(self.mesh, self.V.ufl_element()))
            self.interpolated_adjoint_solution.project(self.adjoint_solution)
