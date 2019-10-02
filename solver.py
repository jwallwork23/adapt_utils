from firedrake import *
from firedrake.petsc import PETSc
from thetis import create_directory

import os
import datetime
from time import clock
import numpy as np
import pickle

from adapt_utils.options import DefaultOptions
from adapt_utils.misc.misc import index_string
from adapt_utils.adapt.adaptation import *
from adapt_utils.adapt.metric import *


__all__ = ["SteadyProblem", "UnsteadyProblem", "MeshOptimisation", "OuterLoop"]


now = datetime.datetime.now()
date = str(now.day) + '-' + str(now.month) + '-' + str(now.year % 2000)


class SteadyProblem():
    """
    Base class for solving steady-state PDE problems using mesh adaptivity.

    There are three main functionalities:
        * solve PDE;
        * solve adjoint PDE;
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
        self.solution_file = File(self.di + 'solution.pvd')
        self.adjoint_solution_file = File(self.di + 'adjoint_solution.pvd')

        self.estimators = {}
        self.indicators = {}

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

    def solve_adjoint(self):
        """
        Solve adjoint problem using specified method.
        """
        PETSc.Sys.Print("Solving adjoint problem...")
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

    def dwp_estimation(self):
        self.estimator = assemble(inner(self.solution, self.adjoint_solution)*dx)

    def explicit_indication(self, space=None, square=True):
        pass

    def explicit_indication_adjoint(self, space=None, square=True):
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
        Indicate errors in the quantity of interest by the Dual Weighted Residual method. This is
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

    def get_loseille_metric(self, adjoint=False, relax=False):
        """
        Construct an anisotropic metric using an approach inspired by [Loseille et al. 2009].
        """
        pass

    def get_power_metric(self, adjoint=False):
        """
        Construct an anisotropic metric using an approach inspired by [Power et al. 2006].

        If `adjoint` mode is turned off, we weight the Hessian of the adjoint solution with a residual
        for the forward PDE. Otherwise, we weight the Hessian of the forward solution with a residual
        for the adjoint PDE.
        """
        if adjoint:
            self.explicit_indication_adjoint(square=False)
            self.p1indicator.interpolate(abs(self.p1cell_res_adjoint))
        else:
            self.explicit_indication(square=False)
            self.p1indicator.interpolate(abs(self.p1cell_res))
        H = self.get_hessian(adjoint=not adjoint)
        for i in range(self.mesh.num_vertices()):
            H.dat.data[i][:,:] *= self.p1indicator.dat.data[i]  # TODO: use pyop2
        if adjoint:
            self.M = steady_metric(self.solution, H=H, op=self.op)
        else:
            self.M = steady_metric(self.adjoint_solution, H=H, op=self.op)

    def indicate_error(self, relaxation_parameter=0.9, prev_metric=None, estimate_error=True):
        """
        Evaluate error estimation strategy of choice in order to obtain a metric field for mesh
        adaptation.

        NOTE: User-provided metrics may be applied by defining a `custom_adapt` method.

        :kwarg relaxation_parameter: Scalar in the range [0, 1] used to take a weighted average
        between metrics at the current and previous steps.
        :kwarg prev_metric: Metric from previous step. If unprovided, metric relaxation cannot be applied.
        :kwarg estimate_error: Toggle computation of global error estimate.
        """
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
            self.explicit_indication()
            self.get_isotropic_metric()
        elif self.approach == 'explicit_adjoint':
            self.explicit_indication_adjoint()
            self.get_isotropic_metric()
        elif self.approach == 'explicit_relaxed':
            self.explicit_indication()
            self.get_isotropic_metric()
            M = self.M.copy()
            self.explicit_indication_adjoint()
            self.get_isotropic_metric()
            self.M = metric_relaxation(M, self.M)
        elif self.approach == 'explicit_superposed':
            self.explicit_indication()
            self.get_isotropic_metric()
            M = self.M.copy()
            self.explicit_indication_adjoint()
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
            #self.M = metric_intersection(self.M, M)
            self.M = metric_intersection(M, self.M)
        else:
            try:
                assert hasattr(self, 'custom_adapt')
                self.custom_adapt()
            except:
                raise ValueError("Adaptivity mode {:s} not regcognised.".format(self.approach))

        # Apply metric relaxation, if requested
        assert relaxation_parameter >= 0
        assert relaxation_parameter <= 1
        self.M_unrelaxed = self.M.copy()
        if prev_metric is not None:
            self.M.project(metric_relaxation(self.M, project(prev_metric, self.P1_ten), relaxation_parameter))

        ## FIXME
        #if hasattr(self, 'p0indicator'):
        #    self.estimator = sum(self.p0indicator.dat.data)
        if estimate_error:
            if self.approach in ('dwr', 'power', 'loseille'):
                self.dwr_estimation()
            elif self.approach in ('dwr_adjoint', 'power_adjoint', 'loseille_adjoint'):
                self.dwr_estimation_adjoint()
            elif self.approach in ('dwr_relaxed', 'dwr_superposed', 'power_relaxed', 'power_superposed', 'loseille_relaxed', 'loseille_superposed'):
                self.estimator = 0.5*(self.dwr_estimation() + self.dwr_estimation_adjoint())
            elif self.approach == 'dwp':
                self.dwp_estimation()
            else:
                raise NotImplementedError  # TODO

    def adapt_mesh(self, relaxation_parameter=0.9, prev_metric=None, estimate_error=True):
        """
        Adapt mesh using metric constructed in error estimation step.

        NOTE: User-provided metrics may be applied by defining a `custom_adapt` method.

        :kwarg relaxation_parameter: Scalar in the range [0, 1] used to take a weighted average
        between metrics at the current and previous steps.
        :kwarg prev_metric: Metric from previous step. If unprovided, metric relaxation cannot be applied.
        :kwarg estimate_error: Toggle computation of global error estimate.
        """
        if not hasattr(self, 'M'):
            self.indicate_error(relaxation_parameter=relaxation_parameter, prev_metric=prev_metric, estimate_error=estimate_error)
        #self.mesh = multi_adapt(self.M, op=self.op)
        self.mesh = adapt(self.mesh, self.M)
        print('Done adapting.')
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
        self.msg = "Mesh %2d: %7d cells, qoi %.4e, estimator %.4e"
        self.conv_msg = "Converged after %d iterations due to %s"
        self.startit = 0
        self.minit = 1
        self.maxit = 35
        self.element_rtol = 0.005    # Following [Power et al 2006]
        self.qoi_rtol = 0.005
        self.estimator_atol = 1e-8

        # Logging
        self.logmsg = ''
        self.log = True

        # Data storage
        self.dat = {'elements': [],
                    'vertices': [],
                    'qoi': [],
                    'estimator': [],
                    'approach': self.op.approach}

    def iterate(self):
        assert self.minit >= self.startit
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
            self.logfile.write('qoi_rtol: {:f}\n'.format(self.qoi_rtol))
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
            self.dat['qoi'].append(tp.quantity_of_interest())
            if self.log:  # TODO: parallelise
                self.logfile.write('Mesh  {:2d}: elements = {:10d}\n'.format(i, self.dat['elements'][j]))
                self.logfile.write('Mesh  {:2d}: vertices = {:10d}\n'.format(i, self.dat['vertices'][j]))
                self.logfile.write('Mesh  {:2d}:        J = {:.4e}\n'.format(i, self.dat['qoi'][j]))

            # Solve adjoint
            if not self.op.approach in ('fixed_mesh', 'uniform', 'hessian', 'explicit', 'vorticity'):
                tp.solve_adjoint()

            # Estimate and record error  # FIXME
            tp.indicate_error()
            self.dat['estimator'].append(tp.estimator)
            PETSc.Sys.Print(self.msg % (i, self.dat['elements'][i], self.dat['qoi'][i], tp.estimator))
            if self.log:  # TODO: parallelise
                self.logfile.write('Mesh  {:2d}: estimator = {:.4e}\n'.format(i, tp.estimator))

            # Stopping criteria
            if i >= self.minit and i > self.startit:
                out = None
                obj_diff = abs(self.dat['qoi'][j] - self.dat['qoi'][j-1])
                el_diff = abs(self.dat['elements'][j] - self.dat['elements'][j-1])
                if obj_diff < self.qoi_rtol*self.dat['qoi'][j-1]:
                    out = self.conv_msg % (i+1, 'convergence in quantity of interest.')
                #elif self.dat['estimator'][j] < self.estimator_atol:  # FIXME
                #    out = self.conv_msg % (i+1, 'convergence in error estimator.')
                elif el_diff < self.element_rtol*self.dat['elements'][j-1] and i > self.startit+1:
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
            #tp.set_target_vertices(num_vertices=self.dat['vertices'][0])
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
    """
    Iterate over a range of tolerated errors for a given (steady) problem setup and mesh adaptation strategy,
    seeking convergence of the quantity of interest.

    :arg problem: Problem type to consider.
    :arg op: Parameter class.
    :arg mesh: Initial mesh.
    """
    def __init__(self, problem, op, mesh=None):
        self.problem = problem
        self.op = op
        self.mesh = op.default_mesh if mesh is None else mesh
        self.di = create_directory(self.op.di)

        # Default tolerances etc
        self.msg = "{:s} {:.2e} elements {:7d} iter {:2d} time {:6.1f} qoi {:.4e} estimator {:.4e}\n"
        self.maxit = 35
        self.element_rtol = 0.005    # Following [Power et al 2006]
        self.qoi_rtol = 0.005
        self.outer_startit = 0
        self.outer_maxit = 4
        #self.log = False
        self.log = True
        self.base = 10
        self.start_error = 1


    def desired_error_loop(self):
        mode = 'desired_error'
        dat = {'elements': [], 'qoi': [], 'time': [], 'estimator': []}

        # Create log file
        logfile = open(self.di + 'desired_error_test.log', 'a+')
        logfile.write('\n' + date + '\n\n')
        logfile.write('maxit: {:d}\n'.format(self.maxit))
        logfile.write('element_rtol: {:.4f}\n'.format(self.element_rtol))
        logfile.write('qoi_rtol: {:.4f}\n'.format(self.qoi_rtol))
        logfile.write('outer_maxit: {:d}\n\n'.format(self.outer_maxit))
        logfile.close()

        for i in range(self.outer_startit, self.outer_maxit):

            # Iterate over increasing target vertex counts
            PETSc.Sys.Print("\nOuter loop %d for approach '%s'" % (i+1, self.op.approach))
            self.op.target = self.start_target*pow(self.base, i)
            opt = MeshOptimisation(self.problem, mesh=self.mesh, op=self.op)
            opt.maxit = self.maxit
            opt.element_rtol = self.element_rtol
            opt.qoi_rtol = self.qoi_rtol
            opt.iterate()
            self.final_mesh = opt.mesh
            self.solution = opt.solution
            self.final_J = opt.dat['qoi'][-1]

            # Logging
            logfile = open(self.di + 'desired_error_test.log', 'a+')
            logfile.write(self.msg.format(mode,
                                          1/self.op.target,
                                          opt.dat['elements'][-1],
                                          len(opt.dat['qoi']),
                                          opt.dat['time'],
                                          opt.dat['qoi'][-1],
                                          opt.dat['estimator'][-1]))
            logfile.close()
            dat['elements'].append(opt.dat['elements'][-1])
            dat['qoi'].append(opt.dat['qoi'][-1])
            dat['time'].append(opt.dat['time'])
            dat['estimator'].append(opt.dat['estimator'][-1])
            PETSc.Sys.Print("Elements %d QoI %.4e Estimator %.4e" % (opt.dat['elements'][-1],
                                                                     opt.dat['qoi'][-1],
                                                                     opt.dat['estimator'][-1]))

            # Convergence criterion: relative tolerance for QoI
            if i > self.outer_startit:
                obj_diff = abs(opt.dat['qoi'][-1] - J_)
                if obj_diff < self.qoi_rtol*J_:
                    PETSc.Sys.Print(opt.conv_msg % (i+1, 'convergence in quantity of interest.'))
                    break
            J_ = opt.dat['qoi'][-1]
        picklefile = open(self.di + 'desired_error.pickle', 'wb')
        pickle.dump(dat, picklefile)
        picklefile.close()


class UnsteadyProblem():
    """
    Base class for solving time-dependent PDE problems using mesh adaptivity.

    There are three main functionalities:
        * solve PDE;
        * solve adjoint PDE;
        * adapt mesh based on some error estimator of choice.
    """
    def __init__(self, mesh, op, finite_element, discrete_adjoint=False):
        self.finite_element = finite_element
        self.discrete_adjoint = discrete_adjoint
        self.op = op
        self.mesh = op.default_mesh if mesh is None else mesh
        self.stab = op.stabilisation
        self.approach = op.approach

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
        self.solution_file = File(self.di + 'solution.pvd')
        self.adjoint_solution_file = File(self.di + 'adjoint_solution.pvd')
        self.indicator_file = File(self.di + 'indicator.pvd')

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
        elif adjoint:
            self.set_start_condition(adjoint)

        # Solve/adapt loop
        while self.step_end <= self.op.end_time:

            # Fixed mesh case
            if self.approach == 'fixed_mesh':
                self.solve_step(adjoint)
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
        if self.approach in ('uniform', 'hessian', 'explicit', 'vorticity'):
            return
        if not hasattr(self, 'V_orig'):
            self.V_orig = FunctionSpace(self.mesh, self.finite_element)
        op = self.op
        names = {'Tracer2d': 'tracer_2d', 'Velocity2d': 'uv_2d', 'Elevation2d': 'elev_2d'}
        i = self.remesh_step*int(self.op.dt_per_export/self.op.dt_per_remesh)

        # FIXME for continuous adjoint
        filename = 'Adjoint2d_{:5s}'.format(index_string(i))

        #filename = '{:s}_{:5s}'.format(variable, index_string(i))
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

    def explicit_indication(self, space=None, square=True):
        pass

    def explicit_indication_adjoint(self, space=None, square=True):
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
        Indicate errors in the quantity of interest by the Dual Weighted Residual method. This is
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

    def get_loseille_metric(self, adjoint=False, relax=False):
        """
        Construct an anisotropic metric using an approach inspired by [Loseille et al. 2009].
        """
        pass

    def get_power_metric(self, adjoint=False):
        """
        Construct an anisotropic metric using an approach inspired by [Power et al. 2006].

        If `adjoint` mode is turned off, we weight the Hessian of the adjoint solution with a residual
        for the forward PDE. Otherwise, we weight the Hessian of the forward solution with a residual
        for the adjoint PDE.
        """
        # TODO: update
        if adjoint:
            self.explicit_indication_adjoint(square=False)
            self.p1indicator.interpolate(abs(self.p1cell_res_adjoint))
        else:
            self.explicit_indication(square=False)
            self.p1indicator.interpolate(abs(self.p1cell_res))
        H = self.get_hessian(adjoint=not adjoint)
        for i in range(self.mesh.num_vertices()):
            H.dat.data[i][:,:] *= self.p1indicator.dat.data[i]  # TODO: use pyop2
        if adjoint:
            self.M = steady_metric(self.solution, H=H, op=self.op)
        else:
            self.M = steady_metric(self.adjoint_solution, H=H, op=self.op)


    def adapt_mesh(self, relaxation_parameter=0.9, prev_metric=None):
        """
        Adapt mesh according to error estimation strategy of choice.

        NOTE: User-provided metrics may be applied by defining a `custom_adapt` method.

        :kwarg relaxation_parameter: Scalar in the range [0, 1] used to take a weighted average
        between metrics at the current and previous steps.
        :kwarg prev_metric: Metric from previous step. If unprovided, metric relaxation cannot be applied.
        :kwarg estimate_error: Toggle computation of global error estimate.
        """
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
            self.explicit_indication()
            self.get_isotropic_metric()
        elif self.approach == 'explicit_adjoint':
            self.explicit_indication_adjoint()
            self.get_isotropic_metric()
        elif self.approach == 'explicit_relaxed':
            self.explicit_indication()
            self.get_isotropic_metric()
            M = self.M.copy()
            self.explicit_indication_adjoint()
            self.get_isotropic_metric()
            self.M = metric_relaxation(M, self.M)
        elif self.approach == 'explicit_superposed':
            self.explicit_indication()
            self.get_isotropic_metric()
            M = self.M.copy()
            self.explicit_indication_adjoint()
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
            #self.M = metric_intersection(self.M, M)
            self.M = metric_intersection(M, self.M)
        else:
            try:
                assert hasattr(self, 'custom_adapt')
                self.custom_adapt()
            except:
                raise ValueError("Adaptivity mode {:s} not regcognised.".format(self.approach))

        # Apply metric relaxation, if requested
        assert relaxation_parameter >= 0
        assert relaxation_parameter <= 1
        self.M_unrelaxed = self.M.copy()
        if prev_metric is not None:
            self.M.project(metric_relaxation(self.M, project(prev_metric, self.P1_ten), relaxation_parameter))

        # Adapt mesh
        if self.M is not None and norm(self.M) > 0.1*norm(Constant(1, domain=self.mesh)):
            # FIXME: The 0.1 factor seems pretty arbitrary
            self.mesh = adapt(self.mesh, self.M)
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
