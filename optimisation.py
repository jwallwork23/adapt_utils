from firedrake import *
from firedrake.petsc import PETSc
from thetis import create_directory

import os
import datetime
from time import clock
import numpy as np
import pickle

from adapt_utils.misc.misc import index_string
from adapt_utils.misc.conditioning import *
from adapt_utils.adapt.adaptation import *
from adapt_utils.adapt.metric import *
from adapt_utils.adapt.p0_metric import *
from adapt_utils.adapt.kernels import matscale_kernel, include_dir


__all__ = ["MeshOptimisation", "OuterLoop"]


now = datetime.datetime.now()
date = str(now.day) + '-' + str(now.month) + '-' + str(now.year % 2000)


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
        # self.msg = "Mesh %2d: %7d cells, qoi %.4e, estimator %.4e"
        self.msg = "Mesh %2d: %7d cells, qoi %.4e"
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

        # Nonlinear problems
        self.use_prev_sol = True

    def iterate(self):
        assert self.minit >= self.startit
        M_ = None
        M = None

        # Create a log file and spit out parameters
        if self.log:
            self.logfile = open(os.path.join(self.di, 'optimisation_log'), 'a+')
            self.logfile.write('\n{:s}{:s}\n\n'.format(date, self.logmsg))
            self.logfile.write('high_order: {:b}\n'.format(self.op.order_increase))
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
            if not self.op.approach in ('fixed_mesh', 'uniform', 'hessian', 'vorticity'):
                tp.solve_adjoint()

            # Estimate and record error  # FIXME
            tp.indicate_error()
            # self.dat['estimator'].append(tp.estimator)
            # PETSc.Sys.Print(self.msg % (i, self.dat['elements'][i], self.dat['qoi'][i], tp.estimator))
            PETSc.Sys.Print(self.msg % (i, self.dat['elements'][i], self.dat['qoi'][i]))
            # if self.log:  # TODO: parallelise
            #     self.logfile.write('Mesh  {:2d}: estimator = {:.4e}\n'.format(i, tp.estimator))

            # Stopping criteria
            if i >= self.minit and i > self.startit:
                out = None
                obj_diff = abs(self.dat['qoi'][j] - self.dat['qoi'][j-1])
                el_diff = abs(self.dat['elements'][j] - self.dat['elements'][j-1])
                if obj_diff < self.qoi_rtol*self.dat['qoi'][j-1]:
                    out = self.conv_msg % (i+1, 'convergence in quantity of interest.')
                # elif self.dat['estimator'][j] < self.estimator_atol:  # FIXME
                #     out = self.conv_msg % (i+1, 'convergence in error estimator.')
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
            # tp.op.target = self.dat['vertices'][0]*tp.op.rescaling
            tp.adapt_mesh()
            tp.plot()
            if tp.nonlinear and self.use_prev_sol:
                prev_sol = tp.solution

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
        # self.log = False
        self.log = True
        self.base = 10
        self.start_error = 1

    def desired_error_loop(self):
        mode = 'desired_error'
        dat = {'elements': [], 'qoi': [], 'time': [], 'estimator': []}

        # Create log file
        logfile = open(os.path.join(self.di, 'desired_error_test.log'), 'a+')
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
            logfile = open(os.path.join(self.di, 'desired_error_test.log'), 'a+')
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
        picklefile = open(os.path.join(self.di, 'desired_error.pickle'), 'wb')
        pickle.dump(dat, picklefile)
        picklefile.close()

