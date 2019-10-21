from firedrake import *

import numpy as np
import numpy
from numpy import linalg as la

from adapt_utils.adapt.adaptation import multi_adapt
from adapt_utils.options import DefaultOptions


__all__ = ["AnisotropicMetricDriver"]


class AnisotropicMetricDriver():
    """
    Driver for anisotropic mesh adaptation using an approach inspired by [Carpio et al. 2013].
    """
    def __init__(self, mesh, hessian=None, indicator=None, op=DefaultOptions()):
        self.mesh = mesh
        try:
            assert self.mesh.topological_dimension() == 2
        except:
            raise NotImplementedError
        self.H = hessian
        self.eta = indicator
        self.op = op

        # Spaces
        self.P0 = FunctionSpace(mesh, "DG", 0)
        self.P0_vec = VectorFunctionSpace(mesh, "DG", 0)
        self.P0_ten = TensorFunctionSpace(mesh, "DG", 0)
        self.P1_ten = TensorFunctionSpace(mesh, "CG", 1)
        self.p0test = TestFunction(self.P0)
        if hessian is not None:
            self.p0hessian = project(hessian, self.P0_ten)

        # Fields related to mesh
        self.h = CellSize(mesh)    # current element size
        self.J = Jacobian(mesh)
        self.ne = mesh.num_cells()
        self.h_opt = Function(self.P0)  # optimal element size

        # Eigenvalues and eigenvectors
        self.eval0 = Function(self.P0)
        self.eval1 = Function(self.P0)
        self.evec0 = Function(self.P0_vec)
        self.evec1 = Function(self.P0_vec)

        # Metrics and error estimators
        self.p0metric = Function(self.P0_ten)
        self.p1metric = Function(self.P1_ten)
        self.estimator = Function(self.P0)

    def get_reference_element_size(self):
        h_ref = Function(self.P0)
        h_ref.interpolate(self.h/abs(det(self.J))/sqrt(0.5*self.ne))
        #h_ref.interpolate(self.h/abs(det(self.J))))
        #assert np.var(h_ref.dat.data) < 1e-10  # FIXME
        self.h_ref = h_ref.dat.data[0]

    def get_eigenpair(self):
        JJt = Function(self.P0_ten)
        JJt.interpolate(self.J*transpose(self.J))

        for i in range(self.ne):
            lam, v = la.eigh(JJt.dat.data[i])
            self.eval0.dat.data[i] = lam[0]
            self.eval1.dat.data[i] = lam[1]
            self.evec0.dat.data[i][:] = v[0]
            self.evec1.dat.data[i][:] = v[1]

    def get_hessian_eigenpair(self):
        assert self.H is not None
        H_avg = Function(self.P0_ten)
        H_avg.interpolate(self.H)

        for i in range(self.ne):
            lam, v = la.eigh(H_avg.dat.data[i])
            if np.abs(lam[0]) > np.abs(lam[1]):
                self.eval0.dat.data[i] = lam[0]
                self.eval1.dat.data[i] = lam[1]
                self.evec0.dat.data[i][:] = v[0]
                self.evec1.dat.data[i][:] = v[1]
            else:
                self.eval0.dat.data[i] = lam[1]
                self.eval1.dat.data[i] = lam[0]
                self.evec0.dat.data[i][:] = v[1]
                self.evec1.dat.data[i][:] = v[0]

    def get_optimal_element_size(self):
        assert self.eta is not None
        alpha = self.op.convergence_rate
        self.h_opt.interpolate(max_value(self.eta**(1/(alpha+1)), self.op.min_norm))
        Sum = np.sum(self.h_opt.dat.data)
        scaling = (pow(self.op.target, -self.op.convergence_rate)/Sum)**(1/alpha)
        self.h_opt.interpolate(self.h*scaling*self.h_opt**(-1))

    def get_optimised_eigenpair(self):
        s = Function(self.P0)
        s.interpolate(sqrt(abs(self.eval0/self.eval1)))
        tmp = Function(self.P0_vec)
        tmp.assign(self.evec0)
        self.evec0.assign(self.evec1)
        self.evec1.assign(tmp)
        self.eval0.interpolate(self.h_opt/self.h_ref*s)
        self.eval1.interpolate(self.h_opt/self.h_ref/s)

    def build_metric(self):
        """
        NOTE: Assumes eigevalues are already squared.
        """
        for i in range(self.ne):
            lam0 = 1/self.eval0.dat.data[i]
            lam1 = 1/self.eval1.dat.data[i]
            v0 = self.evec0.dat.data[i]
            v1 = self.evec1.dat.data[i]
            self.p0metric.dat.data[i][0, 0] = lam0*v0[0]*v0[0] + lam1*v1[0]*v1[0]
            self.p0metric.dat.data[i][0, 1] = lam0*v0[0]*v0[1] + lam1*v1[0]*v1[1]
            self.p0metric.dat.data[i][1, 0] = self.p0metric.dat.data[i][0, 1]
            self.p0metric.dat.data[i][1, 1] = lam0*v0[1]*v0[1] + lam1*v1[1]*v1[1]

    def project_metric(self):
        self.p1metric.project(self.p0metric)

    def get_identity_metric(self):
        self.get_eigenpair()
        self.build_metric()
        self.project_metric()

    def get_anisotropic_metric(self):
        self.get_hessian_eigenpair()
        self.get_reference_element_size()
        self.get_optimal_element_size()
        self.get_optimised_eigenpair()
        self.build_metric()
        self.project_metric()

    def adapt_mesh(self):
        self.mesh = multi_adapt(self.p1metric, op=self.op)

    def Lij(self, i, j):
        eigenvectors = [self.evec0, self.evec1]
        triple_product = dot(eigenvectors[i], dot(self.p0hessian, eigenvectors[j]))
        return assemble(self.p0test*triple_product*triple_product*dx)

    def cell_interpolation_error(self):
        l = self.eval0*self.eval0*self.Lij(0, 0)
        l += self.eval0*self.eval1*self.Lij(0, 1)
        l += self.eval1*self.eval0*self.Lij(1, 0)
        l += self.eval1*self.eval1*self.Lij(1, 1)
        self.estimator.interpolate(sqrt(l))

    def gradient_interpolation_error(self):
        self.cell_interpolation_error()
        coeff = pow(min_value(self.eval0, self.eval1), -0.5)
        self.estimator.interpolate(coeff*self.estimator)

    def edge_interpolation_error(self):
        self.cell_interpolation_error()
        coeff = sqrt((self.eval0+self.eval1)*pow(min_value(self.eval0, self.eval1), 1.5))
        self.estimator.interpolate(coeff*self.estimator)
