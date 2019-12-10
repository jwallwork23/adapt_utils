from firedrake import *

from adapt_utils.options import DefaultOptions
from adapt_utils.adapt.metric import isotropic_metric
from adapt_utils.adapt.kernels import *


__all__ = ["AnisotropicMetricDriver"]


class AnisotropicMetricDriver():
    """
    Driver for anisotropic mesh adaptation using an approach based on [Carpio et al. 2013].
    """
    def __init__(self, mesh, hessian=None, indicator=None, op=DefaultOptions()):
        self.mesh = mesh
        self.dim = self.mesh.topological_dimension()
        try:
            assert self.dim == 2
        except:
            raise NotImplementedError
        self.H = hessian
        self.eta = indicator
        self.op = op

        # Spaces
        self.P0 = FunctionSpace(mesh, "DG", 0)
        self.P1 = FunctionSpace(mesh, "CG", 1)
        self.P0_vec = VectorFunctionSpace(mesh, "DG", 0)
        self.P0_ten = TensorFunctionSpace(mesh, "DG", 0)
        self.P1_ten = TensorFunctionSpace(mesh, "CG", 1)
        self.p0test = TestFunction(self.P0)
        if hessian is not None:
            self.p0hessian = project(hessian, self.P0_ten)

        # Fields related to mesh
        self.J = Jacobian(mesh)
        self.detJ = JacobianDeterminant(mesh)
        self.K_hat = 0.5                # Area of reference element
        self.K = Function(self.P0)      # Current element size
        self.K_opt = Function(self.P0)  # Optimal element size

        # Eigenvalues and eigenvectors
        self.eval = Function(self.P0_vec)
        self.evec = Function(self.P0_ten)

        # Metrics and error estimators
        self.p0metric = Function(self.P0_ten)
        self.p1metric = Function(self.P1_ten)
        self.estimator = Function(self.P0)

    def get_eigenpair(self):
        """
        Extract eigenpairs related to elementwise metric associated with current mesh.
        """
        JJt = Function(self.P0_ten)
        JJt.interpolate(self.J*self.J.T)
        kernel = op2.Kernel(get_eigendecomposition_kernel(self.dim), "get_eigendecomposition", cpp=True, include_dirs=include_dir)
        op2.par_loop(kernel, self.P0_ten.node_set, self.evec.dat(op2.RW), self.eval.dat(op2.RW), JJt.dat(op2.READ))
        self.eval.interpolate(as_vector([1/self.eval[0], 1/self.eval[1]]))  # TODO: avoid interp?

    def get_hessian_eigenpair(self):  # TODO: Enforce max anisotropy
        """
        Extract eigenpairs related to provided Hessian.
        """
        assert self.p0hessian is not None
        kernel = op2.Kernel(get_reordered_eigendecomposition_kernel(self.dim), "get_reordered_eigendecomposition", cpp=True, include_dirs=include_dir)
        op2.par_loop(kernel, self.P0_ten.node_set, self.evec.dat(op2.RW), self.eval.dat(op2.RW), self.p0hessian.dat(op2.READ))
        s = sqrt(abs(self.eval[0]/self.eval[1]))
        self.eval.interpolate(as_vector([abs(self.K_hat/self.K_opt/s), abs(self.K_hat/self.K_opt*s)]))

    def get_element_size(self):
        """
        Compute current element volume, using reference element volume and Jacobian determinant.
        """
        self.K.interpolate(self.K_hat*abs(self.detJ))

    def get_optimal_element_size(self):
        """
        Compute optimal element size as the solution of one of two problems: either minimise
        interpolation error for a given metric complexity or minimise metric complexity for
        a given interpolation error.
        """
        assert self.eta is not None
        alpha = self.op.convergence_rate
        self.K_opt.interpolate(pow(self.eta, 1/(alpha+1)))
        Sum = self.K_opt.vector().gather().sum()
        if self.op.normalisation == 'error':
            scaling = pow(Sum*self.op.target, -1/alpha)  # FIXME
        else:
            scaling = Sum/self.op.target
        self.K_opt.interpolate(min_value(max_value(scaling*self.K/self.K_opt, self.op.h_min**2), self.op.h_max**2))

    def build_metric(self):
        """
        Construct a metric using the eigenvalues and eigenvectors already computed.

        NOTE: Assumes eigevalues are already squared.
        """
        kernel = op2.Kernel(set_eigendecomposition_kernel(self.dim), "set_eigendecomposition", cpp=True, include_dirs=include_dir)
        op2.par_loop(kernel, self.P0_ten.node_set, self.p0metric.dat(op2.RW), self.evec.dat(op2.READ), self.eval.dat(op2.READ))

    def project_metric(self):
        """
        Project elementwise metric to make it vertexwise.
        """
        self.p1metric.project(self.p0metric)

    def get_identity_metric(self):
        """
        Compute identity metric corresponding to current mesh.
        """
        self.get_eigenpair()
        self.build_metric()
        self.project_metric()

    def get_isotropic_metric(self):
        """
        Construct an isotropic metric for the provided error indicator.
        """
        self.get_element_size()
        self.get_optimal_element_size()
        indicator = Function(self.P1).interpolate(abs(self.K_hat/self.K_opt))
        self.p1metric = isotropic_metric(indicator, noscale=True, op=self.op)

    def get_anisotropic_metric(self):
        """
        Construct an anisotropic metric for the provided error indicator and Hessian.
        """
        self.get_element_size()
        self.get_optimal_element_size()
        self.get_hessian_eigenpair()
        self.build_metric()
        self.project_metric()

    def adapt_mesh(self):
        """
        Adapt mesh using vertexwise metric.
        """
        try:
            assert hasattr(self, 'p1metric')
        except ValueError:
            raise ValueError("Vertexwise metric does not exist. Please choose an adaptation strategy.")
        self.mesh = adapt(self.p1metric, op=self.op)

    def Lij(self, i, j):
        """
        Compute triple product integral i,j, to be used in error estimates.
        """
        eigenvectors = [self.evec[0], self.evec[1]]  # NOTE: These may need reordering
        triple_product = dot(eigenvectors[i], dot(self.p0hessian, eigenvectors[j]))
        return assemble(self.p0test*triple_product*triple_product*dx)

    def cell_interpolation_error(self):
        """
        Compute elementwise interpolation error as stated in [Carpio et al. 2013].
        """
        l = self.eval[0]*self.eval[0]*self.Lij(0, 0)
        l += self.eval[0]*self.eval[1]*self.Lij(0, 1)
        l += self.eval[1]*self.eval[0]*self.Lij(1, 0)
        l += self.eval[1]*self.eval[1]*self.Lij(1, 1)
        self.estimator.interpolate(sqrt(l))

    def gradient_interpolation_error(self):
        """
        Compute gradient-based interpolation error as stated in [Carpio et al. 2013].
        """
        self.cell_interpolation_error()
        coeff = pow(min_value(self.eval[0], self.eval[1]), -0.5)
        self.estimator.interpolate(coeff*self.estimator)

    def edge_interpolation_error(self):
        """
        Compute edge-based interpolation error as stated in [Carpio et al. 2013].
        """
        self.cell_interpolation_error()
        coeff = sqrt((self.eval[0]+self.eval[1])*pow(min_value(self.eval[0], self.eval[1]), 1.5))
        self.estimator.interpolate(coeff*self.estimator)
