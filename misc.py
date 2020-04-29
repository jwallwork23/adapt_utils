from thetis import *

import os
import fnmatch
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla

from adapt_utils.adapt.kernels import eigen_kernel, get_eigendecomposition


__all__ = ["copy_mesh", "find", "get_finite_element", "get_component_space", "get_component",
           "cg2dg", "check_spd", "get_boundary_nodes", "index_string"]


def copy_mesh(mesh):
    """Deepcopy a mesh."""
    return Mesh(Function(mesh.coordinates))


def find(pattern, path):
    """Find all files with a specified pattern."""
    result = []
    for root, dirs, files in os.walk(path):
        for name in files:
            if fnmatch.fnmatch(name, pattern):
                result.append(os.path.join(root, name))
    return result


def get_finite_element(fs, variant='equispaced'):
    """
    Extract :class:`FiniteElement` instance from a :class:`FunctionSpace`, with specified variant
    default.
    """
    el = fs.ufl_element()
    if hasattr(el, 'variant'):
        variant = el.variant() or variant
    return FiniteElement(el.family(), fs.mesh().ufl_cell(), el.degree(), variant=variant)


def get_component_space(fs, variant='equispaced'):
    """
    Extract a single (scalar) :class:`FunctionSpace` component from a :class:`VectorFunctionSpace`.
    """
    return FunctionSpace(fs.mesh(), get_finite_element(fs, variant=variant))


def get_component(f, index, component_space=None):
    """
    Extract component `index` of a :class:`Function` from a :class:`VectorFunctionSpace` and store
    it in a :class:`Function` defined on the appropriate (scalar) :class:`FunctionSpace`. The
    component space can either be provided or computed on-the-fly.
    """
    n = f.ufl_shape[0]
    try:
        assert index < n
    except AssertionError:
        raise IndexError("Requested index {:d} of a {:d}-vector.".format(index, n))

    # Create appropriate component space
    fi = Function(component_space or get_component_space(fs))

    # Transfer data
    par_loop(('{[i] : 0 <= i < v.dofs}', 's[i] = v[i, %d]' % index), dx,
             {'v': (f, READ), 's': (fi, WRITE)}, is_loopy_kernel=True)
    return fi


def cg2dg(f_cg, f_dg=None):
    """
    Transfer data from a the degrees of freedom of a Pp field directly to those of the
    corresponding PpDG field, for some p>1.
    """
    n = len(f_cg.ufl_shape)
    assert f_cg.ufl_element().family() == 'Lagrange'
    if n == 0:
        _cg2dg_scalar(f_cg, f_dg)
    elif n == 1:
        _cg2dg_vector(f_cg, f_dg)
    elif n == 2:
        _cg2dg_tensor(f_cg, f_dg)
    else:
        raise NotImplementedError


def _cg2dg_scalar(f_cg, f_dg):
    fs = f_cg.function_space()
    f_dg = f_dg or Function(FunctionSpace(fs.mesh(), "DG", fs.ufl_element().degree()))
    index = '{[i] : 0 <= i < cg.dofs}'
    kernel = 'dg[i] = cg[i]'
    par_loop((index, kernel), dx, {'cg': (f_cg, READ), 'dg': (f_dg, WRITE)}, is_loopy_kernel=True)


def _cg2dg_vector(f_cg, f_dg):
    fs = f_cg.function_space()
    f_dg = f_dg or Function(VectorFunctionSpace(fs.mesh(), "DG", fs.ufl_element().degree()))
    index = '{[i, j] : 0 <= i < cg.dofs and 0 <= j < %d}' % f_cg.ufl_shape
    kernel = 'dg[i, j] = cg[i, j]'
    par_loop((index, kernel), dx, {'cg': (f_cg, READ), 'dg': (f_dg, WRITE)}, is_loopy_kernel=True)


def _cg2dg_tensor(f_cg, f_dg):
    fs = f_cg.function_space()
    f_dg = f_dg or Function(TensorFunctionSpace(fs.mesh(), "DG", fs.ufl_element().degree()))
    index = '{[i, j, k] : 0 <= i < cg.dofs and 0 <= j < %d and 0 <= k < %d}' % f_cg.ufl_shape
    kernel = 'dg[i, j, k] = cg[i, j, k]'
    par_loop((index, kernel), dx, {'cg': (f_cg, READ), 'dg': (f_dg, WRITE)}, is_loopy_kernel=True)


def check_spd(M):
    """
    Verify that a tensor field `M` is symmetric positive-definite (SPD) and hence a Riemannian
    metric.
    """
    print_output("TEST: Checking matrix is SPD...")
    fs = M.function_space()
    fs_vec = VectorFunctionSpace(fs.mesh(), get_finite_element(fs))
    dim = fs.mesh().topological_dimension()

    # Check symmetric
    try:
        assert assemble((M - transpose(M))*dx) < 1e-8
    except AssertionError:
        raise ValueError("FAIL: Matrix is not symmetric")
    print_output("PASS: Matrix is indeed symmetric")

    # Check positive definite
    V = Function(fs, name="Eigenvectors")
    Λ = Function(fs_vec, name="Eigenvalues")
    kernel = eigen_kernel(get_eigendecomposition, dim)
    op2.par_loop(kernel, fs.node_set, V.dat(op2.RW), Λ.dat(op2.RW), M.dat(op2.READ))
    try:
        assert Λ.vector().gather().min() > 0.0
    except AssertionError:
        raise ValueError("FAIL: Matrix is not positive-definite")
    print_output("PASS: Matrix is indeed positive-definite")
    print_output("TEST: Done!")


def index_string(index):
    """
    :arg index: integer form of index.
    :return: five-digit string form of index.
    """
    return (5 - len(str(index)))*'0' + str(index)


def get_boundary_nodes(fs, segment='on_boundary'):
    """
    :arg fs: function space to get boundary nodes for.
    :kwarg segment: segment of boundary to get nodes of (default 'on_boundary').
    """
    return fs.boundary_nodes(segment, 'topological')


# --- Unused


class BaseConditionCheck():

    def __init__(self, bilinear_form, mass_term=None):
        self.a = bilinear_form
        self.A = assemble(bilinear_form).M.handle
        if mass_term is not None:
            self.M = assemble(mass_term).M.handle

    def _get_wrk(self, *args):
        pass

    def _get_csr(self):
        indptr, indices, data = self.wrk.getValuesCSR()
        self.csr = sp.csr_matrix((data, indices, indptr), shape=self.wrk.getSize())

    def condition_number(self, *args, eps=1e-10):
        self._get_wrk(*args)
        try:
            from slepc4py import SLEPc
        except ImportError:
            # If SLEPc is not available, use SciPy
            self._get_csr()
            eigval = sla.eigs(self.csr)[0]
            eigmin = np.min(eigval)
            if eigmin > 0.0:
                print_output("Mass matrix is positive-definite")
            else:
                print_output("Mass matrix is not positive-definite. Minimal eigenvalue: {:.4e}".format(eigmin))
            eigval = np.abs(eigval)
            return np.max(eigval)/max(np.min(eigval), eps)

        if hasattr(self, 'M'):
            raise NotImplementedError  # TODO
        else:
            from firedrake.petsc import PETSc

            # Solve eigenvalue problem
            n = self.wrk.getSize()[0]
            es = SLEPc.EPS().create(comm=COMM_WORLD)
            es.setDimensions(n)
            es.setOperators(self.wrk)
            opts = PETSc.Options()
            opts.setValue('eps_type', 'krylovschur')
            opts.setValue('eps_monitor', None)
            # opts.setValue('eps_pc_type', 'lu')
            # opts.setValue('eps_pc_factor_mat_solver_type', 'mumps')
            es.setFromOptions()
            print_output("Solving eigenvalue problem using SLEPc...")
            es.solve()

            # Check convergence
            nconv = es.getConverged()
            if nconv == 0:
                import sys
                warning("Did not converge any eigenvalues")
                sys.exit(0)

            # Compute condition number
            vr, vi = self.wrk.getVecs()
            eigmax = es.getEigenpair(0, vr, vi).real
            eigmin = es.getEigenpair(n-1, vr, vi).real
            eigmin = max(eigmin, eps)
            if eigmin > 0.0:
                print_output("Mass matrix is positive-definite")
            else:
                print_output("Mass matrix is not positive-definite. Minimal eigenvalue: {:.4e}".format(eigmin))
            return np.abs(eigmax/eigmin)


class UnnestedConditionCheck(BaseConditionCheck):

    def __init__(self, bilinear_form):
        super(UnnestedConditionCheck, self).__init__(bilinear_form)
        try:
            assert self.A.getType() != 'nest'
        except AssertionError:
            raise ValueError("Matrix type 'nest' not supported. Use `NestedConditionCheck` instead.")

    def _get_wrk(self):
        self.wrk = self.A


class NestedConditionCheck(BaseConditionCheck):

    def __init__(self, bilinear_form):
        super(NestedConditionCheck, self).__init__(bilinear_form)
        try:
            assert self.A.getType() == 'nest'
        except AssertionError:
            raise ValueError("Matrix type {:} not supported.".format(self.A.getType()))
        m, n = self.A.getNestSize()
        self.submatrices = {}
        for i in range(m):
            self.submatrices[i] = {}
            for j in range(n):
                self.submatrices[i][j] = self.A.getNestSubMatrix(i, j)

    def _get_wrk(self, i, j):
        self.wrk = self.submatrices[i][j]
