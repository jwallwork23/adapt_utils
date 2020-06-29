from thetis import *

import os
import sys
import math
import fnmatch
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla

from adapt_utils.adapt.kernels import eigen_kernel, get_eigendecomposition


__all__ = ["prod", "combine", "box", "ellipse", "bump", "circular_bump", "gaussian",
           "copy_mesh", "get_finite_element", "get_component_space", "get_component", "cg2dg",
           "check_spd", "get_boundary_nodes", "index_string",
           "find", "suppress_output", "knownargs2dict", "unknownargs2dict"]


def abs(u):
    """Hack due to the fact `abs` seems to be broken in conditional statements."""
    return conditional(u < 0, -u, u)


def prod(arr):
    """Helper function for taking the product of an array (similar to `sum`)."""
    n = len(arr)
    if n == 0:
        raise ValueError
    elif n == 1:
        return arr[0]
    else:
        return arr[0]*prod(arr[1:])


def combine(operator, *args):
    """Helper function for repeatedly application of binary operators."""
    n = len(args)
    if n == 0:
        raise ValueError
    elif n == 1:
        return args[0]
    else:
        return operator(args[0], combine(operator, *args[1:]))


def rotation_matrix(theta):
    """NumPy array 2D rotation matrix associated with angle `theta`."""
    return np.array([[math.cos(theta), -math.sin(theta)],
                    [math.sin(theta), math.cos(theta)]])


def rotate(vectors, angle):
    """Rotate a list of 2D `vector`s by `angle` about the origin."""
    R = rotation_matrix(angle)
    for i in range(len(vectors)):
        assert len(vectors[i]) == 2
        vectors[i] = np.dot(R, vectors[i])


def box(locs, mesh, scale=1.0, rotation=None):
    r"""
    Rectangular indicator functions associated with a list of region of interest tuples.

    Takes the value `scale` in the region

  ..math::
        (|x - x0| < r_x) && (|y - y0| < r_y)

    centred about (x0, y0) and zero elsewhere. Similarly for other dimensions.

    :kwarg scale: scale factor for indicator.
    :kwarg rotation: angle by which to rotate.
    """
    d = mesh.topological_dimension()
    dims, L = range(d), range(len(locs))  # Itersets
    x = SpatialCoordinate(mesh)

    # Get distances from origins and RHS values
    X = [[x[i] - locs[j][i] for i in dims] for j in L]
    r = [[locs[j][d] if len(locs[j]) == d+1 else locs[j][d+i] for i in dims] for j in L]

    # Apply rotations
    if rotation is not None:
        rotate(X, rotation)

    # Combine to get indicator
    expr = [combine(And, *[lt(abs(X[j][i]), r[j][i]) for i in dims]) for j in L]
    return conditional(combine(Or, *expr), scale, 0.0)


def ellipse(locs, mesh, scale=1.0, rotation=None):
    r"""
    Ellipse indicator function associated with a list of region of interest tuples.

    Takes the value `scale` in the region

  ..math::
        (x - x_0)^2/r_x^2 + (y - y_0)^2/r_y^2 < 1

    and zero elsewhere. Similarly for other dimensions.

    :kwarg scale: scale factor for indicator.
    :kwarg rotation: angle by which to rotate.
    """
    d = mesh.topological_dimension()
    dims, L = range(d), range(len(locs))  # Itersets
    x = SpatialCoordinate(mesh)

    # Get distances from origins and RHS values
    X = [[x[i] - locs[j][i] for i in dims] for j in L]
    r = [[locs[j][d] if len(locs[j]) == d+1 else locs[j][d+i] for i in dims] for j in L]

    # Apply rotations
    if rotation is not None:
        rotate(X, rotation)

    # Combine to get indicator
    conditions = [lt(sum((X[j][i]/r[j][i])**2 for i in dims), 1) for j in L]
    return conditional(combine(Or, *conditions), scale, 0)


def bump(locs, mesh, scale=1.0, rotation=None):
    r"""
    Rectangular bump function associated with a list of region of interest tuples.
    (A smooth approximation to the box function.)

    Takes the form

  ..math::
        \exp\left(1 - \frac1{\left(1 - \left(\frac{x - x_0}{r_x}\right)^2\right)}\right)
        * \exp\left(1 - \frac1{\left(1 - \left(\frac{y - y_0}{r_y}\right)^2\right)}\right)

    scaled by `scale` inside the box region. Similarly for other dimensions.

    Note that we assume the provided regions are disjoint for this indicator.

    :kwarg scale: scale factor for indicator.
    :kwarg rotation: angle by which to rotate.
    """
    d = mesh.topological_dimension()
    dims, L = range(d), range(len(locs))  # Itersets
    x = SpatialCoordinate(mesh)

    # Get distances from origins and RHS values
    X = [[x[i] - locs[j][i] for i in dims] for j in L]
    r = [[locs[j][d] if len(locs[j]) == d+1 else locs[j][d+i] for i in dims] for j in L]

    # Apply rotations
    if rotation is not None:
        rotate(X, rotation)

    # Combine to get indicator
    q = [[(X[j][i]/r[j][i])**2 for i in dims] for j in L]  # Quotients of squared distances
    conditions = [combine(And, *[lt(q[j][i], 1) for i in dims]) for j in L]
    bumps = [prod([exp(1 - 1/(1 - q[j][i])) for i in dims]) for j in L]
    return sum([conditional(conditions[j], scale*bumps[j], 0) for j in L])


# TODO: Elliptical bump
def circular_bump(locs, mesh, scale=1.0, rotation=None):
    r"""
    Circular bump function associated with a list of region of interest tuples.
    (A smooth approximation to the ball function.)

    Defining the radius :math:`r^2 := (x - x_0)^2 + (y - y_0)^2`, the circular bump takes the
    form

  ..math::
        \exp\left(1 - \frac1{\left1 - \frac{r^2}{r_0^2}\right)}\right)

    scaled by `scale` inside the ball region. Similarly for other dimensions.

    :kwarg scale: scale factor for indicator.
    :kwarg rotation: angle by which to rotate.
    """
    d = mesh.topological_dimension()
    dims, L = range(d), range(len(locs))  # Itersets
    x = SpatialCoordinate(mesh)

    # Get distances from origins and RHS values
    X = [[x[i] - locs[j][i] for i in dims] for j in L]
    r_sq = [[locs[j][d]**2 if len(locs[j]) == d+1 else locs[j][d+i]**2 for i in dims] for j in L]

    # Apply rotations
    if rotation is not None:
        rotate(X, rotation)

    # Combine to get indicator
    q = [sum([X[j][i]**2 for i in dims])/sum(r_sq[j]) for j in L]  # Quotient of squared 2-norms
    return sum([conditional(lt(q[j], 1), scale*exp(1 - 1/(1 - q[j])), 0) for j in L])


def gaussian(locs, mesh, scale=1.0, rotation=None):
    r"""
    Gaussian bell associated with a list of region of interest tuples.

    Takes the form

  ..math::
        \exp\left(- \left(\frac{x^2}{r_x^2} + \frac{y^2}{r_y^2}\right)\right)

    scaled by `scale` inside the ball region. Similarly for other dimensions.

    :kwarg scale: scale factor for indicator.
    :kwarg rotation: angle by which to rotate.
    """
    d = mesh.topological_dimension()
    dims, L = range(d), range(len(locs))  # Itersets
    x = SpatialCoordinate(mesh)

    # Get distances from origins and RHS values
    X = [[x[i] - locs[j][i] for i in dims] for j in L]
    r = [[locs[j][d] if len(locs[j]) == d+1 else locs[j][d+i] for i in dims] for j in L]

    # Apply rotations
    if rotation is not None:
        rotate(X, rotation)

    # Combine to get indicator
    q_sq = [sum((X[j][i]/r[j][i])**2 for i in dims) for j in L]  # Quotient of squares
    return sum(scale*conditional(lt(q_sq[j], 1), exp(-q_sq[j]), 0) for j in L)


def copy_mesh(mesh):
    """Deepcopy a mesh."""
    return Mesh(Function(mesh.coordinates))


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
    fi = Function(component_space or get_component_space(f.function_space()))

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


def get_boundary_nodes(fs, segment='on_boundary'):
    """
    :arg fs: function space to get boundary nodes for.
    :kwarg segment: segment of boundary to get nodes of (default 'on_boundary').
    """
    return fs.boundary_nodes(segment, 'topological')


def index_string(index):
    """
    :arg index: integer form of index.
    :return: five-digit string form of index.
    """
    return (5 - len(str(index)))*'0' + str(index)


# --- Non-Firedrake specific


def find(pattern, path):
    """Find all files with a specified pattern."""
    result = []
    for root, dirs, files in os.walk(path):
        for name in files:
            if fnmatch.fnmatch(name, pattern):
                result.append(os.path.join(root, name))
    return result


def knownargs2dict(ka):
    """Extract all public attributes from namespace `ka` and return as a dictionary."""
    out = {}
    for arg in [arg for arg in dir(ka) if arg[0] != '_']:
        attr = ka.__getattribute__(arg)
        if attr is not None:
            if attr == '1':
                out[arg] = None
            # TODO: Account for integers
            # TODO: Account for floats
            else:
                out[arg] = attr
    return out


def unknownargs2dict(ua):
    """Extract all public attributes from list `ua` and return as a dictionary."""
    out = {}
    for i in range(len(ua)//2):
        key = ua[2*i][1:]
        val = ua[2*i+1]
        if val == '1':
            out[key] = None
        # TODO: Account for integers
        # TODO: Account for floats
        else:
            out[key] = val
    return out


class suppress_output(object):
    """Class used to suppress standard output or another stream."""
    def __init__(self, stream=None):
        self.origstream = stream or sys.stdout
        self.origstreamfd = self.origstream.fileno()
        self.pipe_out, self.pipe_in = os.pipe()  # Create a pipe to capture stream

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, type, value, traceback):
        self.stop()

    def start(self):
        """Start diverting stream data to pipe."""
        os.dup2(self.pipe_in, self.origstreamfd)

    def stop(self):
        """Stop fiverting stream data to pipe."""
        os.close(self.pipe_in)
        os.close(self.pipe_out)


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
