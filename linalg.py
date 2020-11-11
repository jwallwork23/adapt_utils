from thetis import *

import math
import numpy as np

from adapt_utils.adapt.kernels import eigen_kernel, get_eigendecomposition
from adapt_utils.fem import get_finite_element


__all__ = ["rotation_matrix", "rotate", "is_symmetric", "is_pos_def", "is_spd", "check_spd",
           "gram_schmidt", "get_orthonormal_vectors"]


# --- Rotation

def rotation_matrix(theta):
    """
    NumPy array 2D rotation matrix associated with angle `theta`.
    """
    return np.array([[math.cos(theta), -math.sin(theta)],
                    [math.sin(theta), math.cos(theta)]])


def rotate(vectors, angle):
    """
    Rotate a list of 2D `vector`s by `angle` about the origin.
    """
    R = rotation_matrix(angle)
    for i in range(len(vectors)):
        assert len(vectors[i]) == 2
        vectors[i] = np.dot(R, vectors[i])


# --- Matrix properties

def is_symmetric(M, tol=1.0e-08):
    """
    Determine whether or not a tensor field `M` is symmetric.
    """
    area = assemble(Constant(1.0, domain=M.function_space().mesh())*dx)
    return assemble(abs(det(M - transpose(M)))*dx)/area < tol


def is_pos_def(M):
    """
    Determine whether or not a tensor field `M` is positive-definite.
    """
    fs = M.function_space()
    fs_vec = VectorFunctionSpace(fs.mesh(), get_finite_element(fs))
    V = Function(fs, name="Eigenvectors")
    Λ = Function(fs_vec, name="Eigenvalues")
    kernel = eigen_kernel(get_eigendecomposition, fs.mesh().topological_dimension())
    op2.par_loop(kernel, fs.node_set, V.dat(op2.RW), Λ.dat(op2.RW), M.dat(op2.READ))
    return Λ.vector().gather().min() > 0.0


def is_spd(M):
    """
    Determine whether or not a tensor field `M` is symmetric positive-definite.
    """
    return is_symmetric(M) and is_pos_def(M)


def check_spd(M):
    """
    Verify that a tensor field `M` is symmetric positive-definite.
    """
    print_output("TEST: Checking matrix is SPD...")

    # Check symmetric
    try:
        assert is_symmetric(M)
    except AssertionError:
        raise ValueError("FAIL: Matrix is not symmetric")
    print_output("PASS: Matrix is indeed symmetric")

    # Check positive definite
    try:
        assert is_pos_def(M)
    except AssertionError:
        raise ValueError("FAIL: Matrix is not positive-definite")
    print_output("PASS: Matrix is indeed positive-definite")


# --- Orthogonalisation

def gram_schmidt(*v, normalise=False):
    """
    Apply the Gram-Schmidt orthogonalisation process to a sequence of vectors in order to obtain
    an orthogonal basis.

    :args v: list of vectors to orthogonalise.
    :kwarg normalise: if `True`, an orthonormal basis is constructed.
    """
    if isinstance(v[0], np.ndarray):
        from numpy import dot, sqrt
    else:
        from ufl import dot, sqrt
    u = []
    proj = lambda x, y: dot(x, y)/dot(x, x)*x
    for i, vi in enumerate(v):
        vv = vi
        if i > 0:
            vv -= sum([proj(uj, vi) for uj in u])
        u.append(vv/sqrt(dot(vv, vv)) if normalise else vv)
    if isinstance(v[0], np.ndarray):
        u = [np.array(ui) for ui in u]
    return u


def get_orthonormal_vectors(n, dim=None, seed=0):
    """
    Given a vector `n`, get a set of orthonormal vectors.
    """
    np.random.seed(seed)
    dim = dim or n.ufl_domain().topological_dimension()
    if dim == 2:
        return [perp(n)]
    elif dim > 2:
        vectors = [as_vector(np.random.rand(dim)) for i in range(dim-1)]  # Arbitrary
        return gram_schmidt(n, *vectors, normalise=True)[1:]  # Orthonormal
    else:
        raise ValueError("Cannot get tangent vector in {:} dimensions.".format(dim))
