from thetis import *

import math
import numpy as np

from adapt_utils.adapt.kernels import eigen_kernel, get_eigendecomposition
from adapt_utils.fem import get_finite_element


__all__ = ["rotation_matrix", "rotate", "is_symmetric", "is_pos_def", "is_spd", "check_spd"]


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
