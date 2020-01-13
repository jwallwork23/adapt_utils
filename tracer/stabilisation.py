from firedrake import *
try:
    import firedrake.cython.dmplex as dmplex
except ImportError:
    import firedrake.dmplex as dmplex  # Older Firedrake version

import numpy as np
import numpy.linalg as la

from adapt_utils.adapt.kernels import eigen_kernel, singular_value_decomposition
from adapt_utils.adapt.adaptation import AdaptiveMesh


__all__ = ["anisotropic_stabilisation"]


def cell_metric(mesh, metric=None):
    """
    Compute cell metric associated with mesh.

    Based on code by Lawrence Mitchell.
    """
    #print("Making cell metric on %s" % mesh)
    dim = mesh.topological_dimension()
    assert dim in (2, 3)
    P0_ten = TensorFunctionSpace(mesh, "DG", 0)
    J = interpolate(Jacobian(mesh), P0_ten)
    metric = metric or Function(P0_ten, name="CellMetric")
    kernel = eigen_kernel(singular_value_decomposition, dim)
    op2.par_loop(kernel, P0_ten.node_set, metric.dat(op2.INC), J.dat(op2.READ))
    return metric


def anisotropic_stabilisation(u, mesh=None):
    """
    Compute anisotropic stabilisation coefficient using `cell_metric` and the velocity field.
    """
    if mesh is None:
        mesh = u.function_space().mesh()
    M = cell_metric(mesh)
    P0 = FunctionSpace(mesh, "DG", 0)
    h = dot(u, dot(M, u))  # chosen measure of cell size which accounts for anisotropy
    tau = Function(P0)
    tau.interpolate(0.5*h/sqrt(inner(u, u)))
    return tau


def supg_coefficient(u, nu, mesh=None, anisotropic=False):
    r"""
    Compute SUPG stabilisation coefficent for an advection diffusion problem. There are two modes
    in which this can be calculated, as determined by the Boolean parameter `anisotropic`:

    In isotropic mode, we use the cell diameter as our measure of element size :math:`h_K`.

    In anisotropic mode, we follow [Nguyen et al., 2009] in looping over each element of the mesh,
    projecting the edge of maximal length into a vector space spanning the velocity field `u` and
    taking the length of this projected edge as the measure of element size.

    In both cases, we compute the stabilisation coefficent as

..  math::
    \tau = \frac{h_K}{2\|\textbf{u}\|}

    :arg u: velocity field associated with advection equation being solved.
    :arg nu: diffusivity of fluid.
    :kwarg mesh: mesh upon which problem is defined.
    :kwarg anisotropic: toggle between isotropic and anisotropic mode.
    """
    if mesh is None:
        mesh = u.function_space().mesh()
    h = anisotropic_h(u, mesh) if anisotropic else CellSize(mesh)
    Pe = 0.5*sqrt(inner(u, u))*h/nu
    tau = 0.5*h/sqrt(inner(u, u))
    return tau*min_value(1, Pe/3)


def anisotropic_h(u, mesh=None):
    """
    Measure of element size recommended in [Nguyen et al., 2009]: maximum edge length, projected onto
    the velocity field `u`.
    """
    func = isinstance(u, Function)  # Determine if u is a Function or a Constant
    if not func:
        try:
            assert isinstance(u, Constant)
        except AssertionError:
            raise ValueError("Velocity field should be either `Function` or `Constant`.")
    if mesh is None:
        mesh = AdaptiveMesh(u.function_space().mesh())
    else:
        assert isinstance(mesh, AdaptiveMesh)
    # edge_lengths = get_edge_lengths(mesh)
    # edge_vectors = get_edge_vectors(mesh)
    # TODO: Write a par_loop to get minimum length edge of each element

    # Loop over all elements and find the edge with maximum length
    for c in range(len(cell_to_vertices)):
        endpoints = [coords[v] for v in cell_to_vertices[c]]
        vectors = []
        lengths = []
        for i in range(3):
            vector = endpoints[(i+1) % 3] - endpoints[i] 
            vectors.append(vector)
            lengths.append(la.norm(vector))
        j = np.argmax(lengths)

        # Take maximum over all edges  # TODO: Spatially varying version
        if lengths[j] > global_max_edge_length:
            global_max_edge_length = lengths[j]
            global_max_edge_vector = vectors[j]

    v = global_max_edge_vector
    if func:
        fs = FunctionSpace(mesh, u.ufl_element().family(), u.ufl_element().degree())
        h = interpolate((u[0]*v[0] + u[1]*v[1])/sqrt(dot(u, u)), P0)
        h = h.vector().gather().max()  # TODO: Spatially varying version
    else:
        udat = u.dat.data[0]
        h = Constant(udat[0]*v[0] + udat[1]*v[1])

    return h
