from thetis import *

import matplotlib.pyplot as plt
import numpy as np


__all__ = ["MeshStats", "isotropic_cell_size", "anisotropic_cell_size", "make_consistent",
           "get_patch"]


class MeshStats(object):
    """
    A class for holding various statistics related to a given mesh.
    """
    def __init__(self, op, mesh=None):
        """
        :arg op: :class:`Options` parameter object.
        :kwarg mesh: if a mesh is not provided, the :attr:`default_mesh` associated with :attr:`op`
            is used.
        """
        from .misc import integrate_boundary

        self._op = op
        self._mesh = mesh or op.default_mesh
        self._P0 = FunctionSpace(mesh, "DG", 0)
        self.boundary_markers = mesh.exterior_facets.unique_markers
        self.dim = self._mesh.topological_dimension()
        self.num_cells = mesh.num_cells()
        self.num_edges = mesh.num_edges()
        self.num_vertices = mesh.num_vertices()

        # Compute statistics
        self.get_element_sizes()
        self.facet_areas = get_facet_areas(self._mesh)
        self.boundary_lengths = integrate_boundary(self._mesh)
        self.boundary_length = sum(self.boundary_lengths[tag] for tag in self.boundary_markers)
        if self.dim == 2:
            self.angles_min = get_minimum_angles_2d(self._mesh)
            self.angle_min = self.angles_min.vector().gather().min()
            self.get_element_volumes()
        elif self.dim != 3:
            raise ValueError("Mesh of dimension {:d} not supported.".format(self.dim))
        op.print_debug(self.summary)

    @property
    def summary(self):
        msg = "\n" + 35*"*" + "\n" + 10*" " + "MESH STATISTICS\n" + 35*"*" + "\n"
        msg += "MESH: num cells       = {:11d}\n".format(self.num_cells)
        msg += "MESH: num edges       = {:11d}\n".format(self.num_edges)
        msg += "MESH: num vertices    = {:11d}\n".format(self.num_vertices)
        msg += "MESH: min(dx)         = {:11.4e}\n".format(self.dx_min)
        msg += "MESH: max(dx)         = {:11.4e}\n".format(self.dx_max)
        if self.dim == 2:
            msg += "MESH: min(angles)     = {:11.4e}\n".format(self.angle_min)
            msg += "MESH: min(vol)        = {:11.4e}\n".format(self.volume_min)
            msg += "MESH: max(vol)        = {:11.4e}\n".format(self.volume_max)
            msg += "MESH: boundary length = {:11.4e}\n".format(self.boundary_length)
        msg += 35*"*" + "\n"
        return msg

    def get_element_sizes(self, anisotropic=False):
        cell_size_measure = anisotropic_cell_size if anisotropic else isotropic_cell_size
        self.dx = cell_size_measure(self._mesh)
        self.dx_min = self.dx.vector().gather().min()
        self.dx_max = self.dx.vector().gather().max()

    def get_element_volumes(self):
        if self.dim == 3:
            raise NotImplementedError  # TODO
        self.volume = Function(self._P0, name="Element volume")
        get_horizontal_elem_size_2d(self.volume)
        self.volume_min = self.volume.vector().gather().min()
        self.volume_max = self.volume.vector().gather().max()


def isotropic_cell_size(mesh):
    """
    Standard measure of cell size, as determined by UFL's `CellSize`.
    """
    # print_output("MESH: Computing isotropic cell size")
    P0 = FunctionSpace(mesh, "DG", 0)
    return interpolate(CellSize(mesh), P0)


def anisotropic_cell_size(mesh):
    """
    Measure of cell size for anisotropic meshes, as described in [Micheletti, Perotto & Picasso 2003]
    """
    from adapt_utils.adapt.kernels import eigen_kernel, get_reordered_eigendecomposition, poldec_spd

    # print_output("MESH: Computing anisotropic cell size")
    dim = mesh.topological_dimension()

    # Compute cell Jacobian
    P0_ten = TensorFunctionSpace(mesh, "DG", 0)
    J = Function(P0_ten, name="Cell Jacobian")
    J.interpolate(Jacobian(mesh))

    # Get SPD part of polar decomposition
    B = Function(P0_ten, name="SPD part")
    op2.par_loop(eigen_kernel(poldec_spd, dim), P0_ten.node_set, B.dat(op2.RW), J.dat(op2.READ))

    # Get eigendecomposition with eigenvalues decreasing in magnitude
    P0_vec = VectorFunctionSpace(mesh, "DG", 0)
    evalues = Function(P0_vec, name="Eigenvalues")
    evectors = Function(P0_ten, name="Eigenvectors")
    kernel = eigen_kernel(get_reordered_eigendecomposition, dim)
    op2.par_loop(kernel, P0_ten.node_set, evectors.dat(op2.RW), evalues.dat(op2.RW), B.dat(op2.READ))

    # Return minimum eigenvalue
    return interpolate(evalues[-1], FunctionSpace(mesh, "DG", 0))


def make_consistent(mesh):
    """
    Make the coordinates associated with a Firedrake mesh and its underlying PETSc DMPlex
    use a consistent numbering.
    """
    import firedrake.cython.dmcommon as dmplex

    # Create section
    dim = mesh.topological_dimension()
    gdim = mesh.geometric_dimension()
    entity_dofs = np.zeros(dim+1, dtype=np.int32)
    entity_dofs[0] = gdim
    try:
        coord_section = dmplex.create_section(mesh, entity_dofs)
    except AttributeError:
        P0 = FunctionSpace(mesh, "DG", 0)  # NOQA
        coord_section = dmplex.create_section(mesh, entity_dofs)

    # Set plex coords to mesh coords
    plex = mesh._topology_dm
    dm_coords = plex.getCoordinateDM()
    dm_coords.setDefaultSection(coord_section)
    coords_local = dm_coords.createLocalVec()
    coords_local.array[:] = np.reshape(mesh.coordinates.dat.data, coords_local.array.shape)
    plex.setCoordinatesLocal(coords_local)

    # Functions for getting offsets of entities and coordinates of vertices
    offset = lambda index: coord_section.getOffset(index)//dim
    coordinates = lambda index: mesh.coordinates.dat.data[offset(index)]
    return plex, offset, coordinates


def get_patch(vertex, mesh=None, plex=None, coordinates=None, midfacets=False, extend=None):
    """
    Generate an element patch around a vertex.

    :kwarg extend: optionally take the union with an existing patch.
    """
    if extend is None:
        elements = set([])
        facets = set([])
    else:
        elements = set(extend['elements'].keys())
        if midfacets:
            facets = set(extend['facets'].keys())
    if coordinates is None:
        assert mesh is not None
        plex, offset, coordinates = make_consistent(mesh)
    plex = plex or mesh._topology_dm
    dim = plex.getDimension()
    assert dim in (2, 3)
    if mesh is not None:
        cell = mesh.ufl_cell()
        if (dim == 2 and cell != triangle) or (dim == 3 and cell != tetrahedron):
            raise ValueError("Element type {:} not supported".format(cell))

    # Get patch of neighbouring elements
    for e in plex.getSupport(vertex):
        elements = elements.union(set(plex.getSupport(e)))
    patch = {'elements': {k: {'vertices': []} for k in elements}}

    # Get vertices and centroids in patch
    vertices = set(range(*plex.getDepthStratum(0)))
    patch['vertices'] = set([])
    for k in elements:
        closure = set(plex.getTransitiveClosure(k)[0])
        patch['elements'][k]['vertices'] = vertices.intersection(closure)
        coords = [coordinates(v) for v in patch['elements'][k]['vertices']]
        patch['elements'][k]['centroid'] = np.sum(coords, axis=0)/(dim + 1)
        patch['vertices'] = patch['vertices'].union(set(patch['elements'][k]['vertices']))

    if midfacets:

        # Get facets in patch
        all_facets = set(range(*plex.getDepthStratum(1)))
        patch_facets = set([])
        for k in elements:
            closure = set(plex.getTransitiveClosure(k)[0])
            patch_facets = patch_facets.union(all_facets.intersection(closure))
        patch_facets.union(facets)
        patch['facets'] = {e: {} for e in patch_facets}

        # Get their centroids
        for e in patch['facets']:
            closure = set(plex.getTransitiveClosure(e)[0])
            patch['facets'][e] = {'vertices': vertices.intersection(closure)}
            coords = [coordinates(v) for v in plex.getCone(e)]
            patch['facets'][e]['midfacet'] = np.sum(coords, axis=0)/dim

    return patch


# FIXME: Why do rotations of the same element not have the same quality?
def quality(mesh):
    r"""
    Compute the scaled Jacobian for each mesh element:
..  math::
        Q(K) = \frac{\det(J_K)}{\|\mathbf e_1\|\,\|\mathbf e2\|},

    where element :math:`K` is defined by edges :math:`\mathbf e_1` and :math:`\mathbf e_2`.

    NOTE that :math:`J_K = [\mathbf e_1, \mathbf e_2]`.
    """
    assert mesh.topological_dimension() == 2
    P0 = FunctionSpace(mesh, "DG", 0)
    P0_ten = TensorFunctionSpace(mesh, "DG", 0)
    J = interpolate(Jacobian(mesh), P0_ten)
    detJ = JacobianDeterminant(mesh)
    jacobian_sign = interpolate(sign(detJ), P0)
    # unswapped = as_matrix([[J[0, 0], J[0, 1]], [J[1, 0], J[1, 1]]])
    # swapped = as_matrix([[J[1, 0], J[1, 1]], [J[0, 0], J[0, 1]]])
    # sgn = Function(P0)
    # sgn.dat.data[:] = jacobian_sign.dat.data
    # J.interpolate(conditional(ge(jacobian_sign, 0), unswapped, swapped))
    # J.interpolate(conditional(ge(sgn, 0), unswapped, swapped))
    # detJ = det(J)
    edge1 = as_vector([J[0, 0], J[1, 0]])
    edge2 = as_vector([J[0, 1], J[1, 1]])
    norm1 = sqrt(dot(edge1, edge1))
    norm2 = sqrt(dot(edge2, edge2))
    scaled_jacobian = interpolate(detJ/(norm1*norm2*jacobian_sign), P0)
    return scaled_jacobian


# FIXME: Inverted elements do not show! Tried making transparent but it didn't do anything.
def plot_quality(mesh, fig=None, axes=None, extensions=['png']):
    """
    Plot scaled Jacobian using a discretised scale:
      * green   : high quality elements (over 75%);
      * yellow  : medium quality elements (50 - 75%);
      * blue    : low quality elements (0 - 50%);
      * magenta : inverted elements (quality < 0).
    """
    q = quality(mesh)

    cmap = plt.get_cmap('viridis', 30)
    newcolours = cmap(np.linspace(0, 1, 30))
    newcolours[:10] = np.array([1, 0, 1, 1])    # Magenta
    newcolours[10:20] = np.array([0, 1, 1, 1])  # Cyan
    newcolours[20:25] = np.array([1, 1, 0, 1])  # Yellow
    newcolours[25:] = np.array([0, 1, 0, 1])    # Green

    if fig is None or axes is None:
        fig, axes = plt.subplots()
    triplot(mesh, axes=axes)
    fig.colorbar(tricontourf(q, axes=axes, cmap=cmap), ax=axes)
