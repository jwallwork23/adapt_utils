from thetis import *

import numpy as np

from adapt_utils.plotting import *


__all__ = ["MeshStats", "isotropic_cell_size", "anisotropic_cell_size", "make_consistent",
           "get_patch", "quality", "plot_quality", "aspect_ratio", "plot_aspect_ratio"]


class MeshStats(object):
    """
    A class for holding various statistics related to a given mesh.
    """
    def __init__(self, op, mesh=None, initial_signs=None):
        """
        :arg op: :class:`Options` parameter object.
        :kwarg mesh: if a mesh is not provided, the :attr:`default_mesh` associated with :attr:`op`
            is used.
        """
        from .misc import integrate_boundary

        self._op = op
        self._mesh = mesh or op.default_mesh
        self._P0 = FunctionSpace(self._mesh, "DG", 0)
        self.boundary_markers = self._mesh.exterior_facets.unique_markers
        self.dim = self._mesh.topological_dimension()
        self.num_cells = self._mesh.num_cells()
        self.num_edges = self._mesh.num_edges()
        self.num_vertices = self._mesh.num_vertices()

        # Compute statistics
        self.get_element_sizes()
        self.facet_areas = get_facet_areas(self._mesh)
        self.boundary_lengths = integrate_boundary(self._mesh)
        self.boundary_length = sum(self.boundary_lengths[tag] for tag in self.boundary_markers)
        if self.dim == 2:
            self.angles_min = get_minimum_angles_2d(self._mesh)
            self.angle_min = self.angles_min.vector().gather().min()
            self.get_element_volumes()
            self.aspect_ratio = aspect_ratio(self._mesh)
            self.scaled_jacobian = quality(self._mesh, initial_signs=initial_signs)
        elif self.dim != 3:
            raise ValueError("Mesh of dimension {:d} not supported.".format(self.dim))
        op.print_debug(self.summary)

    @property
    def summary(self):
        msg = "\n" + 40*"*" + "\n" + 10*" " + "MESH STATISTICS\n" + 40*"*" + "\n"
        msg += "MESH: num cells           = {:11d}\n".format(self.num_cells)
        msg += "MESH: num edges           = {:11d}\n".format(self.num_edges)
        msg += "MESH: num vertices        = {:11d}\n".format(self.num_vertices)
        msg += "MESH: min(dx)             = {:11.4e}\n".format(self.dx_min)
        msg += "MESH: max(dx)             = {:11.4e}\n".format(self.dx_max)
        if self.dim == 2:
            msg += "MESH: min(angles)         = {:11.4e}\n".format(self.angle_min)
            msg += "MESH: min(vol)            = {:11.4e}\n".format(self.volume_min)
            msg += "MESH: max(vol)            = {:11.4e}\n".format(self.volume_max)
            msg += "MESH: boundary length     = {:11.4e}\n".format(self.boundary_length)
            msg += "MESH: min aspect ratio    = {:11.4e}\n".format(self.aspect_ratio.dat.data.min())
            msg += "MESH: max aspect ratio    = {:11.4e}\n".format(self.aspect_ratio.dat.data.max())
            msg += "MESH: min scaled Jacobian = {:11.4e}\n".format(self.scaled_jacobian.dat.data.min())
            msg += "MESH: max scaled Jacobian = {:11.4e}\n".format(self.scaled_jacobian.dat.data.max())
        msg += 40*"*" + "\n"
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


def make_consistent(mesh, h_mesh=None):
    """
    Make the coordinates associated with a Firedrake mesh and its underlying PETSc DMPlex
    use a consistent numbering.

    :kwarg h_mesh: uniformly refined mesh for if base mesh is not linear.
    """
    import firedrake.cython.dmcommon as dmplex

    if h_mesh is not None:
        assert len(mesh.coordinates.dat.data) == len(h_mesh.coordinates.dat.data)

    # Create section
    dim = mesh.topological_dimension()
    gdim = mesh.geometric_dimension()
    entity_dofs = np.zeros(dim+1, dtype=np.int32)
    entity_dofs[0] = gdim
    P0 = FunctionSpace(mesh, "DG", 0)  # NOQA
    coord_section = dmplex.create_section(h_mesh or mesh, entity_dofs)

    # Set plex coords to mesh coords
    plex = (h_mesh or mesh)._topology_dm
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

    # Get vertices, facets and centroids in patch
    all_vertices = set(range(*plex.getDepthStratum(0)))
    all_facets = set(range(*plex.getDepthStratum(1)))
    patch['vertices'] = set([])
    patch_facets = set([])
    for k in elements:
        closure = set(plex.getTransitiveClosure(k)[0])
        patch['elements'][k]['vertices'] = all_vertices.intersection(closure)
        patch['elements'][k]['facets'] = all_facets.intersection(closure)
        patch_facets = patch_facets.union(patch['elements'][k]['facets'])
        coords = [coordinates(v) for v in patch['elements'][k]['vertices']]
        patch['elements'][k]['centroid'] = np.sum(coords, axis=0)/(dim + 1)
        patch['vertices'] = patch['vertices'].union(set(patch['elements'][k]['vertices']))
    patch_facets.union(facets)
    patch['facets'] = {e: {} for e in patch_facets}

    # Get facet centroids
    if midfacets:
        for e in patch['facets']:
            closure = set(plex.getTransitiveClosure(e)[0])
            patch['facets'][e] = {'vertices': all_vertices.intersection(closure)}
            coords = [coordinates(v) for v in plex.getCone(e)]
            patch['facets'][e]['midfacet'] = np.sum(coords, axis=0)/dim

    return patch


def remesh(mesh, fname='mymesh', fpath='.', remove=True):
    """
    Remesh a tangled mesh using triangle.
    """
    import meshio
    import triangle

    plex = mesh._topology_dm
    if mesh.topological_dimension() != 2:
        raise NotImplementedError
    vertices = mesh.coordinates.dat.data
    cells = [('triangle', triangle.triangulate({'vertices': vertices})['triangles'])]
    points = np.array([[*point, 0.0] for point in vertices])
    cell_tag = lambda i: plex.getLabelValue("celltype", i)
    # bnd_tag = lambda i: plex.getLabelValue("boundary_faces", i)   # TODO
    cell_data = {
        "gmsh:physical": [np.array([cell_tag(i) for i in range(*plex.getHeightStratum(0))])],
    }
    newmesh = meshio.Mesh(points, cells, cell_data=cell_data)
    filename = os.path.join(fpath, fname + '.msh')
    meshio.gmsh.write(filename, newmesh, fmt_version="2.2", binary=False)
    outmesh = Mesh(filename)
    if remove:
        os.remove(filename)
    return outmesh


def quality(mesh, initial_signs=None):
    r"""
    Compute the scaled Jacobian for each element of a triangular mesh:

  ..math::
        Q(K) = \frac{\det(J_K)}{\|\mathbf e_1\|\,\|\mathbf e2\|},

    where element :math:`K` is defined by it edges of maximum length,
    :math:`\mathbf e_1` and :math:`\mathbf e_2`.

    If :math:`Q(K)<0` then we have an inverted element.

    :arg mesh: mesh to evaluate quality of.
    :kwarg initial_signs: (optional) signs of Jacobian determinant.
    """
    assert mesh.topological_dimension() == 2
    assert mesh.coordinates.ufl_element().cell() == triangle
    P0 = FunctionSpace(mesh, "DG", 0)
    P0_ten = TensorFunctionSpace(mesh, "DG", 0)
    J = interpolate(Jacobian(mesh), P0_ten)
    detJ = JacobianDeterminant(mesh)
    jacobian_sign = initial_signs or interpolate(sign(detJ), P0)
    edge1 = as_vector([J[0, 0], J[1, 0]])
    edge2 = as_vector([J[0, 1], J[1, 1]])
    edge3 = edge1 - edge2
    norm1 = sqrt(dot(edge1, edge1))
    norm2 = sqrt(dot(edge2, edge2))
    norm3 = sqrt(dot(edge3, edge3))
    prod1 = max_value(norm1*norm2, norm1*norm3)
    prod2 = max_value(norm2*norm3, norm2*norm1)
    prod3 = max_value(norm3*norm1, norm3*norm2)
    return interpolate(detJ/(max_value(max_value(prod1, prod2), prod3)*jacobian_sign), P0)


def aspect_ratio(mesh):
    """
    Compute the aspect ratio of each element of a triangular mesh.
    """
    assert mesh.topological_dimension() == 2
    assert mesh.coordinates.ufl_element().cell() == triangle
    P0 = FunctionSpace(mesh, "DG", 0)
    P0_ten = TensorFunctionSpace(mesh, "DG", 0)
    J = interpolate(Jacobian(mesh), P0_ten)
    edge1 = as_vector([J[0, 0], J[1, 0]])
    edge2 = as_vector([J[0, 1], J[1, 1]])
    edge3 = edge1 - edge2
    a = sqrt(dot(edge1, edge1))
    b = sqrt(dot(edge2, edge2))
    c = sqrt(dot(edge3, edge3))
    return interpolate(a*b*c/((a+b-c)*(b+c-a)*(c+a-b)), P0)


# FIXME: Inverted elements do not show! Tried making transparent but it didn't do anything.
def plot_quality(mesh, fig=None, axes=None, show_mesh=True, **kwargs):
    """
    Plot scaled Jacobian using a discretised scale:
      * green   : high quality elements (over 75%);
      * yellow  : medium quality elements (50 - 75%);
      * blue    : low quality elements (0 - 50%);
      * magenta : inverted elements (quality < 0).
    """
    from matplotlib.colors import ListedColormap
    import matplotlib.pyplot as plt

    # Compute mesh quality
    q = quality(mesh, **kwargs)

    # Create a modified colourscale
    cmap = plt.get_cmap('viridis', 201)
    newcolours = cmap(np.linspace(0, 1, 201))
    newcolours[:100, :] = np.array([1, 0, 1, 1])     # Magenta
    newcolours[100:150, :] = np.array([0, 1, 1, 1])  # Cyan
    newcolours[150:175, :] = np.array([1, 1, 0, 1])  # Yellow
    newcolours[175:, :] = np.array([0, 1, 0, 1])     # Green
    cmap = ListedColormap(newcolours)
    eps = 1.0e-06
    levels = np.linspace(-1-eps, 1+eps, 201)

    # Plot quality
    if fig is None or axes is None:
        fig, axes = plt.subplots()
    tc = tricontourf(q, axes=axes, cmap=cmap, levels=levels)
    cbar = fig.colorbar(tc, ax=axes)
    cbar.set_ticks([-1, 0, 0.5, 0.75, 1])
    cbar.set_ticklabels([r"-100\%", r"0\%", r"50\%", r"75\%", r"100\%"])
    if show_mesh:
        triplot(mesh, axes=axes, interior_kw={'linewidth': 0.1}, boundary_kw={'color': 'k'})
    axes.axis(False)
    return fig, axes


def plot_aspect_ratio(mesh, fig=None, axes=None, show_mesh=True, levels=10):
    """
    Plot aspect ratio of a triangular mesh.
    """
    import matplotlib.pyplot as plt

    ar = aspect_ratio(mesh)
    if fig is None or axes is None:
        fig, axes = plt.subplots()
    tc = tricontourf(ar, axes=axes, cmap='coolwarm', levels=levels)
    fig.colorbar(tc, ax=axes)
    if show_mesh:
        triplot(mesh, axes=axes, interior_kw={'linewidth': 0.1}, boundary_kw={'color': 'k'})
    axes.axis(False)
    return fig, axes
