from thetis import *


__all__ = ["MeshStats"]


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
        if self.dim == 2:
            self.angles_min = get_minimum_angles_2d(self._mesh)
            self.angle_min = self.angles_min.vector().gather().min()
            self.get_element_volumes()
            self.boundary_lengths = compute_boundary_length(self._mesh)
            self.boundary_length = sum(self.boundary_lengths[tag] for tag in self.boundary_markers)
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

    def get_element_sizes(self):
        self.dx = Function(self._P0, name="")
        self.dx.interpolate(CellDiameter(self._mesh))
        self.dx_min = self.dx.vector().gather().min()
        self.dx_max = self.dx.vector().gather().max()

    def get_element_volumes(self):
        if self.dim == 3:
            raise NotImplementedError  # TODO
        self.volume = Function(self._P0, name="Element volume")
        get_horizontal_elem_size_2d(self.volume)
        self.volume_min = self.volume.vector().gather().min()
        self.volume_max = self.volume.vector().gather().max()
