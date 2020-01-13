from firedrake import *


__all__ = ["AdaptiveMesh"]


class AdaptiveMesh():
    """
    Wrapper which adds extra features to mesh.
    """
    def __init__(self, mesh, levels=0):
        """
        `AdaptMesh` object is initialised as the basis of a `MeshHierarchy`.
        """
        self.levels = levels
        self.hierarchy = MeshHierarchy(mesh, levels)
        self.mesh = self.hierarchy[0]
        if levels > 0:
            self.refined_mesh = self.hierarchy[1]

    def save_plex(self, filename):
        """
        Save mesh in DMPlex format.
        """
        viewer = PETSc.Viewer().createHDF5(filename, 'r')
        viewer(self.mesh._plex)

    def load_plex(self, filename):
        """
        Load mesh from DMPlex format. The `MeshHierarchy` is reinstated.
        """
        newplex = PETSc.DMPlex().create()
        newplex.createFromFile(filename)
        self.__init__(Mesh(newplex), levels=self.levels)

    def adapt(self, metric):
        """
        Adapt mesh using a specified metric. The `MeshHierarchy` is reinstated.
        """
        self.__init__(adapt(self.mesh, metric), levels=self.levels)

    def copy(self):
        return AdaptiveMesh(Mesh(Function(self.mesh.coordinates)), levels=self.levels)

    def get_edge_lengths(self):
        """
        Compute edge lengths, stored in a HDiv trace field.

         NOTE: The plus sign is arbitrary and could equally well be chosen as minus.
        """
        HDivTrace = FunctionSpace(self.mesh, "HDiv Trace", 0)
        v, u = TestFunction(HDivTrace), TrialFunction(HDivTrace)
        self.edge_lengths = Function(HDivTrace, name="Edge lengths")
        mass_term = v('+')*u('+')*dS + v*u*ds
        rhs = v('+')*FacetArea(self.mesh)*dS + v*FacetArea(self.mesh)*ds
        solve(mass_term == rhs, self.edge_lengths)
