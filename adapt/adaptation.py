from firedrake import *


__all__ = ["AdaptiveMesh", "iso_P2"]


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


# TODO: This will become redundant once `AdaptiveMesh` is used everywhere
def iso_P2(mesh):
    r"""
    Uniformly refine a mesh (in each canonical direction) using an iso-P2 refinement. That is, nodes
    of a quadratic element on the initial mesh become vertices of the new mesh.
    """
    return MeshHierarchy(mesh, 1).__getitem__(1)
