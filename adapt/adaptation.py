from firedrake import *
from firedrake.petsc import PETSc
from firedrake.mesh import MeshGeometry  # FIXME: won't work for adjoint case

from adapt_utils.options import DefaultOptions


__all__ = ["iso_P2"]


# FIXME: Won't work for adjoint case
# NOTE:  However, will work after Mid-December 'pyadjoint sprint'
class AdaptMesh(MeshGeometry):
    """
    Subclassed mesh object which adds some extra features.
    """
    def __init__(self, coordinates, levels=0):
        super(self, AdaptMesh).__init__(self, coordinates)
        if levels > 0:
            self.hierarchy = MeshHierarchy(self, levels)

    def save_plex(self, filename):
        viewer = PETSc.Viewer().createHDF5(filename, 'r')
        viewer(self._plex)

    def load_plex(self, filename):
        newplex = PETSc.DMPlex().create()
        newplex.createFromFile(filename)
        self.__init__(Mesh(newplex).coordinates)


def iso_P2(mesh):
    r"""
    Uniformly refine a mesh (in each canonical direction) using an iso-P2 refinement. That is, nodes
    of a quadratic element on the initial mesh become vertices of the new mesh.
    """
    return MeshHierarchy(mesh, 1).__getitem__(1)
