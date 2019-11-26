from firedrake import *
from firedrake.petsc import PETSc
from firedrake.mesh import MeshGeometry  # FIXME: won't work for adjoint case

from adapt_utils.options import DefaultOptions


__all__ = ["iso_P2", "multi_adapt"]


# FIXME: won't work for adjoint case
class AdaptMesh(MeshGeometry):
    """
    Subclassed mesh object which adds some extra features.
    """
    def __init__(self, coordinates):
        super(self, AdaptMesh).__init__(self, coordinates)

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


def multi_adapt(metric, op=DefaultOptions()):
    r"""
    Adapt mesh multiple times, by repeatedly projecting the metric into the new space.

    This should be done more than once, but at most four times. The more steps applied, the larger
    the errors attributed to the projection.
    """
    for i in range(op.num_adapt):
        mesh = metric.function_space().mesh()
        newmesh = adapt(mesh, metric)
        if i < op.num_adapt-1:
            V = TensorFunctionSpace(newmesh, "CG", 1)
            metric = project(metric, V)
    return newmesh
