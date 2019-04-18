from firedrake import *


__all__ = ["iso_P2"]


def iso_P2(mesh):
    r"""
    Uniformly refine a mesh (in each canonical direction) using an iso-P2 refinement. That is, nodes
    of a quadratic element on the initial mesh become vertices of the new mesh.
    """
    return MeshHierarchy(mesh, 1).__getitem__(1)
