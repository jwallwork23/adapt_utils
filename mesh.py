from thetis import *


__all__ = ["MeshStats"]


class MeshStats(object):
    def __init__(self, op, mesh=None):
        self._op = op
        self._mesh = mesh or op.default_mesh
        self._P0 = FunctionSpace(mesh, "DG", 0)
        self.get_element_sizes()

    def get_element_sizes(self):
        self.dx = interpolate(CellDiameter(self._mesh), self._P0)
        self.dx_min = self.dx.vector().gather().min()
        self.dx_max = self.dx.vector().gather().max()
