import thetis.equation as thetis_eq

from .mesh import anisotropic_cell_size


__all__ = ["Equation"]


class Equation(thetis_eq.Equation):
    """
    Modified version of `thetis.equation.Equation` which enables the use of an anisotropic
    cell size measure.
    """
    def __init__(self, *args, anisotropic=False, **kwargs):
        self.anisotropic = anisotropic
        super(Equation, self).__init__(*args, **kwargs)
        if anisotropic:
            self.cellsize = anisotropic_cell_size(self.mesh)

    def add_term(self, term, label):
        """
        Add :class:`term` to the equation as a :str:`label` type term.

        Also, pass over the chosen cell size measure and any stabilisation parameters.
        """
        super(Equation, self).add_term(term, label)
        key = term.__class__.__name__
        if self.anisotropic:
            self.terms[key].cellsize = self.cellsize
