from firedrake import dot, dx, grad, inner
import thetis.equation as thetis_eq

from .mesh import anisotropic_cell_size


__all__ = ["Equation"]


class Equation(thetis_eq.Equation):
    """
    Modified version of `thetis.equation.Equation` which enables the use of an anisotropic
    cell size measure.
    """
    def __init__(self, *args, anisotropic=False, **kwargs):
        super(Equation, self).__init__(*args, **kwargs)
        self.anisotropic = anisotropic
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
        if hasattr(self, 'stabilisation'):
            self.terms[key].stabilisation = self.stabilisation
        else:
            self.terms[key].stabilisation = None
        if hasattr(self, 'tau'):
            self.terms[key].tau = self.tau

    def mass_term(self, solution, velocity=None):
        test = self.test
        if self.stabilisation == 'supg':  # TODO: Hook up time-dependent SUPG
            assert velocity is not None
            assert hasattr(self, 'tau')
            test = test + dot(velocity, grad(solution))
        return inner(solution, test)*dx
