# TODO: doc
from thetis.equation import *
from thetis.utility import *

from adapt_utils.tracer.equation import *


# --- Terms

class HorizontalAdvectionTerm3D(HorizontalAdvectionTerm):
    # TODO: doc
    def residual(self, solution, solution_old, fields, fields_old, bnd_conditions=None):
        raise NotImplementedError  # TODO


class HorizontalDiffusionTerm3D(HorizontalDiffusionTerm):
    # TODO: doc
    def residual(self, solution, solution_old, fields, fields_old, bnd_conditions=None):
        raise NotImplementedError  # TODO


class SourceTerm3D(SourceTerm):
    # TODO: doc
    def residual(self, solution, solution_old, fields, fields_old, bnd_conditions=None):
        raise NotImplementedError  # TODO


class ConservativeHorizontalAdvectionTerm3D(ConservativeHorizontalAdvectionTerm):
    # TODO: doc
    def residual(self, solution, solution_old, fields, fields_old, bnd_conditions=None):
        raise NotImplementedError  # TODO


class ConservativeHorizontalDiffusionTerm3D(HorizontalDiffusionTerm3D):
    """
    Copied from above.
    """


# --- Equations

class TracerEquation3D(Equation):
    # TODO: doc
    def __init__(self, function_space, depth, anisotropic=False):
        """
        :arg function_space: :class:`FunctionSpace` where the solution belongs
        :arg depth: :class: `DepthExpression` containing depth info
        """
        super(TracerEquation3D, self).__init__(function_space)
        if self.function_space.ufl_element().family() != 'Lagrange':
            raise NotImplementedError  # TODO
        args = (function_space, depth)
        kwargs = {
            'anisotropic': anisotropic,
        }
        self.add_term(HorizontalAdvectionTerm3D(*args, **kwargs), 'explicit')
        self.add_term(HorizontalDiffusionTerm3D(*args, **kwargs), 'explicit')
        self.add_term(SourceTerm3D(*args, **kwargs), 'source')


class ConservativeTracerEquation3D(Equation):
    # TODO: doc
    def __init__(self, function_space, depth, anisotropic=False):
        """
        :arg function_space: :class:`FunctionSpace` where the solution belongs
        :arg depth: :class: `DepthExpression` containing depth info
        """
        super(ConservativeTracerEquation3D, self).__init__(function_space)
        if self.function_space.ufl_element().family() != 'Lagrange':
            raise NotImplementedError  # TODO
        args = (function_space, depth)
        kwargs = {
            'anisotropic': anisotropic,
        }
        self.add_term(ConservativeHorizontalAdvectionTerm(*args, **kwargs), 'explicit')
        self.add_term(ConservativeHorizontalDiffusionTerm(*args, **kwargs), 'explicit')
        self.add_term(SourceTerm(*args, **kwargs), 'source')
