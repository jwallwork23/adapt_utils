r"""
2D advection diffusion equation for sediment transport.

This can be either conservative :math:`q=HT` or non-conservative :math:`T` and allows
for a separate source and sink term. The equation reads

.. math::
    \frac{\partial S}{\partial t}
    + \nabla_h \cdot (\textbf{u} S)
    = \nabla_h \cdot (\mu_h \nabla_h S) + F_{source} - (F_{sink} S)
    :label: sediment_eq_2d

where :math:'S' is :math:'q' for conservative and :math:'T' for non-conservative,
:math:`\nabla_h` denotes horizontal gradient, :math:`\textbf{u}` are the horizontal
velocities, and :math:`\mu_h` denotes horizontal diffusivity.

**********************************************************************************************
*  NOTE: This file is based on the Thetis project (https://thetisproject.org)               *
**********************************************************************************************
"""
from __future__ import absolute_import
from thetis.utility import *
from thetis.equation import Equation
from thetis.tracer_eq_2d import *
from thetis.sediment_eq_2d import SedimentTerm, ConservativeSedimentAdvectionTerm, SedimentAdvectionTerm, SedimentDiffusionTerm

class SedimentErosionTerm(SedimentTerm):
    r"""
    Generic source term

    The weak form reads

    .. math::
        F_{source} = \int_\Omega \sigma \phi dx

    where :math:`\sigma` is a user defined scalar :class:`Function`.

    """
    def residual(self, solution, solution_old, fields, fields_old, bnd_conditions=None):
        source = self.sediment_model.get_erosion_term(self.conservative)
        f = inner(source, self.test)*self.dx
        return f


class SedimentDepositionTerm(SedimentTerm):
    r"""
    Liner sink term

    The weak form reads

    .. math::
        F_{sink} = - \int_\Omega \sigma solution \phi dx

    where :math:`\sigma` is a user defined scalar :class:`Function`.

    """
    def residual(self, solution, solution_old, fields, fields_old, bnd_conditions=None):
        sink = self.sediment_model.get_deposition_coefficient()
        f = inner(-sink*solution, self.test)*self.dx
        return f


class SedimentEquation2D(Equation):
    """
    2D sediment advection-diffusion equation: eq:`tracer_eq` or `conservative_tracer_eq`
    with sediment source and sink term
    """
    def __init__(self, function_space, depth, options, sediment_model,
                 conservative=False):
        """
        :arg function_space: :class:`FunctionSpace` where the solution belongs
        :arg depth: :class: `DepthExpression` containing depth info
        :kwarg bool conservative: whether to use conservative tracer
        """
        super(SedimentEquation2D, self).__init__(function_space)
        args = (function_space, depth, options, sediment_model, conservative)
        if conservative:
            self.add_term(ConservativeSedimentAdvectionTerm(*args), 'explicit')
        else:
            self.add_term(SedimentAdvectionTerm(*args), 'explicit')
        self.add_term(SedimentDiffusionTerm(*args), 'explicit')
        self.add_term(SedimentErosionTerm(*args), 'source')
        self.add_term(SedimentDepositionTerm(*args), 'implicit')
