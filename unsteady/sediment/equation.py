r"""
2D advection diffusion equation for sediment transport.

This can be either conservative :math:`q=HT` or non-conservative :math:`T` and allows
for a separate source and sink term. The equation reads

.. math::
    \frac{\partial S}{\partial t}
    + \nabla_h \cdot (\textbf{u} S)
    = \nabla_h \cdot (\mu_h \nabla_h S) + Source - (Sink S)
    :label: sediment_eq_2d

where :math:'S' is :math:'q' for conservative and :math:'T' for non-conservative,
:math:`\nabla_h` denotes horizontal gradient, :math:`\textbf{u}` are the horizontal
velocities, and :math:`\mu_h` denotes horizontal diffusivity.
"""
from __future__ import absolute_import
from thetis.utility import *
from thetis.equation import Equation
from thetis.tracer_eq_2d import HorizontalDiffusionTerm, TracerTerm
from thetis.sediment_eq_2d import SedimentSourceTerm, SedimentSinkTerm
from thetis.conservative_tracer_eq_2d import ConservativeHorizontalDiffusionTerm
from adapt_utils.unsteady.tracer.equation import HorizontalAdvectionTerm
from adapt_utils.unsteady.tracer.cons_equation import ConservativeHorizontalAdvectionTerm


class SedimentEquation2D(Equation):
    """
    2D sediment advection-diffusion equation: eq:`tracer_eq` or `conservative_tracer_eq`
    with sediment source and sink term
    """
    def __init__(self, function_space, depth,
                 use_lax_friedrichs=False,
                 sipg_parameter=Constant(10.0),
                 conservative=False):
        """
        :arg function_space: :class:`FunctionSpace` where the solution belongs
        :arg depth: :class: `DepthExpression` containing depth info
        :kwarg bool use_lax_friedrichs: whether to use Lax Friedrichs stabilisation
        :kwarg sipg_parameter: :class: `Constant` or :class: `Function` penalty parameter for SIPG
        :kwarg bool conservative: whether to use conservative tracer
        """
        super(SedimentEquation2D, self).__init__(function_space)
        args = (function_space, depth, use_lax_friedrichs, sipg_parameter)
        args_sediment = (function_space, depth, use_lax_friedrichs, sipg_parameter, conservative)
        if conservative:
            self.add_term(ConservativeHorizontalAdvectionTerm(*args), 'explicit')
            self.add_term(ConservativeHorizontalDiffusionTerm(*args), 'explicit')
        else:
            self.add_term(HorizontalAdvectionTerm(*args), 'explicit')
            self.add_term(HorizontalDiffusionTerm(*args), 'explicit')
        self.add_term(SedimentSourceTerm(*args_sediment), 'source')
        self.add_term(SedimentSinkTerm(*args_sediment), 'implicit')
