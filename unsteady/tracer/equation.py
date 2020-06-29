from __future__ import absolute_import
from thetis.equation import *
from thetis.utility import *
from thetis.tracer_eq_2d import *


# TODO: Extend Thetis tracer_2d model to consider CG case
# TODO: Conservative form


class TracerEquation2D(Equation):
    """Copied here to hook up modified terms."""
    def __init__(self, function_space, depth,
                 use_lax_friedrichs=False,
                 sipg_parameter=Constant(10.0)):
        """
        :arg function_space: :class:`FunctionSpace` where the solution belongs
        :arg depth: :class: `DepthExpression` containing depth info
        :kwarg bool use_lax_friedrichs: whether to use Lax Friedrichs stabilisation
        :kwarg sipg_parameter: :class: `Constant` or :class: `Function` penalty parameter for SIPG
        """
        super(TracerEquation2D, self).__init__(function_space)

        args = (function_space, depth, use_lax_friedrichs, sipg_parameter)
        self.add_term(HorizontalAdvectionTerm(*args), 'explicit')
        self.add_term(HorizontalDiffusionTerm(*args), 'explicit')
        self.add_term(SourceTerm(*args), 'source')
