r"""
Exner equation

2D conservation of mass equation describing bed evolution due to sediment transport

The equation reads

.. math::
    \frac{\partial z_b}{\partial t} + (morfac/(1-p)) \nabla_h \cdot (Q_b)
    = (morfac/(1-p)) H ((Sink S) - Source)
    :label: exner_eq

where :math:'z_b' is the bedlevel, :math:'S' is :math:'q=HT' for conservative
and :math:'T' for non-conservative, :math:`\nabla_h` denotes horizontal gradient,
:math:'morfac' is the morphological scale factor, :math:'p' is the porosity and
:math:'Q_b' is the bedload transport vector

"""

from __future__ import absolute_import
from thetis.equation import Equation
from thetis.utility import *
from thetis.exner_eq import ExnerTerm, ExnerBedloadTerm, ExnerSedimentSlideTerm


class ExnerSourceTerm(ExnerTerm):
    r"""
    Source term accounting for suspended sediment transport

    The weak form reads

    .. math::
        F_s = \int_\Omega (\sigma - sediment * \phi) * depth \psi dx

    where :math:`\sigma` is a user defined source scalar field :class:`Function`
    and :math:`\phi` is a user defined source scalar field :class:`Function`.

    """
    def residual(self, solution, solution_old, fields, fields_old, bnd_conditions=None):
        f = 0
        sediment = fields.get('sediment')
        source = self.sediment_model.get_erosion_term()
        sink = self.sediment_model.get_deposition_coefficient()
        morfac = fields.get('morfac')
        porosity = fields.get('porosity')

        fac = Constant(morfac/(1.0-porosity))
        H = self.depth.get_total_depth(fields_old['elev_2d'])

        source_dep = source*H

        if self.depth_integrated_sediment:
            sink_dep = sink
        else:
            sink_dep = sink*H

        f = inner(fac*(source_dep-sediment*sink_dep), self.test)*self.dx

        return f

class ExnerEquation(Equation):
    """
    Exner equation

    2D conservation of mass equation describing bed evolution due to sediment transport
    """
    def __init__(self, function_space, depth, sediment_model, depth_integrated_sediment):
        """
        :arg function_space: :class:`FunctionSpace` where the solution belongs
        :arg depth: :class: `DepthExpression` containing depth info
        :arg sediment_model: :class: `SedimentModel` containing sediment info
        :kwarg bool depth_integrated_sediment: whether to use conservative tracer
        """
        super().__init__(function_space)
        if sediment_model is None:
            raise ValueError('To use the exner equation must define a sediment model')
        args = (function_space, depth, sediment_model, depth_integrated_sediment)
        if sediment_model.solve_suspended_sediment:
            self.add_term(ExnerSourceTerm(*args), 'source')
        if sediment_model.use_bedload:
            self.add_term(ExnerBedloadTerm(*args), 'implicit')
        if sediment_model.use_sediment_slide:
            self.add_term(ExnerSedimentSlideTerm(*args), 'implicit')
