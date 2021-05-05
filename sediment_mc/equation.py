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
*  NOTE: This file is based on the Thetis project (https://thetisproject.org) and contains   *
*        some copied code.                                                                   *
**********************************************************************************************
"""
from __future__ import absolute_import
from thetis.utility import *
from thetis.tracer_eq_2d import *
from thetis.conservative_tracer_eq_2d import ConservativeHorizontalAdvectionTerm


class SedimentTerm(TracerTerm):
    """
    Generic sediment term that provides commonly used members.
    """
    def __init__(self, function_space, depth, options, sediment_model):
        """
        :arg function_space: :class:`FunctionSpace` where the solution belongs
        :arg depth: :class: `DepthExpression` containing depth info
        """
        super(SedimentTerm, self).__init__(function_space, depth, options)
        self.conservative = options.use_tracer_conservative_form
        self.sediment_model = sediment_model

class ConservativeSedimentAdvectionTerm(SedimentTerm, ConservativeHorizontalAdvectionTerm):
    """
    Advection term for sediment equation
    Same as :class:`ConservativeHorizontalAdvectionTerm` but allows for equilibrium boundary condition
    through get_bnd_conditions() inherited from :class:`SedimentTerm`."""
    pass


class SedimentAdvectionTerm(SedimentTerm, HorizontalAdvectionTerm):
    """
    Advection term for sediment equation
    Same as :class:`HorizontalAdvectionTerm` but allows for equilibrium boundary condition
    through get_bnd_conditions() inherited from :class:`SedimentTerm`."""
    pass


class SedimentDiffusionTerm(SedimentTerm, HorizontalDiffusionTerm):
    """
    Diffusion term for sediment equation
    Same as :class:`HorizontalDiffusionTerm` but allows for equilibrium boundary condition
    through get_bnd_conditions() inherited from :class:`SedimentTerm`."""
    pass

class SedimentErosionTerm(SedimentTerm):
    r"""
    Generic source term

    The weak form reads

    .. math::
        F_{source} = \int_\Omega \sigma \phi dx

    where :math:`\sigma` is a user defined scalar :class:`Function`.

    """
    def residual(self, solution, solution_old, fields, fields_old, bnd_conditions=None):
        ero = self.sediment_model.get_erosion_term()
        if not self.conservative:
            elev = fields['elev_2d']
            ero = ero / self.depth.get_total_depth(elev)
        f = self.test * ero * self.dx
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
        depo = self.sediment_model.get_deposition_coefficient()
        elev = fields['elev_2d']
        H = self.depth.get_total_depth(elev)
        f = -self.test * depo/H * solution * self.dx
        return -f


class SedimentEquation2D(TracerEquation2D):
    """
    2D sediment advection-diffusion equation: eq:`tracer_eq` or `conservative_tracer_eq`
    with sediment source and sink term
    """
    def __init__(self, function_space, depth, options, sediment_model, conservative):
        """
        :arg function_space: :class:`FunctionSpace` where the solution belongs
        :arg depth: :class: `DepthExpression` containing depth info
        """
        super(SedimentEquation2D, self).__init__(function_space, depth, options, None)
        args = (function_space, depth, options, sediment_model)
        if conservative:
            self.add_term(ConservativeSedimentAdvectionTerm(*args), 'explicit')
        else:
            self.add_term(SedimentAdvectionTerm(*args), 'explicit')
        self.add_term(SedimentDiffusionTerm(*args), 'explicit')
        self.add_term(SedimentErosionTerm(*args), 'source')
        self.add_term(SedimentDepositionTerm(*args), 'implicit')
