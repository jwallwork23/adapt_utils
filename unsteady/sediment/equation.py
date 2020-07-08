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
from thetis.conservative_tracer_eq_2d import ConservativeHorizontalAdvectionTerm, ConservativeHorizontalDiffusionTerm
from 

__all__ = [
    'SedimentEquation2D',
    'SedimentTerm',
    'SedimentSourceTerm',
    'SedimentSinkTerm',
]


class SedimentTerm(TracerTerm):
    """
    Generic sediment term that provides commonly used members.
    """
    def __init__(self, function_space, depth,
                 use_lax_friedrichs=True, sipg_parameter=Constant(10.0), conservative=False):
        """
        :arg function_space: :class:`FunctionSpace` where the solution belongs
        :arg depth: :class: `DepthExpression` containing depth info
        :kwarg bool use_lax_friedrichs: whether to use Lax Friedrichs stabilisation
        :kwarg sipg_parameter: :class: `Constant` or :class: `Function` penalty parameter for SIPG
        :kwarg bool conservative: whether to use conservative tracer
        """
        super(SedimentTerm, self).__init__(function_space, depth)
        self.conservative = conservative


class HorizontalAdvectionTerm(SedimentTerm):
    def residual(self, solution, solution_old, fields, fields_old, bnd_conditions=None):
        if fields_old.get('uv_2d') is None:
            return 0
        elev = fields_old['elev_2d']
        self.corr_factor = fields_old.get('tracer_advective_velocity_factor')

        uv = self.corr_factor * fields_old['uv_2d']
        # FIXME is this an option?
        lax_friedrichs_factor = fields_old.get('lax_friedrichs_tracer_scaling_factor')

        f = 0
        f += -(Dx(uv[0] * self.test, 0) * solution
               + Dx(uv[1] * self.test, 1) * solution) * self.dx

        mesh_velocity = fields_old.get('mesh_velocity')  # TODO: Make more robust
        if mesh_velocity is not None:
            f += (Dx(mesh_velocity[0] * self.test, 0) * solution
                  + Dx(mesh_velocity[1] * self.test, 1) * solution) * self.dx

        if self.horizontal_dg:
            # add interface term
            uv_av = avg(uv)
            un_av = (uv_av[0]*self.normal('-')[0]
                     + uv_av[1]*self.normal('-')[1])
            s = 0.5*(sign(un_av) + 1.0)
            c_up = solution('-')*s + solution('+')*(1-s)

            f += c_up*(jump(self.test, uv[0] * self.normal[0])
                       + jump(self.test, uv[1] * self.normal[1])) * self.dS
            # Lax-Friedrichs stabilization
            if self.use_lax_friedrichs:
                gamma = 0.5*abs(un_av)*lax_friedrichs_factor
                f += gamma*dot(jump(self.test), jump(solution))*self.dS
            if bnd_conditions is not None:
                for bnd_marker in self.boundary_markers:
                    funcs = bnd_conditions.get(bnd_marker)
                    ds_bnd = ds(int(bnd_marker), degree=self.quad_degree)
                    c_in = solution
                    if funcs is not None and 'value' in funcs:
                        c_ext, uv_ext, eta_ext = self.get_bnd_functions(c_in, uv, elev, bnd_marker, bnd_conditions)
                        uv_av = 0.5*(uv + uv_ext)
                        un_av = self.normal[0]*uv_av[0] + self.normal[1]*uv_av[1]
                        s = 0.5*(sign(un_av) + 1.0)
                        c_up = c_in*s + c_ext*(1-s)
                        f += c_up*(uv_av[0]*self.normal[0]
                                   + uv_av[1]*self.normal[1])*self.test*ds_bnd
                    else:
                        f += c_in * (uv[0]*self.normal[0]
                                     + uv[1]*self.normal[1])*self.test*ds_bnd

        return -f


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
        self.add_term(SedimentSinkTerm(*args_sediment), 'source')
