from __future__ import absolute_import
from thetis.equation import *
from thetis.utility import *
from thetis.conservative_tracer_eq_2d import *


# TODO: Extend to consider CG case

class ConservativeTracerTerm(TracerTerm):
    """
    Generic depth-integrated tracer term that provides commonly used members and mapping for
    boundary functions.
    """
    def __init__(self, function_space, depth,
                 use_lax_friedrichs=False,
                 sipg_parameter=Constant(10.0)):
        """
        :arg function_space: :class:`FunctionSpace` where the solution belongs
        :arg depth: :class: `DepthExpression` containing depth info
        :kwarg bool use_lax_friedrichs: whether to use Lax Friedrichs stabilisation
        :kwarg sipg_parameter: :class: `Constant` or :class: `Function` penalty parameter for SIPG
        """

        super().__init__(function_space, depth,
                         use_lax_friedrichs=use_lax_friedrichs,
                         sipg_parameter=sipg_parameter)

    # TODO: at the moment this is the same as TracerTerm, but we probably want to overload its
    # get_bnd_functions method


class ConservativeHorizontalAdvectionTerm(ConservativeTracerTerm):
    r"""
    Advection of tracer term, :math:`\nabla \cdot \bar{\textbf{u}} \nabla q`

    The weak form is

    .. math::
        \int_\Omega \boldsymbol{\psi} \nabla\cdot \bar{\textbf{u}} \nabla q  dx
        = - \int_\Omega \left(\nabla_h \boldsymbol{\psi})\right) \cdot \bar{\textbf{u}} \cdot q dx
        + \int_\Gamma \text{avg}(q\bar{\textbf{u}}\cdot\textbf{n}) \cdot \text{jump}(\boldsymbol{\psi}) dS

    where the right hand side has been integrated by parts;
    :math:`\textbf{n}` is the unit normal of
    the element interfaces, and :math:`\text{jump}` and :math:`\text{avg}` denote the
    jump and average operators across the interface.
    """
    def residual(self, solution, solution_old, fields, fields_old, bnd_conditions=None):
        if fields_old.get('uv_2d') is None:
            return 0
        elev = fields_old['elev_2d']
        self.corr_factor = fields_old.get('tracer_advective_velocity_factor')

        uv = self.corr_factor * fields_old['uv_2d']
        uv_p1 = fields_old.get('uv_p1')
        uv_mag = fields_old.get('uv_mag')

        lax_friedrichs_factor = fields_old.get('lax_friedrichs_tracer_scaling_factor')

        f = 0
        f += -(Dx(self.test, 0) * uv[0] * solution
               + Dx(self.test, 1) * uv[1] * solution) * self.dx

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
            flux_up = solution('-')*uv('-')*s + solution('+')*uv('+')*(1-s)

            f += (flux_up[0] * jump(self.test, self.normal[0])
                  + flux_up[1] * jump(self.test, self.normal[1])) * self.dS
            # Lax-Friedrichs stabilization
            if self.use_lax_friedrichs:
                if uv_p1 is not None:
                    gamma = 0.5*abs((avg(uv_p1)[0]*self.normal('-')[0]
                                     + avg(uv_p1)[1]*self.normal('-')[1]))*lax_friedrichs_factor
                elif uv_mag is not None:
                    gamma = 0.5*avg(uv_mag)*lax_friedrichs_factor
                else:
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
                        flux_up = c_in*uv*s + c_ext*uv_ext*(1-s)
                        f += (flux_up[0]*self.normal[0]
                              + flux_up[1]*self.normal[1])*self.test*ds_bnd
                    else:
                        f += c_in * (uv[0]*self.normal[0]
                                     + uv[1]*self.normal[1])*self.test*ds_bnd

        return -f


class ConservativeTracerEquation2D(Equation):
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
        super(ConservativeTracerEquation2D, self).__init__(function_space)
        args = (function_space, depth, use_lax_friedrichs, sipg_parameter)
        self.add_term(ConservativeHorizontalAdvectionTerm(*args), 'explicit')
        self.add_term(ConservativeHorizontalDiffusionTerm(*args), 'explicit')
        self.add_term(ConservativeSourceTerm(*args), 'source')
