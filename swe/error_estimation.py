"""
Goal-oriented error indicators for the shallow water hydrodynamics model. See [Wallwork et al. 2020b]
for details on the formulation.

[Wallwork et al. 2020b] J. G. Wallwork, N. Barral, S. C. Kramer, D. A. Ham, M. D. Piggott,
    "Goal-oriented error estimation and mesh adaptation for shallow water modelling" (2020),
    Springer Nature Applied Sciences, volume 2, pp.1053--1063, DOI: 10.1007/s42452-020-2745-9,
    URL: https://rdcu.be/b35wZ.
"""
from __future__ import absolute_import
from thetis.utility import *
from thetis.shallowwater_eq import ShallowWaterTerm
from ..error_estimation import GOErrorEstimatorTerm, GOErrorEstimator
from .utils import speed


__all__ = ['ShallowWaterGOErrorEstimator']


g_grav = physical_constants['g_grav']
rho_0 = physical_constants['rho0']


class ShallowWaterGOErrorEstimatorTerm(GOErrorEstimatorTerm, ShallowWaterTerm):
    """
    Generic :class:`GOErrorEstimatorTerm` term in a goal-oriented error estimator for the shallow
    water model.
    """
    def __init__(self, function_space, depth=None, options=None):
        """
        :arg function_space: :class:`FunctionSpace` where the solution belongs
        :kwarg depth: DepthExpression for the domain
        :kwarg options: :class:`ModelOptions2d` parameter object
        """
        ShallowWaterTerm.__init__(self, function_space, depth, options)
        GOErrorEstimatorTerm.__init__(self, function_space.mesh())

    def element_residual(self, *args):
        return 0

    def inter_element_flux(self, *args):
        return 0

    def boundary_flux(self, *args, **kwargs):
        return 0


class ShallowWaterGOErrorEstimatorMomentumTerm(ShallowWaterGOErrorEstimatorTerm):
    """
    Generic :class:`ShallowWaterGOErrorEstimatorTerm` term that provides commonly used members and
    mapping for boundary functions.
    """
    def __init__(self, u_space, eta_space, depth=None, options=None):
        super(ShallowWaterGOErrorEstimatorMomentumTerm, self).__init__(u_space, depth, options)

        self.options = options

        self.u_space = u_space
        self.eta_space = eta_space

        self.u_continuity = element_continuity(self.u_space.ufl_element()).horizontal
        self.eta_is_dg = element_continuity(self.eta_space.ufl_element()).horizontal == 'dg'


class ShallowWaterGOErrorEstimatorContinuityTerm(ShallowWaterGOErrorEstimatorTerm):
    """
    Generic :class:`ShallowWaterGOErrorEstimatorTerm` term that provides commonly used members and
    mapping for boundary functions.
    """
    def __init__(self, eta_space, u_space, depth=None, options=None):
        super(ShallowWaterGOErrorEstimatorContinuityTerm, self).__init__(eta_space, depth, options)

        self.eta_space = eta_space
        self.u_space = u_space

        self.u_continuity = element_continuity(self.u_space.ufl_element()).horizontal
        self.eta_is_dg = element_continuity(self.eta_space.ufl_element()).horizontal == 'dg'


class ExternalPressureGradientGOErrorEstimatorTerm(ShallowWaterGOErrorEstimatorMomentumTerm):
    """
    :class:`ShallowWaterGOErrorEstimatorTerm` object associated with the
    :class:`ExternalPressureGradientTerm` term of the shallow water model.
    """
    def element_residual(self, solution, solution_old, arg, arg_old, fields, fields_old):
        uv, eta = split(solution)
        z, zeta = split(arg)

        grad_eta_by_parts = self.eta_is_dg

        if grad_eta_by_parts:
            return self.p0test*g_grav*nabla_div(z)*eta*self.dx
        else:
            return -self.p0test*g_grav*inner(z, grad(eta))*self.dx

    def inter_element_flux(self, solution, solution_old, arg, arg_old, fields, fields_old):
        uv, eta = split(solution)
        uv_old, eta_old = split(solution_old)
        z, zeta = split(arg)

        total_h = self.depth.get_total_depth(eta_old)
        head = eta
        grad_eta_by_parts = self.eta_is_dg

        flux_terms = 0
        if grad_eta_by_parts:

            # Terms arising from DG discretisation
            if uv is not None:
                head_star = avg(head) + sqrt(avg(total_h)/g_grav)*jump(uv, self.normal)
            else:
                head_star = avg(head)
            loc = -self.p0test*g_grav*dot(z, self.normal)
            flux_terms += head_star*(loc('+') + loc('-'))*self.dS

            # Term arising from integration by parts
            loc = self.p0test*g_grav*eta*dot(z, self.normal)
            flux_terms += (loc('+') + loc('-'))*self.dS

        return flux_terms

    def boundary_flux(self, solution, solution_old, arg, arg_old, fields, fields_old, bnd_conditions):
        uv, eta = split(solution)
        uv_old, eta_old = split(solution_old)
        z, zeta = split(arg)

        total_h = self.depth.get_total_depth(eta_old)
        head = eta
        grad_eta_by_parts = self.eta_is_dg

        flux_terms = 0
        if grad_eta_by_parts:

            # Term arising from integration by parts
            loc = self.p0test*g_grav*eta*dot(z, self.normal)
            flux_terms += loc*ds

            # Terms arising from boundary conditions
            for bnd_marker in self.boundary_markers:
                funcs = bnd_conditions.get(bnd_marker)
                ds_bnd = ds(int(bnd_marker), degree=self.quad_degree)
                if funcs is not None:
                    eta_ext, uv_ext = self.get_bnd_functions(head, uv, bnd_marker, bnd_conditions)
                    # Compute linear riemann solution with eta, eta_ext, uv, uv_ext
                    un_jump = inner(uv - uv_ext, self.normal)
                    eta_rie = 0.5*(head + eta_ext) + sqrt(total_h/g_grav)*un_jump
                    flux_terms += -self.p0test*g_grav*eta_rie*dot(z, self.normal)*ds_bnd
                if funcs is None or 'symm' in funcs:
                    # assume land boundary
                    # impermeability implies external un=0
                    un_jump = inner(uv, self.normal)
                    head_rie = head + sqrt(total_h/g_grav)*un_jump
                    flux_terms += -self.p0test*g_grav*head_rie*dot(z, self.normal)*ds_bnd
        else:
            # Terms arising from boundary conditions
            for bnd_marker in self.boundary_markers:
                funcs = bnd_conditions.get(bnd_marker)
                ds_bnd = ds(int(bnd_marker), degree=self.quad_degree)
                if funcs is not None and self.options.get('element_family') != 'cg-cg':
                    eta_ext, uv_ext = self.get_bnd_functions(head, uv, bnd_marker, bnd_conditions)
                    # Compute linear riemann solution with eta, eta_ext, uv, uv_ext
                    un_jump = inner(uv - uv_ext, self.normal)
                    eta_rie = 0.5*(head + eta_ext) + sqrt(total_h/g_grav)*un_jump
                    flux_terms += -self.p0test*g_grav*(eta_rie-head)*dot(z, self.normal)*ds_bnd

        return flux_terms


class HUDivGOErrorEstimatorTerm(ShallowWaterGOErrorEstimatorContinuityTerm):
    """
    :class:`ShallowWaterGOErrorEstimatorTerm` object associated with the :class:`HUDivTerm` term of
    the shallow water model.
    """
    def element_residual(self, solution, solution_old, arg, arg_old, fields, fields_old):
        uv, eta = split(solution)
        uv_old, eta_old = split(solution_old)
        z, zeta = split(arg)
        total_h = self.depth.get_total_depth(eta_old)

        return -self.p0test*zeta*div(total_h*uv)*self.dx

    def inter_element_flux(self, solution, solution_old, arg, arg_old, fields, fields_old):
        uv, eta = split(solution)
        uv_old, eta_old = split(solution_old)
        z, zeta = split(arg)

        total_h = self.depth.get_total_depth(eta_old)

        # Term arising from integration by parts
        loc = self.p0test*zeta*dot(total_h*uv, self.normal)
        flux_terms = (loc('+') + loc('-'))*self.dS

        # Terms arising from DG discretisation
        if self.eta_is_dg:
            h = avg(total_h)
            uv_rie = avg(uv) + sqrt(g_grav/h)*jump(eta, self.normal)
            hu_star = h*uv_rie
            loc = -self.p0test*zeta*self.normal
            flux_terms += inner(loc('+') + loc('-'), hu_star)*self.dS

        return flux_terms

    def boundary_flux(self, solution, solution_old, arg, arg_old, fields, fields_old, bnd_conditions):
        uv, eta = split(solution)
        uv_old, eta_old = split(solution_old)
        z, zeta = split(arg)

        total_h = self.depth.get_total_depth(eta_old)

        # Term arising from integration by parts
        flux_terms = self.p0test*zeta*dot(total_h*uv, self.normal)*ds

        # Terms arising from boundary conditions
        for bnd_marker in self.boundary_markers:
            funcs = bnd_conditions.get(bnd_marker)
            ds_bnd = ds(int(bnd_marker), degree=self.quad_degree)
            if funcs is not None and self.options.get('element_family') != 'cg-cg':
                eta_ext, uv_ext = self.get_bnd_functions(eta, uv, bnd_marker, bnd_conditions)
                eta_ext_old, uv_ext_old = self.get_bnd_functions(eta_old, uv_old, bnd_marker, bnd_conditions)
                # Compute linear riemann solution with eta, eta_ext, uv, uv_ext
                total_h_ext = self.depth.get_total_depth(eta_ext_old)
                h_av = 0.5*(total_h + total_h_ext)
                eta_jump = eta - eta_ext
                un_rie = 0.5*inner(uv + uv_ext, self.normal) + sqrt(g_grav/h_av)*eta_jump
                un_jump = inner(uv_old - uv_ext_old, self.normal)
                eta_rie = 0.5*(eta_old + eta_ext_old) + sqrt(h_av/g_grav)*un_jump
                h_rie = self.depth.bathymetry_2d + eta_rie
                flux_terms += -self.p0test*h_rie*un_rie*zeta*ds_bnd

        return flux_terms


class HorizontalAdvectionGOErrorEstimatorTerm(ShallowWaterGOErrorEstimatorMomentumTerm):
    """
    :class:`ShallowWaterGOErrorEstimatorTerm` object associated with the
    :class:`HorizontalAdvectionTerm` term of the shallow water model.
    """
    def element_residual(self, solution, solution_old, arg, arg_old, fields, fields_old):
        if not self.options.use_nonlinear_equations:
            return 0

        if self.options.get('element_family') == 'cg-cg':
            raise NotImplementedError  # TODO

        uv, eta = split(solution)
        uv_old, eta_old = split(solution_old)
        z, zeta = split(arg)

        return -self.p0test*inner(z, dot(uv_old, nabla_grad(uv)))*self.dx

    def inter_element_flux(self, solution, solution_old, arg, arg_old, fields, fields_old):
        if not self.options.use_nonlinear_equations:
            return 0

        if self.options.get('element_family') == 'cg-cg':
            raise NotImplementedError  # TODO

        uv, eta = split(solution)
        uv_old, eta_old = split(solution_old)
        z, zeta = split(arg)

        if self.u_continuity in ['dg', 'hdiv']:

            # Terms arising from DG discretisation
            un_av = dot(avg(uv_old), self.normal('-'))
            uv_up = avg(uv)
            loc = -self.p0test*z[0]
            flux_terms = jump(uv[0], self.normal[0])*dot(uv_up[0], loc('+') + loc('-'))*self.dS
            flux_terms += jump(uv[1], self.normal[1])*dot(uv_up[0], loc('+') + loc('-'))*self.dS
            loc = -self.p0test*z[1]
            flux_terms += jump(uv[0], self.normal[0])*dot(uv_up[1], loc('+') + loc('-'))*self.dS
            flux_terms += jump(uv[1], self.normal[1])*dot(uv_up[1], loc('+') + loc('-'))*self.dS
            if self.options.use_lax_friedrichs_velocity:
                uv_lax_friedrichs = fields_old.get('lax_friedrichs_velocity_scaling_factor')
                gamma = 0.5*abs(un_av)*uv_lax_friedrichs
                local_jump = -self.p0test('+')*z('+') + self.p0test('-')*z('-')
                flux_terms += gamma*dot(local_jump, jump(uv))*dS

        # Term arising from integration by parts
        loc = self.p0test*inner(dot(outer(uv, z), uv), self.normal)
        flux_terms += (loc('+') + loc('-'))*self.dS

        return flux_terms

    def boundary_flux(self, solution, solution_old, arg, arg_old, fields, fields_old, bnd_conditions):
        if not self.options.use_nonlinear_equations:
            return 0

        if self.options.get('element_family') == 'cg-cg':
            raise NotImplementedError  # TODO

        uv, eta = split(solution)
        uv_old, eta_old = split(solution_old)
        z, zeta = split(arg)

        # Term arising from integration by parts
        flux_terms = self.p0test*inner(dot(outer(uv, z), uv), self.normal)*ds

        # Terms arising from boundary conditions
        for bnd_marker in self.boundary_markers:
            funcs = bnd_conditions.get(bnd_marker)
            ds_bnd = ds(int(bnd_marker), degree=self.quad_degree)
            un_av = dot(avg(uv_old), self.normal('-'))
            if funcs is not None:
                eta_ext, uv_ext = self.get_bnd_functions(eta, uv, bnd_marker, bnd_conditions)
                eta_ext_old, uv_ext_old = self.get_bnd_functions(eta_old, uv_old, bnd_marker, bnd_conditions)
                # Compute linear riemann solution with eta, eta_ext, uv, uv_ext
                eta_jump = eta_old - eta_ext_old
                total_h = self.depth.get_total_depth(eta_old)
                un_rie = 0.5*inner(uv_old + uv_ext_old, self.normal) + sqrt(g_grav/total_h)*eta_jump
                uv_av = 0.5*(uv_ext + uv)
                flux_terms += -self.p0test*(uv_av[0]*z[0]*un_rie + uv_av[1]*z[1]*un_rie)*ds_bnd

            if self.options.use_lax_friedrichs_velocity:
                uv_lax_friedrichs = fields_old.get('lax_friedrichs_velocity_scaling_factor')
                gamma = 0.5*abs(un_av)*uv_lax_friedrichs
                if funcs is None:
                    # impose impermeability with mirror velocity
                    n = self.normal
                    uv_ext = uv - 2*dot(uv, n)*n
                    gamma = 0.5*abs(dot(uv_old, n))*uv_lax_friedrichs
                    flux_terms += -self.p0test*gamma*dot(z, uv-uv_ext)*ds_bnd

        return flux_terms


class HorizontalViscosityGOErrorEstimatorTerm(ShallowWaterGOErrorEstimatorMomentumTerm):
    """
    :class:`ShallowWaterGOErrorEstimatorTerm` object associated with the
    :class:`HorizontalViscosityTerm` term of the shallow water model.
    """
    def element_residual(self, solution, solution_old, arg, arg_old, fields, fields_old):
        nu = fields_old.get('viscosity_h')
        if nu is None:
            return 0
        uv, eta = split(solution)
        uv_old, eta_old = split(solution_old)
        z, zeta = split(arg)
        total_h = self.depth.get_total_depth(eta_old)

        if self.options.use_grad_div_viscosity_term:
            stress = 2.0*nu*sym(grad(uv))
        else:
            stress = nu*grad(uv)

        f = self.p0test*inner(z, div(stress))*self.dx
        if self.options.use_grad_depth_viscosity_term:
            f += self.p0test*inner(z, dot(grad(total_h)/total_h))*self.dx

        return f

    def inter_element_flux(self, solution, solution_old, arg, arg_old, fields, fields_old):
        nu = fields_old.get('viscosity_h')
        if nu is None:
            return 0

        flux_terms = 0
        if self.u_continuity not in ['dg', 'hdiv']:
            return flux_terms

        uv, eta = split(solution)
        uv_old, eta_old = split(solution_old)
        z, zeta = split(arg)

        h = self.cellsize
        n = self.normal

        if self.options.use_grad_div_viscosity_term:
            stress = 2.0*nu*sym(grad(uv))
            stress_jump = 2.0*avg(nu)*sym(tensor_jump(uv, n))
        else:
            stress = nu*grad(uv)
            stress_jump = avg(nu)*tensor_jump(uv, n)

        # Terms arising from DG discretisation
        alpha = self.options.sipg_parameter
        assert alpha is not None
        loc = self.p0test*outer(z, n)
        flux_terms += -avg(alpha/h)*inner(loc('+') + loc('-'), stress_jump)*self.dS
        flux_terms += inner(loc('+') + loc('-'), avg(stress))*self.dS
        loc = self.p0test*grad(z)
        flux_terms += 0.5*inner(loc('+') + loc('-'), stress_jump)*self.dS

        # Term arising from integration by parts
        loc = -self.p0test*inner(dot(z, stress), n)
        flux_terms += (loc('+') + loc('-'))*self.dS

        return flux_terms

    def boundary_flux(self, solution, solution_old, arg, arg_old, fields, fields_old, bnd_conditions):
        nu = fields_old.get('viscosity_h')
        if nu is None:
            return 0

        flux_terms = 0
        if self.u_continuity not in ['dg', 'hdiv']:
            return flux_terms

        uv, eta = split(solution)
        uv_old, eta_old = split(solution_old)
        z, zeta = split(arg)

        h = self.cellsize
        n = self.normal

        if self.options.use_grad_div_viscosity_term:
            stress = 2.0*nu*sym(grad(uv))
            stress_jump = 2.0*avg(nu)*sym(tensor_jump(uv, n))
        else:
            stress = nu*grad(uv)
            stress_jump = avg(nu)*tensor_jump(uv, n)

        # Term arising from integration by parts
        flux_terms += -self.p0test*inner(dot(z, stress), n)*ds

        # Terms arising from boundary conditions
        alpha = self.options.sipg_parameter
        assert alpha is not None
        for bnd_marker in self.boundary_markers:
            funcs = bnd_conditions.get(bnd_marker)
            ds_bnd = ds(int(bnd_marker), degree=self.quad_degree)
            if funcs is not None:
                if 'un' in funcs:
                    delta_uv = (dot(uv, n) - funcs['un'])*n
                else:
                    eta_ext, uv_ext = self.get_bnd_functions(eta, uv, bnd_marker, bnd_conditions)
                    if uv_ext is uv:
                        continue
                    delta_uv = uv - uv_ext

                if self.options.use_grad_div_viscosity_term:
                    stress_jump = 2.0*nu*sym(outer(delta_uv, n))
                else:
                    stress_jump = nu*outer(delta_uv, n)

                flux_terms += -self.p0test*alpha/h*inner(outer(z, n), stress_jump)*ds_bnd
                flux_terms += self.p0test*inner(grad(z), stress_jump)*ds_bnd
                flux_terms += self.p0test*inner(outer(z, n), stress)*ds_bnd

        return flux_terms


class CoriolisGOErrorEstimatorTerm(ShallowWaterGOErrorEstimatorMomentumTerm):
    """
    :class:`ShallowWaterGOErrorEstimatorTerm` object associated with the :class:`CoriolisTerm` term
    of the shallow water model.
    """
    def element_residual(self, solution, solution_old, arg, arg_old, fields, fields_old):
        uv, eta = split(solution)
        z, zeta = split(arg)
        coriolis = fields_old.get('coriolis')

        f = 0
        if coriolis is not None:
            f += self.p0test*coriolis*(-uv[1]*z[0] + uv[0]*z[1])*self.dx

        return -f


class WindStressGOErrorEstimatorTerm(ShallowWaterGOErrorEstimatorMomentumTerm):
    """
    :class:`ShallowWaterGOErrorEstimatorTerm` object associated with the :class:`WindStressTerm` term
    of the shallow water model.
    """
    def element_residual(self, solution, solution_old, arg, arg_old, fields, fields_old):
        uv, eta = split(solution)
        uv_old, eta_old = split(solution_old)
        z, zeta = split(arg)

        wind_stress = fields_old.get('wind_stress')
        total_h = self.depth.get_total_depth(eta_old)
        f = 0
        if wind_stress is not None:
            f += self.p0test*dot(wind_stress, z)/total_h/rho_0*self.dx
        return f


class AtmosphericPressureGOErrorEstimatorTerm(ShallowWaterGOErrorEstimatorMomentumTerm):
    """
    :class:`ShallowWaterGOErrorEstimatorTerm` object associated with the
    :class:`AtmosphericPressureTerm` term of the shallow water model.
    """
    def element_residual(self, solution, solution_old, arg, arg_old, fields, fields_old):
        uv, eta = split(solution)
        uv_old, eta_old = split(solution_old)
        z, zeta = split(arg)

        atmospheric_pressure = fields_old.get('atmospheric_pressure')
        f = 0
        if atmospheric_pressure is not None:
            f += self.p0test*dot(grad(atmospheric_pressure), z)/rho_0*self.dx
        return -f


class QuadraticDragGOErrorEstimatorTerm(ShallowWaterGOErrorEstimatorMomentumTerm):
    """
    :class:`ShallowWaterGOErrorEstimatorTerm` object associated with the :class:`QuadraticDragTerm`
    term of the shallow water model.
    """
    def element_residual(self, solution, solution_old, arg, arg_old, fields, fields_old):
        uv, eta = split(solution)
        uv_old, eta_old = split(solution_old)
        z, zeta = split(arg)
        total_h = self.depth.get_total_depth(eta_old)
        manning_drag_coefficient = fields_old.get('manning_drag_coefficient')
        C_D = fields_old.get('quadratic_drag_coefficient')

        f = 0
        if manning_drag_coefficient is not None:
            if C_D is not None:
                raise Exception('Cannot set both dimensionless and Manning drag parameter')
            C_D = g_grav * manning_drag_coefficient**2 / total_h**(1./3.)

        if C_D is not None:
            unorm = speed(solution_old, smoother=self.options.norm_smoother)
            f += self.p0test*C_D*unorm*inner(z, uv)/total_h*self.dx

        return -f


class LinearDragGOErrorEstimatorTerm(ShallowWaterGOErrorEstimatorMomentumTerm):
    """
    :class:`ShallowWaterGOErrorEstimatorTerm` object associated with the :class:`LinearDragTerm`
    term of the shallow water model.
    """
    def element_residual(self, solution, solution_old, arg, arg_old, fields, fields_old):
        uv, eta = split(solution)
        z, zeta = split(arg)

        linear_drag_coefficient = fields_old.get('linear_drag_coefficient')
        f = 0
        if linear_drag_coefficient is not None:
            f += self.p0test*linear_drag_coefficient*inner(z, uv)*self.dx
        return -f


class BottomDrag3DGOErrorEstimatorTerm(ShallowWaterGOErrorEstimatorMomentumTerm):
    """
    :class:`ShallowWaterGOErrorEstimatorTerm` object associated with the :class:`BottomDrag3DTerm`
    term of the shallow water model.
    """
    def element_residual(self, solution, solution_old, arg, arg_old, fields, fields_old):
        uv, eta = split(solution)
        uv_old, eta_old = split(solution_old)
        z, zeta = split(arg)

        total_h = self.depth.get_total_depth(eta_old)
        bottom_drag = fields_old.get('bottom_drag')
        uv_bottom = fields_old.get('uv_bottom')
        f = 0
        if bottom_drag is not None and uv_bottom is not None:
            uvb_mag = sqrt(uv_bottom[0]**2 + uv_bottom[1]**2)
            stress = bottom_drag*uvb_mag*uv_bottom/total_h
            f += self.p0test*dot(stress, z)*self.dx
        return -f


class TurbineDragGOErrorEstimatorTerm(ShallowWaterGOErrorEstimatorMomentumTerm):
    """
    :class:`ShallowWaterGOErrorEstimatorTerm` object associated with the :class:`TurbineDragTerm`
    term of the shallow water model.
    """
    def element_residual(self, solution, solution_old, arg, arg_old, fields, fields_old):
        uv, eta = split(solution)
        uv_old, eta_old = split(solution_old)
        z, zeta = split(arg)
        total_h = self.depth.get_total_depth(eta_old)

        f = 0
        for subdomain_id, farm_options in self.options.tidal_turbine_farms.items():
            density = farm_options.turbine_density
            C_T = farm_options.turbine_options.thrust_coefficient
            A_T = pi * (farm_options.turbine_options.diameter/2.)**2
            C_D = (C_T * A_T * density)/2.
            unorm = speed(solution_old)
            f += self.p0test*C_D*unorm*inner(z, uv)/total_h*self.dx(subdomain_id)

        return -f


class MomentumSourceGOErrorEstimatorTerm(ShallowWaterGOErrorEstimatorMomentumTerm):
    """
    :class:`ShallowWaterGOErrorEstimatorTerm` object associated with the :class:`MomentumSourceTerm`
    term of the shallow water model.
    """
    def element_residual(self, solution, solution_old, arg, arg_old, fields, fields_old):
        z, zeta = split(arg)

        f = 0
        momentum_source = fields_old.get('momentum_source')

        if momentum_source is not None:
            f += self.p0test*inner(momentum_source, z)*self.dx
        return f


class ContinuitySourceGOErrorEstimatorTerm(ShallowWaterGOErrorEstimatorContinuityTerm):
    """
    :class:`ShallowWaterGOErrorEstimatorTerm` object associated with the
    :class:`ContinuitySourceTerm` term of the shallow water model.
    """
    def element_residual(self, solution, solution_old, arg, arg_old, fields, fields_old):
        z, zeta = split(arg)

        f = 0
        volume_source = fields_old.get('volume_source')

        if volume_source is not None:
            f += self.p0test*inner(volume_source, zeta)*self.dx
        return f


class BathymetryDisplacementGOErrorEstimatorTerm(ShallowWaterGOErrorEstimatorContinuityTerm):
    """
    :class:`ShallowWaterGOErrorEstimatorTerm` object associated with the
    :class:`BathymetryDisplacementTerm` term of the shallow water model.
    """
    def element_residual(self, solution, arg):
        uv, eta = split(solution)
        z, zeta = split(arg)

        f = 0
        if self.options.use_wetting_and_drying:
            f += self.p0test*inner(self.depth.wd_bathymetry_displacement(eta), zeta)*self.dx
        return -f


class ShallowWaterGOErrorEstimator(GOErrorEstimator):
    """
    :class:`GOErrorEstimator` for the shallow water model.
    """
    def __init__(self, function_space, depth, options, anisotropic=False):
        super(ShallowWaterGOErrorEstimator, self).__init__(function_space, anisotropic=anisotropic)
        self.depth = depth
        self.options = options

        u_space, eta_space = function_space.split()
        self.add_momentum_terms(u_space, eta_space, depth, options)
        self.add_continuity_terms(eta_space, u_space, depth, options)

    def add_momentum_terms(self, *args):
        self.add_term(ExternalPressureGradientGOErrorEstimatorTerm(*args), 'implicit')
        self.add_term(HorizontalAdvectionGOErrorEstimatorTerm(*args), 'explicit')
        self.add_term(HorizontalViscosityGOErrorEstimatorTerm(*args), 'explicit')
        self.add_term(CoriolisGOErrorEstimatorTerm(*args), 'explicit')
        self.add_term(WindStressGOErrorEstimatorTerm(*args), 'source')
        self.add_term(AtmosphericPressureGOErrorEstimatorTerm(*args), 'source')
        self.add_term(QuadraticDragGOErrorEstimatorTerm(*args), 'explicit')
        self.add_term(LinearDragGOErrorEstimatorTerm(*args), 'explicit')
        self.add_term(BottomDrag3DGOErrorEstimatorTerm(*args), 'source')
        self.add_term(TurbineDragGOErrorEstimatorTerm(*args), 'implicit')
        self.add_term(MomentumSourceGOErrorEstimatorTerm(*args), 'source')

    def add_continuity_terms(self, *args):
        self.add_term(HUDivGOErrorEstimatorTerm(*args), 'implicit')
        self.add_term(ContinuitySourceGOErrorEstimatorTerm(*args), 'source')

    def setup_strong_residual(self, label, solution, solution_old, fields, fields_old):
        P0P0 = VectorFunctionSpace(self.mesh, "DG", 0)*self.P0
        self._strong_residual = Function(P0P0, name="Strong residual for shallow water equations")
        residual_u, residual_eta = self._strong_residual.split()
        residual_u.rename("Strong residual for momentum equation")
        residual_eta.rename("Strong residual for continuity equation")

        # Strong residual for u-component of momentum equation
        adj_u = Function(P0P0)
        adj_u1, adj_u2 = adj_u.split()
        adj_u1.interpolate(as_vector([1.0, 0.0]))
        args = (solution, solution_old, adj_u, adj_u, fields, fields_old)
        self.strong_residual_terms_u = 0
        for term in self.select_terms(label):
            self.strong_residual_terms_u += term.element_residual(*args)

        # Strong residual for v-component of momentum equation
        adj_v = Function(P0P0)
        adj_v1, adj_v2 = adj_v.split()
        adj_v1.interpolate(as_vector([0.0, 1.0]))
        args = (solution, solution_old, adj_v, adj_v, fields, fields_old)
        self.strong_residual_terms_v = 0
        for term in self.select_terms(label):
            self.strong_residual_terms_v += term.element_residual(*args)

        # Strong residual for continuity equation
        adj_eta = Function(P0P0)
        adj_eta1, adj_eta2 = adj_eta.split()
        adj_eta2.assign(1.0)
        args = (solution, solution_old, adj_eta, adj_eta, fields, fields_old)
        self.strong_residual_terms_eta = 0
        for term in self.select_terms(label):
            self.strong_residual_terms_eta += term.element_residual(*args)

    @property
    def strong_residual(self):
        """
        Evaluate strong residual of shallow water equations.
        """
        residual_u, residual_eta = self._strong_residual.split()
        residual_u.interpolate(as_vector([assemble(self.strong_residual_terms_u),
                                          assemble(self.strong_residual_terms_v)]))
        residual_eta.interpolate(assemble(self.strong_residual_terms_eta))
        return self._strong_residual
