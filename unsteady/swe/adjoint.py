from __future__ import absolute_import
from thetis.equation import *
from thetis.shallowwater_eq import ShallowWaterTerm
from thetis.utility import *

import warnings


__all__ = ["AdjointShallowWaterEquations"]


g_grav = physical_constants['g_grav']


class AdjointShallowWaterTerm(ShallowWaterTerm):

    def get_bnd_functions(self, zeta_in, z_in, bnd_id, bnd_conditions):
        """
        In the forward, we permit only free-slip conditions for the velocity and Dirichlet
        conditions for the elevation. Suppose these are imposed on Γ₁ and Γ₂, which are not
        necessarily disjoint. Then the adjoint has a free-slip condition for the adjoint
        velocity on the complement of Γ₂ and Dirichlet conditions for the elevation on the
        complement of Γ₁.
        """
        # bnd_len = self.boundary_len[bnd_id]
        funcs = bnd_conditions.get(bnd_id)
        if 'elev' in funcs and 'un' in funcs:  # Γ₁ ∪ Γ₂
            zeta_ext = zeta_in  # assume symmetry
            z_ext = z_in  # assume symmetry
            warnings.warn("#### TODO: BCs not valid for viscous or nonlinear equations")  # TODO
        elif 'elev' not in funcs:  # ∂Ω \ Γ₂
            zeta_ext = zeta_in  # assume symmetry
            z_ext = Constant(0.0)*self.normal
            warnings.warn("#### TODO: BCs not valid for viscous or nonlinear equations")  # TODO
        elif 'un' not in funcs:  # ∂Ω \ Γ₁
            zeta_ext = Constant(0.0)
            z_ext = z_in  # assume symmetry
            warnings.warn("#### TODO: BCs not valid for viscous or nonlinear equations")  # TODO
        else:
            raise Exception('Unsupported bnd type: {:}'.format(funcs.keys()))
        return zeta_ext, z_ext


class AdjointShallowWaterMomentumTerm(AdjointShallowWaterTerm):
    """
    Generic term in the first component of the adjoint shallow water equation that provides
    commonly used members and mapping for boundary functions.
    """
    def __init__(self, z_test, z_space, zeta_space,
                 depth, options=None):
        super(AdjointShallowWaterMomentumTerm, self).__init__(z_space, depth, options)

        self.options = options

        self.z_test = z_test
        self.z_space = z_space
        self.zeta_space = zeta_space

        self.z_continuity = element_continuity(self.z_space.ufl_element()).horizontal
        self.zeta_is_dg = element_continuity(self.zeta_space.ufl_element()).horizontal == 'dg'


class AdjointShallowWaterContinuityTerm(AdjointShallowWaterTerm):
    """
    Generic term in the second component of the adjoint shallow equation that provides commonly
    used members and mapping for boundary functions.
    """
    def __init__(self, zeta_test, zeta_space, z_space,
                 depth, options=None):
        super(AdjointShallowWaterContinuityTerm, self).__init__(zeta_space, depth, options)

        self.zeta_test = zeta_test
        self.zeta_space = zeta_space
        self.z_space = z_space

        self.z_continuity = element_continuity(self.z_space.ufl_element()).horizontal
        self.zeta_is_dg = element_continuity(self.zeta_space.ufl_element()).horizontal == 'dg'


class ExternalPressureGradientTerm(AdjointShallowWaterContinuityTerm):
    r"""
  ..math::

        g \nabla \cdot \mathbf u^*
    """
    def residual(self, z, zeta, z_old, zeta_old, fields, fields_old, bnd_conditions=None):
        z_by_parts = True  # So we can enforce free-slip conditions

        if z_by_parts:
            f = -g_grav*inner(grad(self.zeta_test), z)*self.dx
            if self.z_continuity in ['dg', 'hdiv']:
                # f += g_grav * self.zeta_test * inner(z, self.normal) * self.dS  # TODO
                raise NotImplementedError
            for bnd_marker in self.boundary_markers:
                funcs = bnd_conditions.get(bnd_marker)
                ds_bnd = ds(int(bnd_marker), degree=self.quad_degree)
                if funcs is not None:
                    # TODO: Riemann solutions
                    zeta_ext, z_ext = self.get_bnd_functions(zeta, z, bnd_marker, bnd_conditions)
                    f += g_grav*self.zeta_test*inner(z_ext, self.normal)*ds_bnd
                else:
                    raise NotImplementedError
        else:
            f = g_grav*self.zeta_test*nabla_div(z)*self.dx

        return -f


class HUDivTermMomentum(AdjointShallowWaterMomentumTerm):
    r"""
    In the nonlinear case,

  ..math::

        H \nabla \eta^*.

    In the linear case,

  ..math::

        b \nabla \eta^*.
    """
    def residual(self, z, zeta, z_old, zeta_old, fields, fields_old, bnd_conditions=None):
        if self.options.use_nonlinear_equations:
            eta = fields.get('elev_2d')  # TODO
            total_h = self.depth.get_total_depth(eta)
        else:
            total_h = self.depth.get_total_depth(zeta_old)

        f = 0
        if self.zeta_is_dg:
            f += -inner(zeta, div(total_h*self.z_test))*self.dx
            # f += total_h * zeta * inner(self.z_test, self.normal) * self.dS  # TODO
            raise NotImplementedError
        else:
            f += inner(self.z_test, total_h*grad(zeta))*self.dx
        return -f


class HUDivTermContinuity(AdjointShallowWaterContinuityTerm):
    r"""
    This term only arises in the nonlinear case.

  ..math::

        \mathbf u \cdot \nabla \eta^*
    """
    def residual(self, z, zeta, z_old, zeta_old, fields, fields_old, bnd_conditions=None):
        f = 0
        if not self.options.use_nonlinear_equations:
            return f
        uv = fields.get('uv_2d')  # TODO
        if self.zeta_is_dg:
            f += -div(self.zeta_test*uv)*zeta*self.dx
            # f += zeta * self.zeta_test * inner(uv, self.normal) * self.dS  # TODO
            raise NotImplementedError
        else:
            f += inner(grad(zeta), self.zeta_test*uv)*self.dx
        return -f


class HorizontalAdvectionTerm(AdjointShallowWaterMomentumTerm):
    r"""
  ..math::

        - (\nabla\mathbf u)^T \mathbf u^*
        + (\nabla \cdot \mathbf u) \mathbf u^*
        + \mathbf u \cdot \nabla \mathbf u^*
    """
    def residual(self, z, zeta, z_old, zeta_old, fields, fields_old, bnd_conditions=None):
        f = 0
        if not self.options.use_nonlinear_equations:
            return f

        raise NotImplementedError  # TODO


class HorizontalViscosityTerm(AdjointShallowWaterMomentumTerm):
    r"""

  ..math::

        \nabla \cdot (\nu \nabla \mathbf u^*)
    """
    def residual(self, z, zeta, z_old, zeta_old, fields, fields_old, bnd_conditions=None):
        # eta = fields.get('elev_2d')
        # total_h = self.depth.get_total_depth(eta)

        nu = fields_old.get('viscosity_h')
        if nu is None:
            return 0

        raise NotImplementedError  # TODO


class CoriolisTerm(AdjointShallowWaterMomentumTerm):
    r"""

  ..math::

        f \widehat{\mathbf z} \times \mathbf u^*
    """
    def residual(self, z, zeta, z_old, zeta_old, fields, fields_old, bnd_conditions=None):
        coriolis = fields_old.get('coriolis')
        f = 0
        if coriolis is not None:
            f += coriolis*(-z[1]*self.z_test[0] + z[0]*self.z_test[1])*self.dx
        return -f


class QuadraticDragTermMomentum(AdjointShallowWaterMomentumTerm):
    r"""

  ..math::

        -\frac{C_d}H \left(
            \|\mathbf u\| \mathbf u
            + \frac{\mathbf u^* \cdot \mathbf u}{\|\mathbf u\|}\mathbf u
        \right)
    """
    def residual(self, z, zeta, z_old, zeta_old, fields, fields_old, bnd_conditions=None):
        total_h = self.depth.get_total_depth(zeta_old)
        manning_drag_coefficient = fields_old.get('manning_drag_coefficient')
        C_D = fields_old.get('quadratic_drag_coefficient')
        f = 0
        if manning_drag_coefficient is not None:
            if C_D is not None:
                raise Exception('Cannot set both dimensionless and Manning drag parameter')
            C_D = g_grav*manning_drag_coefficient**2/total_h**(1./3.)

        uv = fields.get('uv_2d')  # TODO
        if uv is None:
            raise Exception('Adjoint equation does not have access to forward solution velocity')
        if C_D is not None:
            unorm = sqrt(dot(uv, uv) + self.options.norm_smoother**2)
            f += -C_D*unorm*inner(self.z_test, z)*self.dx
            f += -C_D*inner(self.z_test, uv)*inner(z, uv)/unorm*self.dx
        return -f


class QuadraticDragTermContinuity(AdjointShallowWaterContinuityTerm):
    r"""

  ..math::

        \frac{\widetilde{C_d}}{H^2} \|\mathbf u\| \mathbf u \cdot \mathbf u^*,

    where :math:`\widetilde{C_d}` is given by :math:`\frac43 C_d` in the case of Manning friction
    and :math:`C_d` otherwise.
    """
    def residual(self, z, zeta, z_old, zeta_old, fields, fields_old, bnd_conditions=None):
        total_h = self.depth.get_total_depth(zeta_old)
        manning_drag_coefficient = fields_old.get('manning_drag_coefficient')
        C_D = fields_old.get('quadratic_drag_coefficient')
        f = 0
        if manning_drag_coefficient is not None:
            if C_D is not None:
                raise Exception('Cannot set both dimensionless and Manning drag parameter')
            C_D = 4/3*g_grav*manning_drag_coefficient**2/total_h**(1./3.)

        uv = fields.get('uv_2d')  # TODO
        if uv is None:
            raise Exception('Adjoint equation does not have access to forward solution velocity')
        if C_D is not None:
            unorm = sqrt(dot(uv, uv) + self.options.norm_smoother**2)
            f += C_D*unorm*inner(z, uv)*self.zeta_test/total_h**2*self.dx
        return -f


class LinearDragTerm(AdjointShallowWaterMomentumTerm):
    r"""

  ..math::

        C_d \mathbf u^*
    """
    def residual(self, z, zeta, z_old, zeta_old, fields, fields_old, bnd_conditions=None):
        linear_drag_coefficient = fields_old.get('linear_drag_coefficient')
        f = 0
        if linear_drag_coefficient is not None:
            bottom_fri = linear_drag_coefficient*inner(self.z_test, z)*self.dx
            f += bottom_fri
        return -f


class TurbineDragTermMomentum(AdjointShallowWaterMomentumTerm):
    def residual(self, z, zeta, z_old, zeta_old, fields, fields_old, bnd_conditions=None):
        # eta = fields.get('elev_2d')  # TODO
        # total_h = self.depth.get_total_depth(eta)
        f = 0
        for subdomain_id, farm_options in self.options.tidal_turbine_farms.items():
            raise NotImplementedError  # TODO
        return -f


class TurbineDragTermContinuity(AdjointShallowWaterContinuityTerm):
    def residual(self, z, zeta, z_old, zeta_old, fields, fields_old, bnd_conditions=None):
        # eta = fields.get('elev_2d')  # TODO
        # total_h = self.depth.get_total_depth(eta)
        f = 0
        for subdomain_id, farm_options in self.options.tidal_turbine_farms.items():
            raise NotImplementedError  # TODO
        return -f


# NOTE: Used for adjoint RHS
class MomentumSourceTerm(AdjointShallowWaterMomentumTerm):
    def residual(self, z, zeta, z_old, zeta_old, fields, fields_old, bnd_conditions=None):
        f = 0
        momentum_source = fields_old.get('momentum_source')

        if momentum_source is not None:
            f += inner(momentum_source, self.z_test)*self.dx
        return 0


# NOTE: Used for adjoint RHS
class ContinuitySourceTerm(AdjointShallowWaterContinuityTerm):
    def residual(self, z, zeta, z_old, zeta_old, fields, fields_old, bnd_conditions=None):
        f = 0
        volume_source = fields_old.get('volume_source')

        if volume_source is not None:
            f += inner(volume_source, self.zeta_test)*self.dx
        return f


# TODO
class BathymetryDisplacementMassTerm(AdjointShallowWaterContinuityTerm):
    def residual(self, solution):
        if isinstance(solution, list):
            z, zeta = solution
        else:
            z, zeta = split(solution)
        f = inner(self.depth.wd_bathymetry_displacement(zeta), self.zeta_test)*self.dx
        return -f


class BaseAdjointShallowWaterEquation(Equation):
    def __init__(self, function_space,
                 depth, options):
        super(BaseAdjointShallowWaterEquation, self).__init__(function_space)
        self.depth = depth
        self.options = options

    def add_momentum_terms(self, *args):
        self.add_term(HUDivTermMomentum(*args), 'implicit')
        self.add_term(HorizontalAdvectionTerm(*args), 'explicit')
        self.add_term(HorizontalViscosityTerm(*args), 'explicit')
        self.add_term(CoriolisTerm(*args), 'explicit')
        # self.add_term(WindStressTerm(*args), 'source')
        self.add_term(QuadraticDragTermMomentum(*args), 'explicit')
        self.add_term(LinearDragTerm(*args), 'explicit')
        # self.add_term(BottomDrag3DTerm(*args), 'source')
        self.add_term(TurbineDragTermMomentum(*args), 'implicit')
        self.add_term(MomentumSourceTerm(*args), 'source')

    def add_continuity_terms(self, *args):
        self.add_term(HUDivTermContinuity(*args), 'implicit')
        self.add_term(ExternalPressureGradientTerm(*args), 'implicit')
        self.add_term(ContinuitySourceTerm(*args), 'source')
        self.add_term(QuadraticDragTermContinuity(*args), 'explicit')
        self.add_term(TurbineDragTermContinuity(*args), 'implicit')

    def residual_z_zeta(self, label, z, zeta, z_old, zeta_old, fields, fields_old, bnd_conditions):
        f = 0
        for term in self.select_terms(label):
            f += term.residual(z, zeta, z_old, zeta_old, fields, fields_old, bnd_conditions)
        return f


class AdjointShallowWaterEquations(BaseAdjointShallowWaterEquation):
    def __init__(self, function_space, depth, options):
        """
        :arg function_space: Mixed function space where the solution belongs
        :arg depth: :class: `DepthExpression` containing depth info
        :arg options: :class:`.AttrDict` object containing all circulation model options
        """
        super(AdjointShallowWaterEquations, self).__init__(function_space, depth, options)

        if options.get('use_wetting_and_drying'):
            raise NotImplementedError  # TODO

        z_test, zeta_test = TestFunctions(function_space)
        z_space, zeta_space = function_space.split()

        self.add_momentum_terms(z_test, z_space, zeta_space, depth, options)

        self.add_continuity_terms(zeta_test, zeta_space, z_space, depth, options)
        # self.bathymetry_displacement_mass_term = BathymetryDisplacementMassTerm(
        #     zeta_test, zeta_space, z_space, depth, options)  # TODO

    def mass_term(self, solution):
        f = super(AdjointShallowWaterEquations, self).mass_term(solution)
        # f += -self.bathymetry_displacement_mass_term.residual(solution)  # TODO
        return f

    def residual(self, label, solution, solution_old, fields, fields_old, bnd_conditions):
        if isinstance(solution, list):
            z, zeta = solution
        else:
            z, zeta = split(solution)
        z_old, zeta_old = split(solution_old)
        return self.residual_z_zeta(label, z, zeta, z_old, zeta_old, fields, fields_old, bnd_conditions)
