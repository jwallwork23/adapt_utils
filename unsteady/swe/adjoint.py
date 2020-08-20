r"""
Continuous adjoint shallow water equations for some quantity of interest :math:`J`. The momentum equation
is given by

..math::

    -\frac{\partial\mathbf u^*}{\partial t}
      + (\nabla\mathbf u)^T \mathbf u^*
      - (\nabla\cdot\mathbf u) \mathbf u^*
      - \mathbf u\cdot\nabla \mathbf u^*
      - f \widehat{\mathbf z} \times \mathbf u^*
      + \frac{C_d}H \left(
          \|\mathbf u\| \mathbf u^*
          + \frac{\mathbf u \cdot \mathbf u^*}{\|\mathbf u\|} \mathbf u
          \right)
      = \frac{\partial J}{\partial\mathbf u}

and the continuity equation is given by

..math::

    -\frac{\partial\eta}{\partial t}
      - g\nabla \cdot \mathbf u^*
      - \mathbf u \cdot \nabla \eta^*,
      - \frac{\widetilde{C_d}}{H^2} \|\mathbf u\| \mathbf u \cdot \mathbf u^*
      = \frac{\partial J}{\partial\eta},

where :math:`(\mathbf u^*,\eta^*)` are the adjoint variables corresponding to fluid velocity and free
surface elevation, :math:`(\mathbf u, eta)`. If Manning friction is used then
:math:`\widetilde{C_d} = \frac43 C_d`, otherwise :math:`\widetilde{C_d} = C_d`.

This implementation only accounts for Dirichlet conditions for elevation on segment :math:`\Gamma_D` and
free-slip conditions for velocity on segment :math:`\Gamma_{\mathrm{freeslip}}`. In the adjoint model
these become free-slip conditions for the adjoint velocity on :math:`\partial\Omega\backslash\Gamma_D`
and Dirichlet conditions for adjoint elevation on :math:`\partial\Omega\backslash\Gamma_{\mathrm{freeslip}}`.

The mixed discontinuous-continuous :math:`\mathbb P1_{DG}-\mathbb P2` discretisation used is very similar
to that presented in [1]. A :math:`\mathbb P2-\mathbb P1` Taylor-Hood element pair is also allowed.

[1] Funke, S. W., P. E. Farrell, and M. D. Piggott. "Reconstructing wave profiles from inundation data."
    Computer Methods in Applied Mechanics and Engineering 322 (2017): 167-186.
"""
from thetis.equation import *
from thetis.shallowwater_eq import ShallowWaterTerm
from thetis.utility import *


__all__ = ["AdjointShallowWaterEquations"]


g_grav = physical_constants['g_grav']


class AdjointShallowWaterTerm(ShallowWaterTerm):

    def get_bnd_functions(self, eta_star_in, u_star_in, bnd_id, bnd_conditions):
        """
        In the forward, we permit only free-slip conditions for the velocity and Dirichlet
        conditions for the elevation. Suppose these are imposed on Γ₁ and Γ₂, which are not
        necessarily disjoint. Then the adjoint has a free-slip condition for the adjoint
        velocity on the complement of Γ₂ and Dirichlet conditions for the elevation on the
        complement of Γ₁.
        """
        homogeneous = False
        funcs = bnd_conditions.get(bnd_id)
        if funcs is not None and 'elev' not in funcs and 'un' not in funcs:
            raise Exception('Unsupported bnd type: {:}'.format(funcs.keys()))

        if homogeneous:

            # Homogeneous boundary conditions as given in [1].
            if 'elev' in funcs and 'un' in funcs:  # Γ₁ ∪ Γ₂
                eta_star_ext = Constant(0.0)
                u_star_ext = Constant(0.0)*self.normal
            elif 'elev' in funcs:  # Γ₁
                eta_star_ext = Constant(0.0)
                u_star_ext = u_star_in  # assume symmetry
            elif 'un' in funcs:  # Γ₂
                eta_star_ext = eta_star_in  # assume symmetry
                u_star_ext = Constant(0.0)*self.normal
            else:  # funcs is None, ∂Ω \ (Γ₁ ∪ Γ₂)
                eta_star_ext = eta_star_in  # assume symmetry
                u_star_ext = u_star_in  # assume symmetry
        else:

            # Inhomogeneous boundary conditions obtained via integration by parts
            if 'elev' in funcs and 'un' in funcs:  # Γ₁ ∪ Γ₂
                eta_star_ext = eta_star_in  # assume symmetry
                u_star_ext = u_star_in  # assume symmetry
            elif 'elev' not in funcs:  # ∂Ω \ Γ₂
                eta_star_ext = eta_star_in  # assume symmetry
                u_star_ext = Constant(0.0)*self.normal
            elif 'un' not in funcs:  # ∂Ω \ Γ₁
                eta_star_ext = Constant(0.0)
                u_star_ext = u_star_in  # assume symmetry
            else:  # funcs is None, ∂Ω \ (Γ₁ ∪ Γ₂)
                eta_star_ext = Constant(0.0)
                u_star_ext = Constant(0.0)*self.normal

        return eta_star_ext, u_star_ext


class AdjointShallowWaterMomentumTerm(AdjointShallowWaterTerm):
    """
    Generic term in the first component of the adjoint shallow water equation that provides
    commonly used members and mapping for boundary functions.
    """
    def __init__(self, u_star_test, u_star_space, eta_star_space,
                 depth, options=None):
        super(AdjointShallowWaterMomentumTerm, self).__init__(u_star_space, depth, options)

        self.options = options

        self.u_star_test = u_star_test
        self.u_star_space = u_star_space
        self.eta_star_space = eta_star_space

        self.u_star_continuity = element_continuity(self.u_star_space.ufl_element()).horizontal
        self.eta_star_is_dg = element_continuity(self.eta_star_space.ufl_element()).horizontal == 'dg'


class AdjointShallowWaterContinuityTerm(AdjointShallowWaterTerm):
    """
    Generic term in the second component of the adjoint shallow equation that provides commonly
    used members and mapping for boundary functions.
    """
    def __init__(self, eta_star_test, eta_star_space, u_star_space,
                 depth, options=None):
        super(AdjointShallowWaterContinuityTerm, self).__init__(eta_star_space, depth, options)

        self.eta_star_test = eta_star_test
        self.eta_star_space = eta_star_space
        self.u_star_space = u_star_space

        self.u_star_continuity = element_continuity(self.u_star_space.ufl_element()).horizontal
        self.eta_star_is_dg = element_continuity(self.eta_star_space.ufl_element()).horizontal == 'dg'


class ExternalPressureGradientTerm(AdjointShallowWaterContinuityTerm):
    r"""
    Term resulting from differentiating the external pressure gradient term with respect to elevation.
    Note that, where the original term appeared in the momentum equation of the forward model, this
    term appears in the continuity equation of the adjoint model.

  ..math::

        -\langle g \nabla \cdot \mathbf u^*, \zeta\rangle_K
          = \langle g\nabla(\zeta), \mathbf u^*\rangle_K
            -\langle g\zeta, \mathbf u^*\cdot\widehat{\mathbf n}\rangle_{\partial K}.

    Unlike in the discretisation of the forward equations, we do not include fluxes for this term.
    """
    def residual(self, u_star, eta_star, fields, fields_old, bnd_conditions=None):

        u_star_by_parts = self.u_star_continuity in ['dg', 'hdiv']

        if u_star_by_parts:
            f = g_grav*inner(grad(self.eta_star_test), u_star)*self.dx
            for bnd_marker in self.boundary_markers:
                funcs = bnd_conditions.get(bnd_marker)
                ds_bnd = ds(int(bnd_marker), degree=self.quad_degree)
                eta_star_ext, u_star_ext = self.get_bnd_functions(eta_star, u_star, bnd_marker, bnd_conditions)
                # Compute linear Riemann solution with eta_star, eta_star_ext, u_star, u_star_ext
                n = self.normal
                total_h = self.depth.get_total_depth(fields.get('elev_2d'))
                eta_star_jump = eta_star - eta_star_ext
                un_star_rie = 0.5*inner(u_star + u_star_ext, n) + sqrt(total_h/g_grav)*eta_star_jump
                # un_star_jump = inner(u_star - u_star_ext, n)
                f += -g_grav*self.eta_star_test*un_star_rie*ds_bnd

                # f += -g_grav*self.eta_star_test*inner(u_star_ext, self.normal)*ds_bnd
        else:
            f = -g_grav*self.eta_star_test*div(u_star)*self.dx
            for bnd_marker in self.boundary_markers:
                funcs = bnd_conditions.get(bnd_marker)
                ds_bnd = ds(int(bnd_marker), degree=self.quad_degree)
                if funcs is not None and 'elev' not in funcs:
                    f += g_grav*dot(u_star, self.normal)*self.eta_star_test*ds_bnd

        return -f


class HUDivTermMomentum(AdjointShallowWaterMomentumTerm):
    r"""
    Term resulting from differentiating the :math:`\nabla\cdot(H\mathbf u)` term with respect to
    velocity. Note that, where the original term appeared in the continuity equation of the forward
    model, this term appears in the momentum equation of the adjoint model.

    The term is given by

  ..math::

        -\langle H \nabla \eta^*, z\rangle_K

    where :math:`H = b + \eta` in the nonlinear case and :math:`H = b` otherwise.

    Note that, unlike in the forward model, Dirichlet boundary conditions on the adjoint free surface
    elevation are applied strongly.
    """
    def residual(self, u_star, eta_star, fields, fields_old, bnd_conditions=None):
        total_h = self.depth.get_total_depth(fields.get('elev_2d'))

        eta_star_by_parts = self.eta_star_is_dg
        assert not eta_star_by_parts

        f = 0
        if not eta_star_by_parts:
            f += -total_h*inner(grad(eta_star), self.u_star_test)*self.dx
            # for bnd_marker in self.boundary_markers:
            #     funcs = bnd_conditions.get(bnd_marker)
            #     ds_bnd = ds(int(bnd_marker), degree=self.quad_degree)
            #     if funcs is not None:
            #         eta_star_ext, u_star_ext = self.get_bnd_functions(eta_star, u_star, bnd_marker, bnd_conditions)
            #         # Compute linear riemann solution with eta_star, eta_star_ext, u_star, u_star_ext
            #         un_star_jump = inner(u_star - u_star_ext, self.normal)
            #         eta_star_rie = 0.5*(eta_star + eta_star_ext) + sqrt(g_grav/total_h)*un_star_jump
            #         f += -total_h*(eta_star_rie-eta_star)*dot(self.u_star_test, self.normal)*ds_bnd
        return -f


class HUDivTermContinuity(AdjointShallowWaterContinuityTerm):
    r"""
    Term resulting from differentiating the :math:`\nabla\cdot(H\mathbf u)` term with respect to
    elevation.

  ..math::

        -\langle \zeta, \mathbf u \cdot \nabla \eta^* \rangle_\Omega
          + \langle \zeta, \eta^*\mathbf u\cdot\widehat{\mathbf n}\rangle_\Gamma,

    where the integral over :math:`\Gamma = \partial\Omega\backslash\Gamma_D` enforces the no-slip
    condition on the adjoint velocity.

    Note that this term only arises in the nonlinear case.
    """
    def residual(self, u_star, eta_star, fields, fields_old, bnd_conditions=None):
        if not self.options.use_nonlinear_equations:
            return 0

        eta_star_by_parts = self.eta_star_is_dg
        assert not eta_star_by_parts

        # Account for wetting and drying
        test = self.eta_star_test
        if self.options.get('use_wetting_and_drying'):
            test = test*self.depth.heaviside_approx(self.options.get('elev_2d'))

        f = 0
        uv = fields.get('uv_2d')
        if not eta_star_by_parts:
            f += -inner(grad(eta_star), test*uv)*self.dx
            for bnd_marker in self.boundary_markers:
                funcs = bnd_conditions.get(bnd_marker)
                ds_bnd = ds(int(bnd_marker), degree=self.quad_degree)
                if funcs is not None and 'elev' not in funcs:
                    f += eta_star*dot(uv, self.normal)*test*ds_bnd
        return -f


class HorizontalAdvectionTerm(AdjointShallowWaterMomentumTerm):
    r"""
    Term resulting from differentiating the nonlinear advection term by velocity.

  ..math::

        langle (\nabla\mathbf u)^T \mathbf u^*, \mathbf z \rangle_K
        - langle (\nabla \cdot \mathbf u) \mathbf u^*, \mathbf z \rangle_K
        - langle \mathbf u \cdot \nabla \mathbf u^*, \mathbf z \rangle_K
          = \langle (\mathbf z \cdot \nabla)\mathbf u, \mathbf u^* \rangle_K
            + \langle (\mathbf u \cdot \nabla)\mathbf z, \mathbf u^* \rangle_K
            - \langle [[\mathbf z]], \mathbf u \cdot \widehat{\mathbf n} \mathbf u^*\rangle_{\Gamma^-}
            - \langle [[\mathbf u]], \mathbf z \cdot \widehat{\mathbf n} \mathbf u^*\rangle_{\Gamma^-},

    where :math:`\Gamma^-=\{\gamma\in\partial K\mid\mathbf u\cdot \widehat{\mathbf n}|_\gamma < 0\}` is
    comprised of downwind faces of :math:`K`.
    """
    def residual(self, u_star, eta_star, fields, fields_old, bnd_conditions=None):
        if not self.options.use_nonlinear_equations:
            return 0

        horiz_advection_by_parts = self.u_star_continuity in ['dg', 'hdiv']

        # Downwind velocity
        uv = fields.get('uv_2d')
        un = 0.5*(abs(dot(uv, self.normal)) - dot(uv, self.normal))  # u.n if u.n < 0 else 0
        downwind = lambda x: conditional(un < 0, dot(x, self.normal), 0)

        f = 0
        f += inner(dot(self.u_star_test, nabla_grad(uv)), u_star)*self.dx
        f += inner(dot(uv, nabla_grad(self.u_star_test)), u_star)*self.dx
        if horiz_advection_by_parts:
            f += -inner(jump(self.u_star_test), 2*avg(un*u_star))*self.dS
            f += -inner(2*avg(downwind(self.u_star_test)*u_star), jump(uv))*self.dS
        else:
            f += inner(dot(transpose(grad(uv)), u_star), self.u_star_test)*self.dx

        return -f


class HorizontalViscosityTerm(AdjointShallowWaterMomentumTerm):
    r"""

  ..math::

        \langle \nabla \cdot (\nu \nabla \mathbf u^*), \mathbf z\rangle_K
    """  # TODO: doc
    def residual(self, u_star, eta_star, fields, fields_old, bnd_conditions=None):
        # total_h = self.depth.get_total_depth(fields.get('elev_2d'))

        nu = fields_old.get('viscosity_h')
        if nu is None:
            return 0

        raise NotImplementedError  # TODO


class CoriolisTerm(AdjointShallowWaterMomentumTerm):
    r"""
    This term is identical to that in the forward model, except for the opposite sign:

  ..math::

        -\langle f \widehat{\mathbf z} \times \mathbf u^*, \mathbf z\rangle_K

    where :math:`f` is the user-specified Coriolis parameter.
    """
    def residual(self, u_star, eta_star, fields, fields_old, bnd_conditions=None):
        coriolis = fields_old.get('coriolis')
        f = 0
        if coriolis is not None:
            f += -coriolis*(-u_star[1]*self.u_star_test[0] + u_star[0]*self.u_star_test[1])*self.dx
        return -f


class QuadraticDragTermMomentum(AdjointShallowWaterMomentumTerm):
    r"""
    Term resulting from differentiating the quadratic drag term with respect to velocity.

  ..math::

        \left\langle
            \frac{\C_d}H \left(
                \|\mathbf u\| \mathbf u
                + \frac{\mathbf u^* \cdot \mathbf u}{\|\mathbf u\|}\mathbf u
            \right), \mathbf z
        \right\rangle_K
    """
    def residual(self, u_star, eta_star, fields, fields_old, bnd_conditions=None):
        if not self.options.use_nonlinear_equations:
            return 0
        total_h = self.depth.get_total_depth(fields.get('elev_2d'))
        manning_drag_coefficient = fields_old.get('manning_drag_coefficient')
        C_D = fields_old.get('quadratic_drag_coefficient')
        f = 0
        if manning_drag_coefficient is not None:
            if C_D is not None:
                raise Exception('Cannot set both dimensionless and Manning drag parameter')
            C_D = g_grav*manning_drag_coefficient**2/total_h**(1./3.)
        if C_D is None:
            return -f

        uv = fields.get('uv_2d')
        if uv is None:
            raise Exception('Adjoint equation does not have access to forward solution velocity')

        unorm = sqrt(dot(uv, uv) + self.options.norm_smoother**2)
        f += C_D*unorm*inner(self.u_star_test, u_star)*self.dx
        f += C_D*inner(self.u_star_test, uv)*inner(u_star, uv)/unorm*self.dx
        return -f


class QuadraticDragTermContinuity(AdjointShallowWaterContinuityTerm):
    r"""
    Term resulting from differentiating the quadratic drag term with respect to elevation.

  ..math::

        \left\langle
            \frac{\widetilde{C_d}}{H^2} \|\mathbf u\| \mathbf u \cdot \mathbf u^*, \zeta
        \right\rangle_K,

    where :math:`\widetilde{C_d}` is given by :math:`\frac43 C_d` in the case of Manning friction
    and :math:`C_d` in other cases where the drag coefficient is independent of the water depth.
    """
    def residual(self, u_star, eta_star, fields, fields_old, bnd_conditions=None):
        if not self.options.use_nonlinear_equations:
            return 0
        total_h = self.depth.get_total_depth(fields.get('elev_2d'))
        manning_drag_coefficient = fields_old.get('manning_drag_coefficient')
        # TODO: Account for Nikuradse bed roughness
        C_D = fields_old.get('quadratic_drag_coefficient')
        f = 0
        if manning_drag_coefficient is not None:
            if C_D is not None:
                raise Exception('Cannot set both dimensionless and Manning drag parameter')
            C_D = 4/3*g_grav*manning_drag_coefficient**2/total_h**(1./3.)
        if C_D is None:
            return -f
        uv = fields.get('uv_2d')
        if uv is None:
            raise Exception('Adjoint equation does not have access to forward solution velocity')

        # Account for wetting and drying
        if self.options.get('use_wetting_and_drying'):
            C_D = C_D*self.depth.heaviside_approx(self.options.get('elev_2d'))

        unorm = sqrt(dot(uv, uv) + self.options.norm_smoother**2)
        f += -C_D*unorm*inner(u_star, uv)*self.eta_star_test/total_h**2*self.dx
        return -f


class LinearDragTerm(AdjointShallowWaterMomentumTerm):
    r"""
    Identical to the linear drag term in the forward model, :math:`C \mathbf u^*`, where :math:`C` is
    a user-defined drag coefficient.
    """
    def residual(self, u_star, eta_star, fields, fields_old, bnd_conditions=None):
        linear_drag_coefficient = fields_old.get('linear_drag_coefficient')
        f = 0
        if linear_drag_coefficient is not None:
            f += linear_drag_coefficient*inner(self.u_star_test, u_star)*self.dx
        return -f


class TurbineDragTermMomentum(AdjointShallowWaterMomentumTerm):
    def residual(self, u_star, eta_star, fields, fields_old, bnd_conditions=None):
        if not self.options.use_nonlinear_equations:
            return 0
        # total_h = self.depth.get_total_depth(fields.get('elev_2d'))
        f = 0
        for subdomain_id, farm_options in self.options.tidal_turbine_farms.items():
            raise NotImplementedError  # TODO
        return -f


class TurbineDragTermContinuity(AdjointShallowWaterContinuityTerm):
    def residual(self, u_star, eta_star, fields, fields_old, bnd_conditions=None):
        if not self.options.use_nonlinear_equations:
            return 0
        # total_h = self.depth.get_total_depth(fields.get('elev_2d'))
        f = 0
        for subdomain_id, farm_options in self.options.tidal_turbine_farms.items():
            raise NotImplementedError  # TODO
        return -f


class MomentumSourceTerm(AdjointShallowWaterMomentumTerm):
    r"""
    Term on the right hand side of the adjoint momentum equation corresponding to the derivative of
    the quantity of interest :math:`J` with respect to velocity, :math:`\partial J/\partial u`.
    """
    def residual(self, u_star, eta_star, fields, fields_old, bnd_conditions=None):
        dJdu = fields_old.get('dJdu')

        f = 0
        if dJdu is not None:
            f += inner(dJdu, self.u_star_test)*self.dx
        return f


class ContinuitySourceTerm(AdjointShallowWaterContinuityTerm):
    r"""
    Term on the right hand side of the adjoint continuity equation corresponding to the derivative of
    the quantity of interest :math:`J` with respect to elevation, :math:`\partial J/\partial\eta`.
    """
    def residual(self, u_star, eta_star, fields, fields_old, bnd_conditions=None):
        dJdeta = fields_old.get('dJdeta')

        f = 0
        if dJdeta is not None:
            f += inner(dJdeta, self.eta_star_test)*self.dx
        return f


class BaseAdjointShallowWaterEquation(Equation):
    # TODO: doc
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

    def residual_u_star_eta_star(self, label, u_star, eta_star, fields, fields_old, bnd_conditions):
        f = 0
        for term in self.select_terms(label):
            f += term.residual(u_star, eta_star, fields, fields_old, bnd_conditions)
        return f


class AdjointShallowWaterEquations(BaseAdjointShallowWaterEquation):
    # TODO: doc
    def __init__(self, function_space, depth, options):
        """
        :arg function_space: Mixed function space where the solution belongs
        :arg depth: :class: `DepthExpression` containing depth info
        :arg options: :class:`.AttrDict` object containing all circulation model options
        """
        super(AdjointShallowWaterEquations, self).__init__(function_space, depth, options)
        if options.get('family') == 'dg-dg':
            raise NotImplementedError("Equal order DG finite element not supported.")
        if options.get('wind_stress') is not None:
            raise NotImplementedError("Wind stress not supported in continuous adjoint model.")
        if options.get('bottom_drag') is not None:
            raise NotImplementedError("3D bottom drag not supported in continuous adjoint model.")

        u_star_test, eta_star_test = TestFunctions(function_space)
        u_star_space, eta_star_space = function_space.split()

        self.add_momentum_terms(u_star_test, u_star_space, eta_star_space, depth, options)

        self.add_continuity_terms(eta_star_test, eta_star_space, u_star_space, depth, options)

    def mass_term(self, solution):
        if self.options.get('use_wetting_and_drying'):
            raise NotImplementedError  # TODO: Need old elevation
            # u_star, eta_star = solution if isinstance(solution, list) else split(solution)
            # f = inner(u_star, self.u_star_test)*dx
            # f += inner(eta_star, self.depth.heaviside_approx(eta)*self.eta_star_test)*dx
        else:
            f = super(AdjointShallowWaterEquations, self).mass_term(solution)
        return f

    def residual(self, label, solution, solution_old, fields, fields_old, bnd_conditions):
        # NOTE: `solution_old` doesn't do anything since the equation is linear
        u_star, eta_star = solution if isinstance(solution, list) else split(solution)
        return self.residual_u_star_eta_star(label, u_star, eta_star, fields, fields_old, bnd_conditions)
