"""
Depth averaged shallow water equations as in Thetis, with a few minor modifications:

    1. Allow the use of a P2-P1 Taylor-Hood discretisation. Note that Dirichlet boundary conditions
       on the elevation are enforced strongly.
    2. Do not include friction terms if the `use_nonlinear_equations` option is set to false.
    3. Do not include 3D bottom drag or atmospheric pressure terms.
    4. Enable choice of anisotropic cell size measure.

**********************************************************************************************
*  NOTE: This file is based on the Thetis project (https://thetisproject.org) and contains   *
*        some copied code.                                                                   *
**********************************************************************************************
"""
from __future__ import absolute_import
from thetis.equation import *
from thetis.utility import *
import thetis.shallowwater_eq as thetis_sw
from ..equation import Equation


__all__ = ["ShallowWaterEquations", "ShallowWaterMomentumEquation"]


g_grav = physical_constants['g_grav']


class ExternalPressureGradientTerm(thetis_sw.ExternalPressureGradientTerm):
    """
    External pressure gradient term from Thetis, modified to account for P2-P1 Taylor-Hood
    discretisation. In that case, Dirichlet conditions on free surface elevation are enforced
    strongly.
    """
    def residual(self, *args, **kwargs):
        if self.options.get('element_family') == 'cg-cg':
            return -g_grav*inner(grad(args[1]), self.u_test)*self.dx
        else:
            return super(ExternalPressureGradientTerm, self).residual(*args, **kwargs)


class HorizontalAdvectionTerm(thetis_sw.HorizontalAdvectionTerm):
    """
    Nonlinear advection term from Thetis, modified to account for mesh movement under a prescribed
    mesh velocity and also to allow for the use of a P2-P1 Taylor-Hood discretisation. In that case,
    this term is not integrated by parts.
    """
    def residual(self, *args, **kwargs):
        uv = args[0]
        uv_old = args[2]

        f = 0
        if not self.options.use_nonlinear_equations:
            return -f

        # Allow for Taylor-Hood discretisation
        if self.u_continuity in ['dg', 'hdiv']:
            f += -super(HorizontalAdvectionTerm, self).residual(*args, **kwargs)
        else:
            f += inner(dot(uv_old, nabla_grad(uv)), self.u_test)*dx

        return -f


class QuadraticDragTerm(thetis_sw.QuadraticDragTerm):
    """
    Quadratic bottom friction term from Thetis, modified so that it isn't included in the linear
    model.
    """
    def residual(self, *args, **kwargs):
        if not self.options.use_nonlinear_equations:
            return 0
        return super(QuadraticDragTerm, self).residual(*args, **kwargs)


class TurbineDragTerm(thetis_sw.TurbineDragTerm):
    """
    Turbine drag term from Thetis, modified so that it isn't included in the linear model.
    """
    def residual(self, *args, **kwargs):
        if not self.options.use_nonlinear_equations:
            return 0
        return super(TurbineDragTerm, self).residual(*args, **kwargs)


class BaseShallowWaterEquation(Equation):
    """
    Copied here from `thetis/shallowwater_eq` to hook up modified terms.
    """
    def __init__(self, function_space, depth, options):
        super(BaseShallowWaterEquation, self).__init__(function_space)
        self.depth = depth
        self.options = options

    def add_momentum_terms(self, *args):
        self.add_term(ExternalPressureGradientTerm(*args), 'implicit')
        self.add_term(HorizontalAdvectionTerm(*args), 'explicit')
        self.add_term(thetis_sw.HorizontalViscosityTerm(*args), 'explicit')
        self.add_term(thetis_sw.CoriolisTerm(*args), 'explicit')
        self.add_term(thetis_sw.WindStressTerm(*args), 'source')
        # self.add_term(thetis_sw.AtmosphericPressureTerm(*args), 'source')
        self.add_term(QuadraticDragTerm(*args), 'explicit')
        self.add_term(thetis_sw.LinearDragTerm(*args), 'explicit')
        # self.add_term(thetis_sw.BottomDrag3DTerm(*args), 'source')
        self.add_term(TurbineDragTerm(*args), 'implicit')
        self.add_term(thetis_sw.MomentumSourceTerm(*args), 'source')

    def add_continuity_terms(self, *args):
        self.add_term(thetis_sw.HUDivTerm(*args), 'implicit')
        self.add_term(thetis_sw.ContinuitySourceTerm(*args), 'source')

    def residual_uv_eta(self, label, uv, eta, uv_old, eta_old, fields, fields_old, bnd_conditions):
        f = 0
        for term in self.select_terms(label):
            f += term.residual(uv, eta, uv_old, eta_old, fields, fields_old, bnd_conditions)
        return f


class ShallowWaterEquations(BaseShallowWaterEquation):
    """
    Copied here from `thetis/shallowwater_eq` to hook up modified terms.
    """
    def __init__(self, function_space, depth, options):
        """
        :arg function_space: Mixed function space where the solution belongs
        :arg depth: :class: `DepthExpression` containing depth info
        :arg options: :class:`.AttrDict` object containing all circulation model options
        """
        super(ShallowWaterEquations, self).__init__(function_space, depth, options)

        u_test, eta_test = TestFunctions(function_space)
        u_space, eta_space = function_space.split()

        self.add_momentum_terms(u_test, u_space, eta_space, depth, options)

        self.add_continuity_terms(eta_test, eta_space, u_space, depth, options)
        self.bathymetry_displacement_mass_term = thetis_sw.BathymetryDisplacementMassTerm(
            eta_test, eta_space, u_space, depth, options)

    def mass_term(self, solution):
        f = super(ShallowWaterEquations, self).mass_term(solution)
        f += -self.bathymetry_displacement_mass_term.residual(solution)
        return f

    def residual(self, label, solution, solution_old, fields, fields_old, bnd_conditions):
        if isinstance(solution, list):
            uv, eta = solution
        else:
            uv, eta = split(solution)
        uv_old, eta_old = split(solution_old)
        return self.residual_uv_eta(label, uv, eta, uv_old, eta_old, fields, fields_old, bnd_conditions)


class ShallowWaterMomentumEquation(BaseShallowWaterEquation):
    """
    Copied here from `thetis/shallowwater_eq` to hook up modified terms.
    """
    def __init__(self, u_test, u_space, eta_space, depth, options):
        """
        :arg u_test: test function of the velocity function space
        :arg u_space: velocity function space
        :arg eta_space: elevation function space
        :arg depth: :class: `DepthExpression` containing depth info
        :arg options: :class:`.AttrDict` object containing all circulation model options
        """
        super(ShallowWaterMomentumEquation, self).__init__(u_space, depth, options)
        self.add_momentum_terms(u_test, u_space, eta_space, depth, options)

    def residual(self, label, solution, solution_old, fields, fields_old, bnd_conditions):
        uv = solution
        uv_old = solution_old
        eta = fields['eta']
        eta_old = fields_old['eta']
        return self.residual_uv_eta(label, uv, eta, uv_old, eta_old, fields, fields_old, bnd_conditions)
