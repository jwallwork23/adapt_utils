"""
Goal-oriented error indicators for the tracer transport model. See [Wallwork et al. 2021] for details
on the formulation.

[Wallwork et al. 2021] J. G. Wallwork, N. Barral, D. A. Ham, M. D. Piggott, "Goal-Oriented Error
    Estimation and Mesh Adaptation for Tracer Transport Problems", submitted to Computer
    Aided Design.
"""
from __future__ import absolute_import
from thetis.utility import *
from thetis.tracer_eq_2d import TracerTerm

import numpy as np

from ..error_estimation import GOErrorEstimatorTerm, GOErrorEstimator


__all__ = ['TracerGOErrorEstimator']


g_grav = physical_constants['g_grav']


class TracerGOErrorEstimatorTerm(GOErrorEstimatorTerm, TracerTerm):
    """
    Generic :class:`GOErrorEstimatorTerm` term in a goal-oriented error estimator for the 2D tracer
    model.
    """
    def __init__(self, function_space, depth, options):
        """
        :arg function_space: :class:`FunctionSpace` where the solution belongs
        :kwarg depth: DepthExpression for the domain
        """
        TracerTerm.__init__(self, function_space, depth, options)
        GOErrorEstimatorTerm.__init__(self, function_space.mesh())

    def inter_element_flux(self, *args, **kwargs):
        return 0

    def boundary_flux(self, *args, **kwargs):
        return 0


class HorizontalAdvectionGOErrorEstimatorTerm(TracerGOErrorEstimatorTerm):
    """
    :class:`TracerGOErrorEstimatorTerm` object associated with the :class:`HorizontalAdvectionTerm`
    term of the 2D tracer model.
    """
    def element_residual(self, c, _, e_star, __, ___, fields_old):
        if fields_old.get('uv_2d') is None:
            return 0
        self.corr_factor = fields_old.get('tracer_advective_velocity_factor')
        uv = self.corr_factor*fields_old['uv_2d']

        # Apply SUPG stabilisation
        if not self.horizontal_dg and self.options.use_supg_tracer:
            e_star = e_star + self.supg_stabilisation*dot(uv, grad(e_star))

        return -self.p0test*e_star*inner(uv, grad(c))*self.dx

    def inter_element_flux(self, c, _, e_star, __, ___, fields_old):
        if fields_old.get('uv_2d') is None:
            return 0
        self.corr_factor = fields_old.get('tracer_advective_velocity_factor')
        uv = self.corr_factor*fields_old['uv_2d']
        uv_p1 = fields_old.get('uv_p1')
        uv_mag = fields_old.get('uv_mag')
        # FIXME is this an option?
        lax_friedrichs_factor = fields_old.get('lax_friedrichs_tracer_scaling_factor')

        flux_terms = 0
        if self.adjoint:
            flux_terms += self.restrict(inner(c*dot(uv, self.normal), e_star))*self.dS
        if self.horizontal_dg:

            # Interface term
            #   NOTE: It is ds, rather than dS, due to cancellations
            un_av = dot(avg(uv), self.normal('-'))
            s = 0.5*(sign(un_av) + 1.0)
            c_up = c('-')*s + c('+')*(1-s)
            flux_terms += -c_up*self.restrict(e_star)*jump(uv, self.normal)*ds

            # Lax-Friedrichs stabilization
            if self.options.use_lax_friedrichs_tracer:
                if uv_p1 is not None:
                    gamma = 0.5*abs((avg(uv_p1)[0]*self.normal('-')[0]
                                     + avg(uv_p1)[1]*self.normal('-')[1]))*lax_friedrichs_factor
                elif uv_mag is not None:
                    gamma = 0.5*avg(uv_mag)*lax_friedrichs_factor
                else:
                    gamma = 0.5*abs(un_av)*lax_friedrichs_factor
                flux_terms += -gamma*dot(self.restrict(e_star), jump(c))*self.dS

        return flux_terms

    def boundary_flux(self, c, _, e_star, __, ___, fields_old, bnd_conditions):
        if fields_old.get('uv_2d') is None:
            return 0
        elev = fields_old['elev_2d']
        self.corr_factor = fields_old.get('tracer_advective_velocity_factor')
        uv = self.corr_factor * fields_old['uv_2d']

        flux_terms = 0
        if self.horizontal_dg and bnd_conditions is not None:  # TODO: Check signs
            for bnd_marker in self.boundary_markers:
                funcs = bnd_conditions.get(bnd_marker)
                ds_bnd = ds(int(bnd_marker), degree=self.quad_degree)
                c_in = c
                if funcs is not None and 'value' in funcs:
                    c_ext, uv_ext, eta_ext = self.get_bnd_functions(c_in, uv, elev, bnd_marker, bnd_conditions)
                    uv_av = 0.5*(uv + uv_ext)
                    s = 0.5*(sign(dot(uv_av, self.normal)) + 1.0)
                    c_up = c_in*s + c_ext*(1-s)
                    flux_terms += -c_up*dot(uv_av, self.normal)*e_star*ds_bnd
                else:
                    flux_terms += -c_in*dot(uv, self.normal)*e_star*ds_bnd

        return flux_terms


class ConservativeHorizontalAdvectionGOErrorEstimatorTerm(TracerGOErrorEstimatorTerm):
    """
    :class:`TracerGOErrorEstimatorTerm` object associated with the
    :class:`ConservativeHorizontalAdvectionTerm` term of the 2D conservative tracer model.
    """
    def element_residual(self, c, _, e_star, __, ___, fields_old):
        if fields_old.get('uv_2d') is None:
            return 0
        self.corr_factor = fields_old.get('tracer_advective_velocity_factor')
        uv = self.corr_factor*fields_old['uv_2d']

        # Apply SUPG stabilisation
        if not self.horizontal_dg and self.options.use_supg_tracer:
            e_star = e_star + self.supg_stabilisation*dot(uv, grad(e_star))

        return -self.p0test*e_star*div(uv*c)*self.dx

    def inter_element_flux(self, c, _, e_star, __, ___, fields_old):
        if fields_old.get('uv_2d') is None:
            return 0
        self.corr_factor = fields_old.get('tracer_advective_velocity_factor')
        uv = self.corr_factor*fields_old['uv_2d']

        flux_terms = 0
        if self.adjoint:
            flux_terms += self.restrict(inner(c*dot(uv, self.normal), e_star))*self.dS
        if self.horizontal_dg:
            raise NotImplementedError  # TODO

        return flux_terms

    def boundary_flux(self, c, _, e_star, __, ___, fields_old, bnd_conditions):
        if fields_old.get('uv_2d') is None:
            return 0
        # elev = fields_old['elev_2d']
        # self.corr_factor = fields_old.get('tracer_advective_velocity_factor')
        # uv = self.corr_factor*fields_old['uv_2d']

        flux_terms = 0
        if self.horizontal_dg and bnd_conditions is not None:
            raise NotImplementedError  # TODO

        return flux_terms


class HorizontalDiffusionGOErrorEstimatorTerm(TracerGOErrorEstimatorTerm):
    """
    :class:`TracerGOErrorEstimatorTerm` object associated with the :class:`HorizontalDiffusionTerm`
    term of the 2D tracer model.
    """
    def element_residual(self, c, _, e_star, __, ___, fields_old):
        if fields_old.get('diffusivity_h') is None:
            return 0
        diffusivity_h = fields_old['diffusivity_h']
        diff_tensor = as_matrix([[diffusivity_h, 0, ],
                                 [0, diffusivity_h, ]])

        # Apply SUPG stabilisation
        uv = fields_old.get('uv_2d')
        if not self.horizontal_dg and self.options.use_supg_tracer and uv is not None:
            self.corr_factor = fields_old.get('tracer_advective_velocity_factor')
            uv = self.corr_factor*uv
            e_star = e_star + self.supg_stabilisation*dot(uv, grad(e_star))

        return self.p0test*e_star*div(dot(diff_tensor, grad(c)))*self.dx

    def inter_element_flux(self, c, _, e_star, __, ___, fields_old):
        if fields_old.get('diffusivity_h') is None:
            return 0
        diffusivity_h = fields_old['diffusivity_h']
        diff_tensor = as_matrix([[diffusivity_h, 0, ],
                                 [0, diffusivity_h, ]])

        flux_terms = 0
        if self.horizontal_dg:  # TODO: Check signs
            alpha = self.options.sipg_factor_tracer
            cell = self.mesh.ufl_cell()
            p = self.function_space.ufl_element().degree()
            cp = (p + 1)*(p + 2)/2 if cell == triangle else (p + 1)**2
            l_normal = CellVolume(self.mesh)/FacetArea(self.mesh)
            sigma = alpha*cp/l_normal
            sp = sigma('+')
            sm = sigma('-')
            sigma = conditional(sp > sm, sp, sm)
            e_star_n = self.restrict(e_star*self.normal)
            flux_terms += -sigma*inner(e_star_n, dot(avg(diff_tensor), jump(c, self.normal)))*self.dS
            flux_terms += inner(e_star_n, avg(dot(diff_tensor, grad(c))))*self.dS
            e_star_av = 0.5*self.p0test*e_star
            flux_terms += inner(dot(avg(diff_tensor), grad(e_star_av)), jump(c, self.normal))*self.dS
        else:
            flux_terms += self.restrict(inner(dot(diff_tensor, grad(c)), e_star*self.normal))*self.dS
        return flux_terms

    def boundary_flux(self, c, _, e_star, __, ___, fields_old, bnd_conditions):
        flux_terms = 0
        if fields_old.get('diffusivity_h') is None:
            return flux_terms
        diffusivity_h = fields_old['diffusivity_h']
        diff_tensor = as_matrix([[diffusivity_h, 0, ],
                                 [0, diffusivity_h, ]])

        if bnd_conditions is not None:
            for bnd_marker in self.boundary_markers:
                funcs = bnd_conditions.get(bnd_marker)
                ds_bnd = ds(int(bnd_marker), degree=self.quad_degree)
                c_in = c

                # Ignore open boundaries
                if funcs is None or not ('value' in funcs or 'diff_flux' in funcs):
                    continue

                # Ignore Dirichlet boundaries for CG
                elif 'value' in funcs and not self.horizontal_dg:
                    continue

                # Term from integration by parts
                diff_flux = dot(diff_tensor, grad(c_in))
                flux_terms += self.p0test*inner(diff_flux, e_star*self.normal)*ds_bnd

                # Terms from boundary conditions
                elev = fields_old['elev_2d']
                self.corr_factor = fields_old.get('tracer_advective_velocity_factor')
                uv = self.corr_factor*fields_old['uv_2d']
                if 'value' in funcs:
                    c_ext, uv_ext, eta_ext = self.get_bnd_functions(c_in, uv, elev, bnd_marker, bnd_conditions)
                    uv_av = 0.5*(uv + uv_ext)
                    s = 0.5*(sign(dot(uv_av, self.normal)) + 1.0)
                    c_up = c_in*s + c_ext*(1-s)
                    diff_flux_up = dot(diff_tensor, grad(c_up))
                    flux_terms += -self.p0test*e_star*dot(diff_flux_up, self.normal)*ds_bnd
                elif 'diff_flux' in funcs:
                    if funcs['diff_flux'] == 'adjoint':
                        flux_terms += self.p0test*e_star*dot(c, self.normal)*ds_bnd
                    else:
                        flux_terms += -self.p0test*e_star*funcs['diff_flux']*ds_bnd
        return flux_terms


class SourceGOErrorEstimatorTerm(TracerGOErrorEstimatorTerm):
    """
    :class:`TracerGOErrorEstimatorTerm` object associated with the :class:`SourceTerm` term of the
    2D tracer model.
    """
    def element_residual(self, c, _, e_star, __, ___, fields_old):
        f = 0
        source = fields_old.get('source')
        if source is None:
            return f

        # Apply SUPG stabilisation
        uv = fields_old.get('uv_2d')
        if not self.horizontal_dg and self.options.use_supg_tracer and uv is not None:
            self.corr_factor = fields_old.get('tracer_advective_velocity_factor')
            uv = self.corr_factor*uv
            e_star = e_star + self.supg_stabilisation*dot(uv, grad(e_star))

        f += self.p0test*inner(source, e_star)*self.dx
        return f


class TracerGOErrorEstimator(GOErrorEstimator):
    """
    :class:`GOErrorEstimator` for the 2D tracer model.
    """
    def __init__(self, function_space, depth, options, velocity, adjoint=False):
        self.adjoint = adjoint
        super(TracerGOErrorEstimator, self).__init__(function_space, anisotropic=anisotropic)

        # Apply SUPG stabilisation
        tau = None
        if options.use_supg_tracer:
            unorm = options.horizontal_velocity_scale
            if unorm.values()[0] > 0:
                cellsize = anisotropic_cell_size(function_space.mesh())
                tau = 0.5*cellsize/unorm
                D = options.horizontal_diffusivity_scale
                if D.values()[0] > 0:
                    Pe = 0.5*unorm*cellsize/D
                    tau = min_value(tau, Pe/3)
        self.supg_stabilisation = tau

        args = (function_space, depth, options)
        self.add_terms(*args)

    def add_terms(self, *args):
        if conservative:
            self.add_term(ConservativeHorizontalAdvectionGOErrorEstimatorTerm(*args), 'explicit')
        else:
            self.add_term(HorizontalAdvectionGOErrorEstimatorTerm(*args), 'explicit')
        self.add_term(HorizontalDiffusionGOErrorEstimatorTerm(*args), 'explicit')
        self.add_term(SourceGOErrorEstimatorTerm(*args), 'source')

    def add_term(self, term, label):
        """
        Add :class:`term` to the equation as a :str:`label` type term.

        Also, pass over the chosen cell size measure and any stabilisation parameters.
        """
        super(TracerGOErrorEstimator, self).add_term(term, label)
        key = term.__class__.__name__
        self.terms[key].supg_stabilisation = self.supg_stabilisation
        self.terms[key].adjoint = self.adjoint

    def mass_term(self, c, e_star, vector=False, velocity=None, **kwargs):
        """
        Account for SUPG stabilisation in mass term.
        """
        if self.options.use_supg_tracer:
            e_star = e_star + self.supg_stabilisation*dot(velocity, grad(e_star))
        mass = self.p0test*inner(c, e_star)*dx
        if vector:
            mass = np.array([mass])
        return mass

    def setup_strong_residual(self, label, c, c_old, fields, fields_old):
        """
        Setup strong residual for tracer transport model.
        """
        adj = Function(self.P0).assign(1.0)
        args = (c, c_old, adj, adj, fields, fields_old)
        self._strong_residual_terms = 0
        for term in self.select_terms(label):
            self._strong_residual_terms += term.element_residual(*args)
        self._strong_residual_terms = np.array([self._strong_residual_terms])
