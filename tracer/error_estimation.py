"""
Goal-oriented error indicators for the tracer transport model. See [Wallwork et al. 2021] for details
on the formulation.

[Wallwork et al. 2021] J. G. Wallwork, N. Barral, D. A. Ham, M. D. Piggott, "Goal-Oriented Error
    Estimation and Mesh Adaptation for Tracer Transport Problems", to be submitted to Computer
    Aided Design.
"""
from __future__ import absolute_import
from thetis.utility import *
from thetis.tracer_eq_2d import TracerTerm
from ..error_estimation import GOErrorEstimatorTerm, GOErrorEstimator


__all__ = ['TracerGOErrorEstimator']


g_grav = physical_constants['g_grav']


class TracerGOErrorEstimatorTerm(GOErrorEstimatorTerm, TracerTerm):
    """
    Generic :class:`GOErrorEstimatorTerm` term in a goal-oriented error estimator for the 2D tracer
    model.
    """
    def __init__(self, function_space,
                 depth=None,
                 use_lax_friedrichs=True,
                 sipg_parameter=Constant(10.0)):
        """
        :arg function_space: :class:`FunctionSpace` where the solution belongs
        :kwarg depth: DepthExpression for the domain
        """
        TracerTerm.__init__(self, function_space, depth, use_lax_friedrichs, sipg_parameter)
        GOErrorEstimatorTerm.__init__(self, function_space.mesh())

    def inter_element_flux(self, *args, **kwargs):
        return 0

    def boundary_flux(self, *args, **kwargs):
        return 0


class TracerHorizontalAdvectionGOErrorEstimatorTerm(TracerGOErrorEstimatorTerm):
    """
    :class:`TracerGOErrorEstimatorTerm` object associated with the :class:`HorizontalAdvectionTerm`
    term of the 2D tracer model.
    """
    def element_residual(self, solution, solution_old, arg, arg_old, fields, fields_old):
        if fields_old.get('uv_2d') is None:
            return 0
        self.corr_factor = fields_old.get('tracer_advective_velocity_factor')
        uv = self.corr_factor*fields_old['uv_2d']

        # Apply SUPG stabilisation
        if not self.horizontal_dg and self.stabilisation in ('su', 'supg'):
            arg = arg + self.supg_stabilisation*dot(uv, grad(arg))

        return -self.p0test*arg*inner(uv, grad(solution))*self.dx

    def inter_element_flux(self, solution, solution_old, arg, arg_old, fields, fields_old):
        if fields_old.get('uv_2d') is None:
            return 0
        self.corr_factor = fields_old.get('tracer_advective_velocity_factor')

        uv = self.corr_factor*fields_old['uv_2d']
        uv_p1 = fields_old.get('uv_p1')
        uv_mag = fields_old.get('uv_mag')
        # FIXME is this an option?
        lax_friedrichs_factor = fields_old.get('lax_friedrichs_tracer_scaling_factor')

        flux_terms = 0
        if self.horizontal_dg:

            # Interface term
            uv_av = avg(uv)
            un_av = (uv_av[0]*self.normal('-')[0]
                     + uv_av[1]*self.normal('-')[1])
            s = 0.5*(sign(un_av) + 1.0)
            c_up = solution('-')*s + solution('+')*(1-s)
            loc = self.p0test*arg
            flux_terms += -c_up*(loc('+') + loc('-'))*jump(uv, self.normal)*self.dS

            # Lax-Friedrichs stabilization
            if self.use_lax_friedrichs:
                if uv_p1 is not None:
                    gamma = 0.5*abs((avg(uv_p1)[0]*self.normal('-')[0]
                                     + avg(uv_p1)[1]*self.normal('-')[1]))*lax_friedrichs_factor
                elif uv_mag is not None:
                    gamma = 0.5*avg(uv_mag)*lax_friedrichs_factor
                else:
                    gamma = 0.5*abs(un_av)*lax_friedrichs_factor
                arg_jump = self.p0test*arg('+') - self.p0test*arg('-')
                flux_terms += -gamma*dot(arg_jump, jump(solution))*self.dS
        return flux_terms

    def boundary_flux(self, solution, solution_old, arg, arg_old, fields, fields_old, bnd_conditions):
        if fields_old.get('uv_2d') is None:
            return 0
        elev = fields_old['elev_2d']
        self.corr_factor = fields_old.get('tracer_advective_velocity_factor')
        uv = self.corr_factor * fields_old['uv_2d']

        flux_terms = 0
        if self.horizontal_dg:  # TODO: Check signs
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
                        flux_terms += -c_up*(uv_av[0]*self.normal[0]
                                             + uv_av[1]*self.normal[1])*arg*ds_bnd
                    else:
                        flux_terms += -c_in*(uv[0]*self.normal[0]
                                             + uv[1]*self.normal[1])*arg*ds_bnd

        return flux_terms


class TracerHorizontalDiffusionGOErrorEstimatorTerm(TracerGOErrorEstimatorTerm):
    """
    :class:`TracerGOErrorEstimatorTerm` object associated with the :class:`HorizontalDiffusionTerm`
    term of the 2D tracer model.
    """
    def element_residual(self, solution, solution_old, arg, arg_old, fields, fields_old):
        if fields_old.get('diffusivity_h') is None:
            return 0
        diffusivity_h = fields_old['diffusivity_h']
        diff_tensor = as_matrix([[diffusivity_h, 0, ],
                                 [0, diffusivity_h, ]])

        # Apply SUPG stabilisation
        uv = fields_old.get('uv_2d')
        if not self.horizontal_dg and self.stabilisation == 'supg' and uv is not None:
            self.corr_factor = fields_old.get('tracer_advective_velocity_factor')
            uv = self.corr_factor*uv
            arg = arg + self.supg_stabilisation*dot(uv, grad(arg))

        return self.p0test*arg*div(dot(diff_tensor, grad(solution)))*self.dx

    def inter_element_flux(self, solution, solution_old, arg, arg_old, fields, fields_old):
        if fields_old.get('diffusivity_h') is None:
            return 0
        diffusivity_h = fields_old['diffusivity_h']
        diff_tensor = as_matrix([[diffusivity_h, 0, ],
                                 [0, diffusivity_h, ]])

        flux_terms = 0
        ds_interior = self.dS
        if self.horizontal_dg:  # TODO: Check signs
            alpha = self.sipg_parameter
            assert alpha is not None
            sigma = avg(alpha/self.cellsize)
            arg_n = self.p0test*arg*self.normal('+') + self.p0test*arg*self.normal('-')
            flux_terms += -sigma*inner(arg_n,
                                       dot(avg(diff_tensor),
                                           jump(solution, self.normal)))*ds_interior
            flux_terms += inner(arg_n, avg(dot(diff_tensor, grad(solution))))*ds_interior
            arg_av = self.p0test*0.5*arg
            flux_terms += inner(dot(avg(diff_tensor), grad(arg_av)),
                                jump(solution, self.normal))*ds_interior
        else:
            I = self.p0test*inner(dot(diff_tensor, grad(solution)), arg*self.normal)
            flux_terms += (I('+') + I('-'))*ds_interior
        return flux_terms

    def boundary_flux(self, solution, solution_old, arg, arg_old, fields, fields_old, bnd_conditions):
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
                c_in = solution

                # Ignore open boundaries
                if funcs is None or not ('value' in funcs or 'diff_flux' in funcs):
                    continue

                # Ignore Dirichlet boundaries for CG
                elif 'value' in funcs and not self.horizontal_dg:
                    continue

                # Term from integration by parts
                diff_flux = dot(diff_tensor, grad(c_in))
                flux_terms += self.p0test*inner(diff_flux, arg*self.normal)*ds_bnd

                # Terms from boundary conditions
                elev = fields_old['elev_2d']
                self.corr_factor = fields_old.get('tracer_advective_velocity_factor')
                uv = self.corr_factor * fields_old['uv_2d']
                if 'value' in funcs:
                    c_ext, uv_ext, eta_ext = self.get_bnd_functions(c_in, uv, elev, bnd_marker, bnd_conditions)
                    uv_av = 0.5*(uv + uv_ext)
                    un_av = self.normal[0]*uv_av[0] + self.normal[1]*uv_av[1]
                    s = 0.5*(sign(un_av) + 1.0)
                    c_up = c_in*s + c_ext*(1-s)
                    diff_flux_up = dot(diff_tensor, grad(c_up))
                    flux_terms += -self.p0test*arg*dot(diff_flux_up, self.normal)*ds_bnd
                elif 'diff_flux' in funcs:
                    flux_terms += -self.p0test*arg*funcs['diff_flux']*ds_bnd
        return flux_terms


class TracerSourceGOErrorEstimatorTerm(TracerGOErrorEstimatorTerm):
    """
    :class:`TracerGOErrorEstimatorTerm` object associated with the :class:`SourceTerm` term of the
    2D tracer model.
    """
    def element_residual(self, solution, solution_old, arg, arg_old, fields, fields_old):
        f = 0
        source = fields_old.get('source')
        if source is None:
            return f

        # Apply SUPG stabilisation
        uv = fields_old.get('uv_2d')
        if not self.horizontal_dg and self.stabilisation == 'supg' and uv is not None:
            self.corr_factor = fields_old.get('tracer_advective_velocity_factor')
            uv = self.corr_factor*uv
            arg = arg + self.supg_stabilisation*dot(uv, grad(arg))

        f += self.p0test*inner(source, arg)*self.dx
        return f


class TracerGOErrorEstimator(GOErrorEstimator):
    """
    :class:`GOErrorEstimator` for the 2D tracer model.
    """
    def __init__(self, function_space,
                 depth=None,
                 stabilisation='lax_friedrichs',
                 anisotropic=False,
                 sipg_parameter=Constant(10.0),
                 su_stabilisation=None,
                 supg_stabilisation=None):
        self.stabilisation = stabilisation
        self.su_stabilisation = su_stabilisation
        self.supg_stabilisation = supg_stabilisation
        super(TracerGOErrorEstimator, self).__init__(function_space, anisotropic=anisotropic)
        args = (function_space, depth, stabilisation == 'lax_friedrichs', sipg_parameter)
        self.add_term(TracerHorizontalAdvectionGOErrorEstimatorTerm(*args), 'explicit')
        self.add_term(TracerHorizontalDiffusionGOErrorEstimatorTerm(*args), 'explicit')
        self.add_term(TracerSourceGOErrorEstimatorTerm(*args), 'source')

    def add_term(self, term, label):
        """
        Add :class:`term` to the equation as a :str:`label` type term.

        Also, pass over the chosen cell size measure and any stabilisation parameters.
        """
        super(TracerGOErrorEstimator, self).add_term(term, label)
        key = term.__class__.__name__
        self.terms[key].stabilisation = self.stabilisation
        self.terms[key].su_stabilisation = self.su_stabilisation
        self.terms[key].supg_stabilisation = self.supg_stabilisation

    def mass_term(self, solution, arg, velocity=None):
        """
        Account for SUPG stabilisation in mass term.
        """
        if self.stabilisation == 'supg':
            arg = arg + self.supg_stabilisation*dot(velocity, grad(arg))
        return self.p0test*inner(solution, arg)*dx

    def setup_strong_residual(self, label, solution, solution_old, fields, fields_old):
        adj = Function(self.P0).assign(1.0)
        args = (solution, solution_old, adj, adj, fields, fields_old)
        self.strong_residual_terms = 0
        for term in self.select_terms(label):
            self.strong_residual_terms += term.element_residual(*args)
        self._strong_residual = Function(self.P0, name="Strong residual")

    @property
    def strong_residual(self):
        """
        Evaluate strong residual of 2D tracer equation.
        """
        self._strong_residual.assign(assemble(self.strong_residual_terms))
        return self._strong_residual
