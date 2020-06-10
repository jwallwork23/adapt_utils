from __future__ import absolute_import
from thetis.utility import *
from adapt_utils.error_estimation import GOErrorEstimatorTerm, GOErrorEstimator
from thetis.tracer_eq_2d import TracerTerm


__all__ = ['TracerGOErrorEstimator']


g_grav = physical_constants['g_grav']


class TracerGOErrorEstimatorTerm(GOErrorEstimatorTerm, TracerTerm):
    """
    Generic :class:`GOErrorEstimatorTerm` term in a goal-oriented error estimator for the 2D tracer
    model.
    """
    def __init__(self, function_space,
                 depth=None, use_lax_friedrichs=True, sipg_parameter=Constant(10.0)):
        """
        :arg function_space: :class:`FunctionSpace` where the solution belongs
        :kwarg depth: DepthExpression for the domain
        """
        TracerTerm.__init__(self, function_space, depth, use_lax_friedrichs, sipg_parameter)
        GOErrorEstimatorTerm.__init__(self, function_space.mesh())


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
        if self.horizontal_dg:
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

        return self.p0test*arg*div(dot(diff_tensor, grad(solution)))*self.dx

    def inter_element_flux(self, solution, solution_old, arg, arg_old, fields, fields_old):
        if fields_old.get('diffusivity_h') is None:
            return 0
        diffusivity_h = fields_old['diffusivity_h']
        diff_tensor = as_matrix([[diffusivity_h, 0, ],
                                 [0, diffusivity_h, ]])

        flux_terms = 0
        if self.horizontal_dg:
            alpha = self.sipg_parameter
            assert alpha is not None
            sigma = avg(alpha/self.cellsize)
            ds_interior = self.dS
            arg_n = self.p0test*arg*self.normal('+') + self.p0test*arg*self.normal('-')
            flux_terms += -sigma*inner(arg_n,
                                       dot(avg(diff_tensor),
                                           jump(solution, self.normal)))*ds_interior
            flux_terms += inner(arg_n, avg(dot(diff_tensor, grad(solution))))*ds_interior
            arg_av = self.p0test*0.5*arg
            flux_terms += inner(dot(avg(diff_tensor), grad(arg_av)),
                                jump(solution, self.normal))*ds_interior
        return flux_terms

    def boundary_flux(self, solution, solution_old, arg, arg_old, fields, fields_old, bnd_conditions):
        if fields_old.get('diffusivity_h') is None:
            return 0
        diffusivity_h = fields_old['diffusivity_h']
        diff_tensor = as_matrix([[diffusivity_h, 0, ],
                                 [0, diffusivity_h, ]])

        flux_terms = 0

        if bnd_conditions is not None:
            for bnd_marker in self.boundary_markers:
                funcs = bnd_conditions.get(bnd_marker)
                ds_bnd = ds(int(bnd_marker), degree=self.quad_degree)
                c_in = solution
                elev = fields_old['elev_2d']
                self.corr_factor = fields_old.get('tracer_advective_velocity_factor')
                uv = self.corr_factor * fields_old['uv_2d']
                if funcs is not None:
                    if 'value' in funcs:
                        c_ext, uv_ext, eta_ext = self.get_bnd_functions(c_in, uv, elev, bnd_marker, bnd_conditions)
                        uv_av = 0.5*(uv + uv_ext)
                        un_av = self.normal[0]*uv_av[0] + self.normal[1]*uv_av[1]
                        s = 0.5*(sign(un_av) + 1.0)
                        c_up = c_in*s + c_ext*(1-s)
                        diff_flux_up = dot(diff_tensor, grad(c_up))
                        flux_terms += self.p0test*arg*dot(diff_flux_up, self.normal)*ds_bnd
                    elif 'diff_flux' in funcs:
                        flux_terms += self.p0test*arg*funcs['diff_flux']*ds_bnd
                    else:
                        # Open boundary case
                        flux_terms += self.p0test*arg*dot(diff_flux, self.normal)*ds_bnd
        return flux_terms


class TracerSourceGOErrorEstimatorTerm(TracerGOErrorEstimatorTerm):
    """
    :class:`TracerGOErrorEstimatorTerm` object associated with the :class:`SourceTerm` term of the
    2D tracer model.
    """
    def element_residual(self, solution, solution_old, arg, arg_old, fields, fields_old):
        f = 0
        source = fields_old.get('source')
        if source is not None:
            f += -self.p0test*inner(source, self.test)*self.dx
        return -f

    def inter_element_flux(self, solution, solution_old, arg, arg_old, fields, fields_old):
        return 0

    def boundary_flux(self, solution, solution_old, arg, arg_old, fields, fields_old, bnd_conditions):
        return 0


class TracerGOErrorEstimator(GOErrorEstimator):
    """
    :class:`GOErrorEstimator` for the 2D tracer model.
    """
    def __init__(self, function_space,
                 depth=None, use_lax_friedrichs=True, sipg_parameter=Constant(10.0)):
        super(TracerGOErrorEstimator, self).__init__(function_space)

        args = (function_space, depth, use_lax_friedrichs, sipg_parameter)
        self.add_term(TracerHorizontalAdvectionGOErrorEstimatorTerm(*args), 'explicit')
        self.add_term(TracerHorizontalDiffusionGOErrorEstimatorTerm(*args), 'explicit')
        self.add_term(TracerSourceGOErrorEstimatorTerm(*args), 'source')

    def setup_strong_residual(self, label, solution, solution_old, fields, fields_old):
        adj = Function(self.P0_2d).assign(1.0)
        args = (solution, solution_old, adj, adj, fields, fields_old)
        self.strong_residual_terms = 0
        for term in self.select_terms(label):
            self.strong_residual_terms += term.element_residual(*args)
        self.strong_residual = Function(self.P0_2d, name="Strong residual")

    def evaluate_strong_residual(self):
        """Evaluate strong residual of 2D tracer equation."""
        self.strong_residual.assign(assemble(self.strong_residual_terms))
        return self.strong_residual

    def evaluate_flux_jump(self, sol):
        """Evaluate flux jump as element-wise indicator functions."""
        flux_jump = Function(VectorFunctionSpace(self.mesh, "DG", 0)*self.P0_2d)
        solve(self.p0test*self.p0trial*dx == jump(self.p0test*sol)*dS, flux_jump)
        return flux_jump