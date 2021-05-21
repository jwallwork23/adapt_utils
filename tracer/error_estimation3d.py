"""
Extension of the goal-oriented error indicators in `tracer/error_estimation.py` to the 3D case.
"""
from __future__ import absolute_import
from thetis.utility import *
from .error_estimation import TracerGOErrorEstimatorTerm, TracerGOErrorEstimator


__all__ = ['TracerGOErrorEstimator3D']


g_grav = physical_constants['g_grav']


class TracerGOErrorEstimatorTerm3D(TracerGOErrorEstimatorTerm):
    def __init__(self, *args, **kwargs):
        super(TracerGOErrorEstimatorTerm3D, self).__init__(*args, **kwargs)
        assert not self.horizontal_dg


class TracerHorizontalAdvectionGOErrorEstimatorTerm3D(TracerGOErrorEstimatorTerm3D):
    """
    :class:`TracerGOErrorEstimatorTerm3D` object associated with the
    :class:`HorizontalAdvectionTerm3D` term of the 3D tracer model.
    """
    def element_residual(self, solution, solution_old, arg, arg_old, fields, fields_old):
        if fields_old.get('uv_3d') is None:
            return 0
        self.corr_factor = fields_old.get('tracer_advective_velocity_factor')
        uv = self.corr_factor*fields_old['uv_3d']

        # Apply SUPG stabilisation
        if self.options.use_supg_tracer:
            arg = arg + self.supg_stabilisation*dot(uv, grad(arg))

        return -self.p0test*arg*inner(uv, grad(solution))*self.dx


class TracerHorizontalDiffusionGOErrorEstimatorTerm3D(TracerGOErrorEstimatorTerm3D):
    """
    :class:`TracerGOErrorEstimatorTerm3D` object associated with the
    :class:`HorizontalDiffusionTerm3D` term of the 3D tracer model.
    """
    def element_residual(self, solution, solution_old, arg, arg_old, fields, fields_old):
        if fields_old.get('diffusivity_h') is None:
            return 0
        diffusivity_h = fields_old['diffusivity_h']
        diff_tensor = as_matrix([[diffusivity_h, 0, 0, ],
                                 [0, diffusivity_h, 0, ],
                                 [0, 0, diffusivity_h, ]])

        # Apply SUPG stabilisation
        uv = fields_old.get('uv_3d')
        if self.options.use_supg_tracer and uv is not None:
            self.corr_factor = fields_old.get('tracer_advective_velocity_factor')
            uv = self.corr_factor*uv
            arg = arg + self.supg_stabilisation*dot(uv, grad(arg))

        return self.p0test*arg*div(dot(diff_tensor, grad(solution)))*self.dx

    def inter_element_flux(self, solution, solution_old, arg, arg_old, fields, fields_old):
        if fields_old.get('diffusivity_h') is None:
            return 0
        diffusivity_h = fields_old['diffusivity_h']
        diff_tensor = as_matrix([[diffusivity_h, 0, 0, ],
                                 [0, diffusivity_h, 0, ],
                                 [0, 0, diffusivity_h, ]])

        flux_terms = 0
        I = self.p0test*inner(dot(diff_tensor, grad(solution)), arg*self.normal)
        flux_terms += (I('+') + I('-'))*self.dS
        return flux_terms

    def boundary_flux(self, solution, solution_old, arg, arg_old, fields, fields_old, bnd_conditions):
        flux_terms = 0
        if fields_old.get('diffusivity_h') is None:
            return flux_terms
        diffusivity_h = fields_old['diffusivity_h']
        diff_tensor = as_matrix([[diffusivity_h, 0, 0, ],
                                 [0, diffusivity_h, 0, ],
                                 [0, 0, diffusivity_h, ]])

        if bnd_conditions is not None:
            for bnd_marker in self.boundary_markers:
                funcs = bnd_conditions.get(bnd_marker)
                ds_bnd = ds(int(bnd_marker), degree=self.quad_degree)
                c_in = solution

                # Ignore open boundaries
                if funcs is None or not ('value' in funcs or 'diff_flux' in funcs):
                    continue

                # Ignore Dirichlet boundaries for CG
                elif 'value' in funcs:
                    continue

                # Term from integration by parts
                diff_flux = dot(diff_tensor, grad(c_in))
                flux_terms += self.p0test*inner(diff_flux, arg*self.normal)*ds_bnd

                # Terms from boundary conditions
                elev = fields_old['elev_3d']
                self.corr_factor = fields_old.get('tracer_advective_velocity_factor')
                uv = self.corr_factor * fields_old['uv_3d']
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


class TracerSourceGOErrorEstimatorTerm3D(TracerGOErrorEstimatorTerm3D):
    """
    :class:`TracerGOErrorEstimatorTerm3D` object associated with the :class:`SourceTerm3D` term of
    the 3D tracer model.
    """
    def element_residual(self, solution, solution_old, arg, arg_old, fields, fields_old):
        f = 0
        source = fields_old.get('source')
        if source is None:
            return f

        # Apply SUPG stabilisation
        uv = fields_old.get('uv_3d')
        if self.options.use_supg_tracer and uv is not None:
            self.corr_factor = fields_old.get('tracer_advective_velocity_factor')
            uv = self.corr_factor*uv
            arg = arg + self.supg_stabilisation*dot(uv, grad(arg))

        f += self.p0test*inner(source, arg)*self.dx
        return f


class TracerGOErrorEstimator3D(TracerGOErrorEstimator):
    """
    :class:`GOErrorEstimator` for the 3D tracer model.
    """
    def add_terms(self, *args):
        self.add_term(TracerHorizontalAdvectionGOErrorEstimatorTerm3D(*args), 'explicit')
        self.add_term(TracerHorizontalDiffusionGOErrorEstimatorTerm3D(*args), 'explicit')
        self.add_term(TracerSourceGOErrorEstimatorTerm3D(*args), 'source')
