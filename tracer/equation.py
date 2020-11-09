"""
2D non-conservative and conservative tracer equations as in Thetis, with a few minor modifications:
    1. Allow for CG discretisations and either SU or SUPG stabilisation.
    2. Account for mesh movement under a prescribed mesh velocity.
    3. Enable choice of anisotropic cell size measure.
"""
from __future__ import absolute_import
from thetis.utility import *
import thetis.tracer_eq_2d as thetis_tracer
import thetis.conservative_tracer_eq_2d as thetis_cons_tracer
from ..equation import Equation


# --- Modified terms for the non-conservative form

class HorizontalAdvectionTerm(thetis_tracer.HorizontalAdvectionTerm):
    """
    Horizontal advection term from Thetis, modified to allow for mesh movement and CG
    discretisations, with SU and SUPG stabilisation options.
    """
    def residual(self, solution, solution_old, fields, fields_old, bnd_conditions=None):
        f = 0

        # Account for mesh movement
        mesh_velocity = fields_old.get('mesh_velocity')  # TODO: Make more robust
        if mesh_velocity is not None:
            f += (Dx(mesh_velocity[0]*self.test, 0)*solution
                  + Dx(mesh_velocity[1]*self.test, 1)*solution)*self.dx

        if self.horizontal_dg:
            args = (solution, solution_old, fields, fields_old, )
            f += -super(HorizontalAdvectionTerm, self).residual(*args, bnd_conditions=bnd_conditions)
        else:
            uv = fields_old.get('uv_2d')
            if uv is None:
                return -f
            f += self.test*dot(uv, grad(solution))*dx

            # Apply SU / SUPG stabilisation
            tau = fields.get('su_stabilisation')
            if tau is None:
                tau = fields.get('supg_stabilisation')
            if tau is not None:
                h = self.cellsize
                unorm = sqrt(dot(uv, uv))
                tau = 0.5*h/unorm
                diffusivity_h = fields_old['diffusivity_h']
                if diffusivity_h is not None:
                    Pe = 0.5*h*unorm/diffusivity_h
                    tau *= min_value(1, Pe/3)
                f += tau*dot(uv, grad(self.test))*dot(uv, grad(solution))*dx

        return -f


class HorizontalDiffusionTerm(thetis_tracer.HorizontalDiffusionTerm):
    """
    Horizontal diffusion term from Thetis, modified to allow for CG discretisations, with SU and
    SUPG stabilisation options.
    """
    def residual(self, solution, solution_old, fields, fields_old, bnd_conditions=None):
        f = 0
        if fields_old.get('diffusivity_h') is None:
            return -f
        if self.horizontal_dg:
            args = (solution, solution_old, fields, fields_old, )
            f += -super(HorizontalDiffusionTerm, self).residual(*args, bnd_conditions=bnd_conditions)
        else:

            # Get diffusion tensor etc.
            diffusivity_h = fields_old['diffusivity_h']
            diff_tensor = as_matrix([[diffusivity_h, 0, ],
                                     [0, diffusivity_h, ]])
            diff_flux = dot(diff_tensor, grad(solution))

            # Element interior term
            f += inner(grad(self.test), diff_flux)*self.dx

            # Apply boundary conditions
            if bnd_conditions is not None:
                for bnd_marker in self.boundary_markers:
                    funcs = bnd_conditions.get(bnd_marker)
                    ds_bnd = ds(int(bnd_marker), degree=self.quad_degree)
                if funcs is not None:
                    if 'diff_flux' in funcs:
                        f += -self.test*funcs['diff_flux']*ds_bnd
                    else:
                        f += -self.test*solution*ds_bnd

            # Apply SUPG stabilisation
            uv = fields_old.get('uv_2d')
            if uv is None:
                return -f
            tau = fields.get('supg_stabilisation')
            if tau is not None:
                h = self.cellsize
                unorm = sqrt(dot(uv, uv))
                tau = 0.5*h/unorm
                Pe = 0.5*h*unorm/diffusivity_h
                tau *= min_value(1, Pe/3)
                f += -tau*dot(uv, grad(self.test))*div(dot(diff_tensor, grad(solution)))*dx

        return -f


class SourceTerm(thetis_tracer.SourceTerm):
    """
    Source term from Thetis, modified to allow for CG discretisations, with SU and SUPG stabilisation
    options.
    """
    def residual(self, solution, solution_old, fields, fields_old, bnd_conditions=None):
        f = 0
        source = fields_old.get('source')
        if source is None:
            return -f
        args = (solution, solution_old, fields, fields_old, )
        f += -super(SourceTerm, self).residual(*args, bnd_conditions=bnd_conditions)

        # Apply SUPG stabilisation
        if not self.horizontal_dg:
            uv = fields_old.get('uv_2d')
            if uv is None:
                return -f
            tau = fields.get('supg_stabilisation')
            if tau is not None:
                h = self.cellsize
                unorm = sqrt(dot(uv, uv))
                tau = 0.5*h/unorm
                diffusivity_h = fields_old['diffusivity_h']
                if diffusivity_h is not None:
                    Pe = 0.5*h*unorm/diffusivity_h
                    tau *= min_value(1, Pe/3)
                f += -tau*dot(uv, grad(self.test))*source*dx

        return -f


# --- Modified terms for the conservative form

class ConservativeHorizontalAdvectionTerm(thetis_cons_tracer.ConservativeHorizontalAdvectionTerm):
    """
    Conservative horizontal advection term from Thetis, modified to allow for mesh movement and CG
    discretisations, with SU and SUPG stabilisation options.
    """
    def residual(self, solution, solution_old, fields, fields_old, bnd_conditions=None):
        f = 0

        # Account for mesh movement
        mesh_velocity = fields_old.get('mesh_velocity')  # TODO: Make more robust
        if mesh_velocity is not None:
            f += (Dx(mesh_velocity[0]*self.test, 0)*solution
                  + Dx(mesh_velocity[1]*self.test, 1)*solution)*self.dx

        if self.horizontal_dg:
            args = (solution, solution_old, fields, fields_old, )
            f += -super(ConservativeHorizontalAdvectionTerm, self).residual(*args, bnd_conditions=bnd_conditions)

        else:
            # NOTE: This is a different formulation as for DG!
            uv = fields_old.get('uv_2d')
            if uv is None:
                return -f
            f += self.test*div(uv*solution)*dx

            # Apply SU / SUPG stabilisation
            tau = fields.get('su_stabilisation')
            if tau is None:
                tau = fields.get('supg_stabilisation')
            if tau is not None:
                h = self.cellsize
                unorm = sqrt(dot(uv, uv))
                tau = 0.5*h/unorm
                diffusivity_h = fields_old['diffusivity_h']
                if diffusivity_h is not None:
                    Pe = 0.5*h*unorm/diffusivity_h
                    tau *= min_value(1, Pe/3)
                f += tau*dot(uv, grad(self.test))*div(uv*solution)*dx
        return -f


class ConservativeHorizontalDiffusionTerm(HorizontalDiffusionTerm):
    """
    Copied from above.
    """


# --- Equations

class TracerEquation2D(Equation):
    """
    Copied here from `thetis/tracer_eq_2d` to hook up modified terms.
    """
    def __init__(self, function_space, depth,
                 use_lax_friedrichs=False,
                 sipg_parameter=Constant(10.0),
                 anisotropic=False):
        """
        :arg function_space: :class:`FunctionSpace` where the solution belongs
        :arg depth: :class: `DepthExpression` containing depth info
        :kwarg bool use_lax_friedrichs: whether to use Lax Friedrichs stabilisation
        :kwarg sipg_parameter: :class: `Constant` or :class: `Function` penalty parameter for SIPG
        :kwarg anisotropic: toggle anisotropic cell size measure
        """
        super(TracerEquation2D, self).__init__(function_space, anisotropic=anisotropic)
        args = (function_space, depth)
        kwargs = {
            'use_lax_friedrichs': use_lax_friedrichs,
            'sipg_parameter': sipg_parameter,
        }
        self.add_term(HorizontalAdvectionTerm(*args, **kwargs), 'explicit')
        self.add_term(HorizontalDiffusionTerm(*args, **kwargs), 'explicit')
        self.add_term(SourceTerm(*args, **kwargs), 'source')
        try:
            args = (function_space, depth, use_lax_friedrichs, sipg_parameter)
            self.add_term(thetis_tracer.SinkTerm(*args), 'source')
        except Exception:
            print_output("WARNING: Cannot import SinkTerm.")


class ConservativeTracerEquation2D(Equation):
    """
    Copied here from `thetis/conservative_tracer_eq_2d` to hook up modified terms.
    """
    def __init__(self, function_space, depth,
                 use_lax_friedrichs=False,
                 sipg_parameter=Constant(10.0),
                 anisotropic=False):
        """
        :arg function_space: :class:`FunctionSpace` where the solution belongs
        :arg depth: :class: `DepthExpression` containing depth info
        :kwarg bool use_lax_friedrichs: whether to use Lax Friedrichs stabilisation
        :kwarg sipg_parameter: :class: `Constant` or :class: `Function` penalty parameter for SIPG
        :kwarg anisotropic: toggle anisotropic cell size measure
        """
        super(ConservativeTracerEquation2D, self).__init__(function_space, anisotropic=anisotropic)
        args = (function_space, depth)
        kwargs = {
            'use_lax_friedrichs': use_lax_friedrichs,
            'sipg_parameter': sipg_parameter,
        }
        self.add_term(ConservativeHorizontalAdvectionTerm(*args, **kwargs), 'explicit')
        self.add_term(ConservativeHorizontalDiffusionTerm(*args, **kwargs), 'explicit')
        if self.function_space.ufl_element().family() == 'Lagrange':
            self.add_term(SourceTerm(*args, **kwargs), 'source')
        else:
            self.add_term(thetis_cons_tracer.ConservativeSourceTerm(*args, **kwargs), 'source')
        try:
            args = (function_space, depth, use_lax_friedrichs, sipg_parameter)
            self.add_term(thetis_cons_tracer.ConservativeSinkTerm(*args), 'source')
        except Exception:
            print_output("WARNING: Cannot import ConservativeSinkTerm.")