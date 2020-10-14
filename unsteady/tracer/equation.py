"""
2D non-conservative and conservative tracer equations as in Thetis, with a few minor modifications:
    1. Allow for CG discretisations and either SU or SUPG stabilisation.
    2. Account for mesh movement under a prescribed mesh velocity.
"""
from thetis.equation import *
from thetis.utility import *
import thetis.tracer_eq_2d as thetis_tracer
import thetis.conservative_tracer_eq_2d as thetis_cons_tracer


# TODO: SU stabilisation
# TODO: SUPG stabilisation

# --- Modified terms for the non-conservative form

class HorizontalAdvectionTerm(thetis_tracer.HorizontalAdvectionTerm):
    def residual(self, solution, solution_old, fields, fields_old, bnd_conditions=None):
        f = 0

        # Account for mesh movement
        mesh_velocity = fields_old.get('mesh_velocity')  # TODO: Make more robust
        if mesh_velocity is not None:
            f += (Dx(mesh_velocity[0]*self.test, 0)*solution
                  + Dx(mesh_velocity[1]*self.test, 1)*solution)*self.dx

        # DG tracers
        if self.horizontal_dg:
            args = (solution, solution_old, fields, fields_old, )
            f += -super(HorizontalAdvectionTerm, self).residual(*args, bnd_conditions=bnd_conditions)
            return -f
        else:
            uv = fields_old.get('uv_2d')
            if uv is None:
                return -f
            f += self.test*dot(uv, grad(solution))*dx
            return -f


class HorizontalDiffusionTerm(thetis_tracer.HorizontalDiffusionTerm):
    def residual(self, solution, solution_old, fields, fields_old, bnd_conditions=None):
        if fields_old.get('diffusivity_h') is None:
            return 0
        if self.horizontal_dg:
            args = (solution, solution_old, fields, fields_old, )
            return super(HorizontalDiffusionTerm, self).residual(*args, bnd_conditions=bnd_conditions)
        else:
            diffusivity_h = fields_old['diffusivity_h']
            diff_tensor = as_matrix([[diffusivity_h, 0, ],
                                     [0, diffusivity_h, ]])
            diff_flux = dot(diff_tensor, grad(solution))

            f = 0
            f += inner(grad(self.test), diff_flux)*self.dx

            if bnd_conditions is not None:
                for bnd_marker in self.boundary_markers:
                    funcs = bnd_conditions.get(bnd_marker)
                    ds_bnd = ds(int(bnd_marker), degree=self.quad_degree)
                if funcs is not None:
                    if 'diff_flux' in funcs:
                        f += -self.test*funcs['diff_flux']*ds_bnd
                    else:
                        f += -self.test*solution*ds_bnd
            return -f


# --- Modified terms for the conservative form

class ConservativeHorizontalAdvectionTerm(thetis_cons_tracer.ConservativeHorizontalAdvectionTerm):
    def residual(self, solution, solution_old, fields, fields_old, bnd_conditions=None):
        f = 0

        # Account for mesh movement
        mesh_velocity = fields_old.get('mesh_velocity')  # TODO: Make more robust
        if mesh_velocity is not None:
            f += (Dx(mesh_velocity[0]*self.test, 0)*solution
                  + Dx(mesh_velocity[1]*self.test, 1)*solution)*self.dx

        # DG tracers
        if self.horizontal_dg:
            args = (solution, solution_old, fields, fields_old, )
            f += -super(ConservativeHorizontalAdvectionTerm, self).residual(*args, bnd_conditions=bnd_conditions)
            return -f

        raise NotImplementedError  # TODO: Consider CG case


# --- Equations

class TracerEquation2D(Equation):
    """
    Copied here from `thetis/tracer_eq_2d` to hook up modified terms.
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
        super(TracerEquation2D, self).__init__(function_space)

        args = (function_space, depth, use_lax_friedrichs, sipg_parameter)
        self.add_term(HorizontalAdvectionTerm(*args), 'explicit')
        self.add_term(HorizontalDiffusionTerm(*args), 'explicit')
        self.add_term(thetis_tracer.SourceTerm(*args), 'source')
        try:
            self.add_term(thetis_tracer.SinkTerm(*args), 'source')
        except Exception:
            print_output("WARNING: Cannot import SinkTerm.")


class ConservativeTracerEquation2D(Equation):
    """
    Copied here from `thetis/conservative_tracer_eq_2d` to hook up modified terms.
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
        super(ConservativeTracerEquation2D, self).__init__(function_space)
        args = (function_space, depth, use_lax_friedrichs, sipg_parameter)
        self.add_term(ConservativeHorizontalAdvectionTerm(*args), 'explicit')
        self.add_term(thetis_cons_tracer.ConservativeHorizontalDiffusionTerm(*args), 'explicit')
        self.add_term(thetis_cons_tracer.ConservativeSourceTerm(*args), 'source')
        try:
            self.add_term(thetis_cons_tracer.ConservativeSinkTerm(*args), 'source')
        except Exception:
            print_output("WARNING: Cannot import ConservativeSinkTerm.")
