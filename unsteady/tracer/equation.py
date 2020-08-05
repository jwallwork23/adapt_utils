"""
2D non-conservative and conservative tracer equations as in Thetis, with a few minor modifications:
    1. Allow for CG discretisations and either SU or SUPG stabilisation.
    2. Account for mesh movement under a prescribed mesh velocity.
"""
from thetis.equation import *
from thetis.utility import *
import thetis.tracer_eq_2d as thetis_tracer
import thetis.conservative_tracer_eq_2d as thetis_cons_tracer


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

        raise NotImplementedError  # TODO: Consider CG case


class HorizontalDiffusionTerm(thetis_tracer.HorizontalDiffusionTerm):
    def residual(self, solution, solution_old, fields, fields_old, bnd_conditions=None):
        if self.horizontal_dg:
            args = (solution, solution_old, fields, fields_old, )
            return super(HorizontalDiffusionTerm, self).residual(*args, bnd_conditions=bnd_conditions)

        raise NotImplementedError  # TODO: Consider CG case


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
            f += -super(HorizontalAdvectionTerm, self).residual(*args, bnd_conditions=bnd_conditions)
            return -f

        raise NotImplementedError  # TODO: Consider CG case


# --- Equations

class TracerEquation2D(Equation):
    """Copied here from `thetis/tracer_eq_2d` to hook up modified terms."""
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
        self.add_term(thetis_tracer.SinkTerm(*args), 'source')


class ConservativeTracerEquation2D(Equation):
    """Copied here from `thetis/conservative_tracer_eq_2d` to hook up modified terms."""
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
        self.add_term(thetis_cons_tracer.ConservativeSinkTerm(*args), 'source')
