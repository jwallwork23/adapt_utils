# TODO: doc
from thetis.equation import *
from thetis.tracer_eq_2d import TracerTerm
from thetis.utility import *


__all__ = ["AdjointTracerEquation2D", "AdjointConservativeTracerEquation2D"]


# --- Base classes

class AdjointTracerTerm(TracerTerm):

    def get_bnd_functions(self, c_star_in, uv_in, elev_in, bnd_id, bnd_conditions):
        funcs = bnd_conditions.get(bnd_id)

        # Boundary conditions on hydrodynamics as in forward model
        if 'elev' in funcs:
            elev_ext = funcs['elev']
        else:
            elev_ext = elev_in
        if 'uv' in funcs:
            uv_ext = self.corr_factor * funcs['uv']
        elif 'flux' in funcs:
            h_ext = self.depth.get_total_depth(elev_ext)
            area = h_ext*self.boundary_len[bnd_id]  # NOTE using external data only
            uv_ext = self.corr_factor * funcs['flux']/area*self.normal
        elif 'un' in funcs:
            uv_ext = funcs['un']*self.normal
        else:
            uv_ext = uv_in

        # Boundary conditions on adjoint concentration  # FIXME: will not work due to diff term
        if 'diff_flux' not in funcs:
            c_ext = Constant(0.0)
        elif 'value' not in funcs:
            nu = fields_old.get('diffusivity_h')
            if nu is None:
                nu = Constant(0.0)
            c_ext = -nu*dot(uv_ext, self.normal)
        else:
            c_ext = c_in

        return c_ext, uv_ext, elev_ext


class AdjointConservativeTracerTerm(AdjointTracerTerm):
    """Following current Thetis implementation, this is no different from :class:`AdjointTracerTerm`."""


# --- Terms for adjoint of non-conservative form

class AdjointAdvectionTerm(AdjointTracerTerm):
    # TODO: doc
    def residual(self, solution, solution_old, fields, fields_old, bnd_conditions=None):
        raise NotImplementedError  # TODO


class AdjointDiffusionTerm(AdjointTracerTerm):
    # TODO: doc
    def residual(self, solution, solution_old, fields, fields_old, bnd_conditions=None):
        raise NotImplementedError  # TODO: May need fully reimplementing to account for BCs


class AdjointSourceTerm(AdjointTracerTerm):
    r"""
    Term on the right hand side of the adjoint equation corresponding to the derivative of the quantity
    of interest :math:`J`.
    """
    def residual(self, solution, solution_old, fields, fields_old, bnd_conditions=None):
        f = 0

        dJdc = fields_old.get('dJdc')
        if dJdc is not None:
            f += -inner(dJdc, self.test)*self.dx

        return -f


class AdjointSinkTerm(AdjointTracerTerm):
    # TODO: doc
    def residual(self, solution, solution_old, fields, fields_old, bnd_conditions=None):
        raise NotImplementedError  # TODO


# --- Terms for adjoint of conservative form

class AdjointConservativeAdvectionTerm(AdjointConservativeTracerTerm):
    # TODO: doc
    def residual(self, solution, solution_old, fields, fields_old, bnd_conditions=None):
        raise NotImplementedError  # TODO


class AdjointConservativeDiffusionTerm(AdjointConservativeTracerTerm):
    # TODO: doc
    def residual(self, solution, solution_old, fields, fields_old, bnd_conditions=None):
        raise NotImplementedError  # TODO


class AdjointConservativeSourceTerm(AdjointConservativeTracerTerm):
    # TODO: doc
    def residual(self, solution, solution_old, fields, fields_old, bnd_conditions=None):
        raise NotImplementedError  # TODO


class AdjointConservativeSinkTerm(AdjointConservativeTracerTerm):
    # TODO: doc
    def residual(self, solution, solution_old, fields, fields_old, bnd_conditions=None):
        raise NotImplementedError  # TODO


# --- Equations

class AdjointTracerEquation2D(Equation):
    # TODO: doc
    def __init__(self, function_space, depth,
                 use_lax_friedrichs=False,
                 sipg_parameter=Constant(10.0)):
        """
        :arg function_space: :class:`FunctionSpace` where the solution belongs
        :arg depth: :class: `DepthExpression` containing depth info
        :kwarg bool use_lax_friedrichs: whether to use Lax Friedrichs stabilisation
        :kwarg sipg_parameter: :class: `Constant` or :class: `Function` penalty parameter for SIPG
        """
        super(AdjointTracerEquation2D, self).__init__(function_space)

        args = (function_space, depth, use_lax_friedrichs, sipg_parameter)
        self.add_term(AdjointHorizontalAdvectionTerm(*args), 'explicit')
        self.add_term(AdjointHorizontalDiffusionTerm(*args), 'explicit')
        self.add_term(AdjointSourceTerm(*args), 'source')
        self.add_term(AdjointSinkTerm(*args), 'source')


class AdjointConservativeTracerEquation2D(Equation):
    # TODO: doc
    def __init__(self, function_space, depth,
                 use_lax_friedrichs=False,
                 sipg_parameter=Constant(10.0)):
        """
        :arg function_space: :class:`FunctionSpace` where the solution belongs
        :arg depth: :class: `DepthExpression` containing depth info
        :kwarg bool use_lax_friedrichs: whether to use Lax Friedrichs stabilisation
        :kwarg sipg_parameter: :class: `Constant` or :class: `Function` penalty parameter for SIPG
        """
        super(AdjointConservativeTracerEquation2D, self).__init__(function_space)

        args = (function_space, depth, use_lax_friedrichs, sipg_parameter)
        self.add_term(AdjointConservativeHorizontalAdvectionTerm(*args), 'explicit')
        self.add_term(AdjointConservativeHorizontalDiffusionTerm(*args), 'explicit')
        self.add_term(AdjointConservativeSourceTerm(*args), 'source')
        self.add_term(AdjointConservativeSinkTerm(*args), 'source')
