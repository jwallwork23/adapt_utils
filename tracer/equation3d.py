"""
3D non-conservative and conservative tracer equations. Note that only the CG case is considered.
"""
from thetis.equation import *
from thetis.utility import *

from adapt_utils.tracer.equation import *


# --- Terms

class HorizontalAdvectionTerm3D(HorizontalAdvectionTerm):
    # TODO: doc
    def residual(self, solution, solution_old, fields, fields_old, bnd_conditions=None):
        f = 0
        uv = fields_old.get('uv_3d')
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


class HorizontalDiffusionTerm3D(HorizontalDiffusionTerm):
    # TODO: doc
    def residual(self, solution, solution_old, fields, fields_old, bnd_conditions=None):
        f = 0
        if fields_old.get('diffusivity_h') is None:
            return -f

        # Get diffusion tensor etc.
        diffusivity_h = fields_old['diffusivity_h']
        diff_tensor = as_matrix([[diffusivity_h, 0, 0, ],
                                 [0, diffusivity_h, 0, ],
                                 [0, 0, diffusivity_h, ]])
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
        uv = fields_old.get('uv_3d')
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


class SourceTerm3D(SourceTerm):
    # TODO: doc
    def residual(self, solution, solution_old, fields, fields_old, bnd_conditions=None):
        f = 0
        source = fields_old.get('source')
        if source is None:
            return -f
        args = (solution, solution_old, fields, fields_old, )
        f += -super(SourceTerm3D, self).residual(*args, bnd_conditions=bnd_conditions)

        # Apply SUPG stabilisation
        uv = fields_old.get('uv_3d')
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


class ConservativeHorizontalAdvectionTerm3D(ConservativeHorizontalAdvectionTerm):
    # TODO: doc
    def residual(self, solution, solution_old, fields, fields_old, bnd_conditions=None):
        f = 0
        uv = fields_old.get('uv_3d')
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


class ConservativeHorizontalDiffusionTerm3D(HorizontalDiffusionTerm3D):
    """
    Copied from above.
    """


# --- Equations

class TracerEquation3D(Equation):
    # TODO: doc
    def __init__(self, function_space, depth, anisotropic=False):
        """
        :arg function_space: :class:`FunctionSpace` where the solution belongs
        :arg depth: :class: `DepthExpression` containing depth info
        """
        super(TracerEquation3D, self).__init__(function_space)
        if self.function_space.ufl_element().family() != 'Lagrange':
            raise NotImplementedError  # TODO
        args = (function_space, depth)
        kwargs = {
            'anisotropic': anisotropic,
        }
        self.add_term(HorizontalAdvectionTerm3D(*args, **kwargs), 'explicit')
        self.add_term(HorizontalDiffusionTerm3D(*args, **kwargs), 'explicit')
        self.add_term(SourceTerm3D(*args, **kwargs), 'source')


class ConservativeTracerEquation3D(Equation):
    # TODO: doc
    def __init__(self, function_space, depth, anisotropic=False):
        """
        :arg function_space: :class:`FunctionSpace` where the solution belongs
        :arg depth: :class: `DepthExpression` containing depth info
        """
        super(ConservativeTracerEquation3D, self).__init__(function_space)
        if self.function_space.ufl_element().family() != 'Lagrange':
            raise NotImplementedError  # TODO
        args = (function_space, depth)
        kwargs = {
            'anisotropic': anisotropic,
        }
        self.add_term(ConservativeHorizontalAdvectionTerm(*args, **kwargs), 'explicit')
        self.add_term(ConservativeHorizontalDiffusionTerm(*args, **kwargs), 'explicit')
        self.add_term(SourceTerm(*args, **kwargs), 'source')
