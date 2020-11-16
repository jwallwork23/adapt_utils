"""
3D non-conservative and conservative tracer equations. Note that only the CG case is considered.

**********************************************************************************************
*  NOTE: This file is based on the Thetis project (https://thetisproject.org) and contains   *
*        some copied code.                                                                   *
**********************************************************************************************
"""
from __future__ import absolute_import
from thetis.utility import *
from .equation import *


# --- Terms

class HorizontalAdvectionTerm3D(HorizontalAdvectionTerm):
    """
    Horizontal advection term for the 3D tracer model.
    """
    def residual(self, solution, solution_old, fields, fields_old, bnd_conditions=None):
        assert not self.horizontal_dg
        f = 0
        uv = fields_old.get('uv_3d')
        if uv is None:
            return -f
        f += self.test*dot(uv, grad(solution))*dx

        # Apply SU / SUPG stabilisation
        if self.stabilisation in ('su', 'supg'):
            assert hasattr(self, 'tau')
            f += self.tau*dot(uv, grad(self.test))*dot(uv, grad(solution))*dx

        return -f


class HorizontalDiffusionTerm3D(HorizontalDiffusionTerm):
    """
    Horizontal diffusion term for the 3D tracer model.
    """
    def residual(self, solution, solution_old, fields, fields_old, bnd_conditions=None):
        assert not self.horizontal_dg
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
        if self.stabilisation == 'supg' and uv is not None:
            assert hasattr(self, 'tau')
            f += -self.tau*dot(uv, grad(self.test))*div(dot(diff_tensor, grad(solution)))*dx

        return -f


class SourceTerm3D(SourceTerm):
    """
    Source term for the 3D tracer model.
    """
    def residual(self, solution, solution_old, fields, fields_old, bnd_conditions=None):
        assert not self.horizontal_dg
        f = 0
        source = fields_old.get('source')
        if source is None:
            return -f
        args = (solution, solution_old, fields, fields_old, )
        f += -super(SourceTerm3D, self).residual(*args, bnd_conditions=bnd_conditions)

        # Apply SUPG stabilisation
        uv = fields_old.get('uv_3d')
        if self.stabilisation == 'supg' and uv is not None:
            assert hasattr(self, 'tau')
            f += -self.tau*dot(uv, grad(self.test))*source*dx

        return -f


class ConservativeHorizontalAdvectionTerm3D(ConservativeHorizontalAdvectionTerm):
    """
    Conservative horizontal advection term for the 3D tracer model.
    """
    def residual(self, solution, solution_old, fields, fields_old, bnd_conditions=None):
        assert not self.horizontal_dg
        f = 0
        uv = fields_old.get('uv_3d')
        if uv is None:
            return -f
        f += self.test*div(uv*solution)*dx

        # Apply SU / SUPG stabilisation
        if self.stabilisation == 'supg':
            assert hasattr(self, 'tau')
            f += self.tau*dot(uv, grad(self.test))*div(uv*solution)*dx
        return -f


class ConservativeHorizontalDiffusionTerm3D(HorizontalDiffusionTerm3D):
    """
    Copied from above.
    """


# --- Equations

class TracerEquation3D(TracerEquation2D):
    """
    3D tracer advection-diffusion equation in non-conservative form.

    NOTE: Only CG discretisations are currently implemented, with SU and SUPG stabilisation options.
    """
    def __init__(self, function_space, depth, anisotropic=False):
        """
        :arg function_space: :class:`FunctionSpace` where the solution belongs
        :arg depth: :class: `DepthExpression` containing depth info
        :kwarg anisotropic: toggle anisotropic cell size measure
        """
        if function_space.ufl_element().family() != 'Lagrange':
            raise NotImplementedError  # TODO
        super(TracerEquation3D, self).__init__(function_space, depth, anisotropic=anisotropic)

    def add_terms(self, function_space, depth, *unused):
        args = (function_space, depth)
        self.add_term(HorizontalAdvectionTerm3D(*args), 'explicit')
        self.add_term(HorizontalDiffusionTerm3D(*args), 'explicit')
        self.add_term(SourceTerm3D(*args), 'source')


class ConservativeTracerEquation3D(TracerEquation3D):
    """
    3D tracer advection-diffusion equation in conservative form.

    NOTE: Only CG discretisations are currently implemented, with SU and SUPG stabilisation options.
    """
    def add_terms(self, function_space, depth, *unused):
        args = (function_space, depth)
        self.add_term(ConservativeHorizontalAdvectionTerm3D(*args), 'explicit')
        self.add_term(ConservativeHorizontalDiffusionTerm3D(*args), 'explicit')
        self.add_term(SourceTerm3D(*args), 'source')
