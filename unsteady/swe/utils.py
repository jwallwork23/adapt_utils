from __future__ import absolute_import

from firedrake import *
import thetis.utility as thetis_utils

from adapt_utils.adapt.metric import *
from adapt_utils.adapt.recovery import DoubleL2ProjectorHessian, L2Projector
from ..options import CoupledOptions
from adapt_utils.misc import get_component, get_component_space


__all__ = ["recover_hessian_metric", "ShallowWaterHessianRecoverer",
           "recover_vorticity", "recover_vorticity_metric", "L2ProjectorVorticity",
           "speed", "heaviside_approx", "DepthExpression"]


# --- Hessian

def recover_hessian_metric(sol, adapt_field, **kwargs):
    """
    Recover the Hessian of some scalar field Hessian from the shallow water solution tuple.

    :arg adapt_field: string defining the field to be recovered.
    :kwarg op: :class:`Options` parameter object.
    """
    op = kwargs.get('op', CoupledOptions())
    rec = ShallowWaterHessianRecoverer(sol.function_space(), op=op)
    return rec.construct_metric(sol, adapt_field, **kwargs)


class ShallowWaterHessianRecoverer():
    """
    Class which allows the repeated recovery of Hessians for shallow water problems.

    Hessians may be recovered from fields which lie in either the elevation function space or the
    scalar version of the velocity space (i.e. the speed space).

    :arg function_space: :class:`MixedFunctionSpace` instance for prognostic tuple.
    :kwarg constant_fields: dictionary of fields whose Hessian can be computed once per mesh
        iteration
    :kwarg op: :class:`CoupledOptions` parameters class
    """
    def __init__(self, function_space, constant_fields={}, op=CoupledOptions(), **kwargs):
        kwargs.setdefault('normalise', True)
        self.op = op

        # FunctionSpaces
        self.function_space = function_space
        self.mesh = function_space.mesh()
        uv_space, self.elev_space = function_space.split()
        self.speed_space = get_component_space(uv_space)

        # Projectors
        self.speed_projector = DoubleL2ProjectorHessian(self.speed_space, op=op)
        self.elev_projector = DoubleL2ProjectorHessian(self.elev_space, op=op)

        # Compute Hessian for constant fields
        self.constant_hessians = {f: steady_metric(constant_fields[f], **kwargs) for f in constant_fields}

    def construct_metric(self, sol, adapt_field=None, fields={}, **kwargs):
        """
        Recover the Hessian of the scalar `adapt_field` related to solution tuple `sol`.

        Currently supported adaptation fields:
          * 'elevation'  - the free surface elevation;
          * 'velocity_x' - the x-component of velocity;
          * 'velocity_y' - the y-component of velocity;
          * 'speed'      - the magnitude of velocity.

        Currently supported adaptation fields which are constant in time:
          * 'bathymetry' - the fluid depth at rest.

        NOTE: `sol` could be a forward or adjoint solution tuple.
        """
        kwargs.setdefault('normalise', True)
        op = self.op
        kwargs['op'] = op
        adapt_field = adapt_field or op.adapt_field
        u, eta = sol.split()
        uv_space_fields = {'speed': speed}

        # --- Recover Hessian and construct metric(s)

        # Fields which are constant in time
        if adapt_field in self.constant_hessians:
            return self.constant_hessians[adapt_field]

        # Elevation already lives in the correct space
        if adapt_field == 'elevation':
            f = eta
            proj = self.elev_projector

        # Velocity components can be extracted directly
        elif adapt_field == 'velocity_x':
            f = get_component(u, 0, component_space=self.speed_space)
            proj = self.speed_projector
        elif adapt_field == 'velocity_y':
            f = get_component(u, 1, component_space=self.speed_space)
            proj = self.speed_projector

        # Fields which need projecting into speed space
        elif adapt_field in uv_space_fields:
            f = project(uv_space_fields[adapt_field](sol), self.speed_space)
            proj = self.speed_projector
            return steady_metric(f, projector=self.speed_projector, **kwargs)

        else:
            raise ValueError("Adaptation field {:s} not recognised.".format(adapt_field))
        return steady_metric(f, projector=proj, **kwargs)


# --- Vorticity

def recover_vorticity(u, **kwargs):
    return L2ProjectorVorticity(u.function_space(), **kwargs).project(u)


def recover_vorticity_metric(u, **kwargs):
    r"""
    Assuming the velocity field `u` is P1 (piecewise linear and continuous), direct computation of
    the curl will give a vorticity field which is P0 (piecewise constant and discontinuous). Since
    we would prefer a smooth gradient, we solve an auxiliary finite element problem in P1 space.
    This "L2 projection" gradient recovery technique makes use of the Cl\'ement interpolation
    operator. That `f` is P1 is not actually a requirement.

    :arg u: (vector) P1 velocity field.
    :kwarg bcs: boundary conditions for L2 projection.
    :param op: `Options` class object providing min/max cell size values.
    :return: reconstructed vorticity associated with `u`.
    """
    return isotropic_metric(recover_vorticity(u), **kwargs)


class L2ProjectorVorticity(L2Projector):
    def __init__(self, *args, **kwargs):
        super(L2ProjectorVorticity, self).__init__(*args, **kwargs)
        self.op = kwargs.get('op')
        self.kwargs['solver_parameters'] = {'ksp_type': 'cg'}

    def setup(self):
        fs = self.field.function_space()
        assert self.mesh.topological_dimension() == 2
        P1 = FunctionSpace(self.mesh, "CG", 1)
        n = FacetNormal(self.mesh)
        zeta, φ = TrialFunction(P1), TestFunction(P1)
        if hasattr(fs, 'shape'):
            assert fs.shape == (2, ), "Expected fs.shape == (2, ), got {:}.".format(fs.shape)
            uv = self.field
        elif hasattr(fs, 'num_sub_spaces'):
            num_sub_spaces = fs.num_sub_spaces()
            assert num_sub_spaces == 2, "Expected 2 subspaces, got {:d}.".format(num_sub_spaces)
            uv, elev = self.field.split()
        else:
            raise ValueError("Expected a mixed solution tuple or the velocity component thereof.")

        a = φ*zeta*dx
        L = (Dx(φ, 1)*uv[0] - Dx(φ, 0)*uv[1])*dx + (φ*uv[1]*n[0] - φ*uv[0]*n[1])*ds
        self.l2_projection = Function(P1, name="Recovered vorticity")
        prob = LinearVariationalProblem(a, L, self.l2_projection, bcs=self.bcs)
        self.projector = LinearVariationalSolver(prob, **self.kwargs)

    def construct_metric(self, sol, **kwargs):
        kwargs['op'] = self.op
        assert kwargs.pop('adapt_field') == 'vorticity'
        kwargs.pop('fields')
        self.project(sol)
        return isotropic_metric(self.l2_projection, **kwargs)


# --- Misc

def speed(sol):
    """Fluid velocity magnitude, i.e. fluid speed."""
    uv, elev = sol.split()
    return sqrt(inner(uv, uv))


def heaviside_approx(H, alpha):
    """C0 continuous approximation to Heaviside function."""
    return 0.5*(H/(sqrt(H**2 + alpha**2))) + 0.5


class DepthExpression(thetis_utils.DepthExpression):
    """
    Depth expression from Thetis modified to include an approximation to the Heaviside function.
    """
    def heaviside_approx(self, eta):
        return heaviside_approx(self.get_total_depth(eta), self.wetting_and_drying_alpha)
