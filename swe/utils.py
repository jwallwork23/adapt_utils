from firedrake import *

from adapt_utils.adapt.metric import *
from adapt_utils.adapt.recovery import DoubleL2ProjectorHessian
from adapt_utils.swe.options import ShallowWaterOptions
from adapt_utils.misc import get_component, get_component_space


__all__ = ["get_hessian_metric", "ShallowWaterHessianRecoverer",
           "vorticity", "speed", "heaviside_approx"]


def get_hessian_metric(sol, adapt_field, **kwargs):
    """Recover `adapt_field` Hessian from shallow water solution tuple `sol`."""
    op = kwargs.get('op')
    rec = ShallowWaterHessianRecoverer(sol.function_space(), op=op)
    return rec.get_hessian_metric(sol, adapt_field, **kwargs)


class ShallowWaterHessianRecoverer():
    """
    Class which allows the repeated recovery of Hessians for shallow water problems.

    Hessians may be recovered from fields which lie in either the elevation function space or the
    scalar version of the velocity space (i.e. the speed space).

    :arg function_space: :class:`MixedFunctionSpace` instance for prognostic tuple.
    :kwarg constant_fields: dictionary of fields whose Hessian can be computed once per mesh
        iteration
    :kwarg op: :class:`ShallowWaterOptions` parameters class
    """
    def __init__(self, function_space, constant_fields={}, op=ShallowWaterOptions(), **kwargs):
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

    def get_hessian_metric(self, sol, adapt_field=None, fields={}, **kwargs):
        """
        Recover the Hessian of the scalar `adapt_field` related to solution tuple `sol`.

        Currently supported adaptation fields:
          * 'elevation'  - the free surface elevation;
          * 'velocity_x' - the x-component of velocity;
          * 'velocity_y' - the y-component of velocity;
          * 'speed'      - the magnitude of velocity;
          * 'vorticity'  - the fluid vorticity, interpreted as a scalar field.

        Currently supported adaptation fields which are constant in time:
          * 'bathymetry' - the fluid depth at rest.

        Multiple fields can be combined using double-understrokes and either 'avg' for metric
        average or 'int' for metric intersection. We assume distributivity of intersection over
        averaging.

        For example, `adapt_field = 'elevation__avg__velocity_x__int__bathymetry'` would imply
        first intersecting the Hessians recovered from the x-component of velocity and bathymetry
        and then averaging the result with the Hessian recovered from the elevation.

        NOTE: `sol` could be a forward or adjoint solution tuple.
        """
        kwargs.setdefault('normalise', True)
        op = self.op
        kwargs['op'] = op
        adapt_field = adapt_field or op.adapt_field
        u, eta = sol.split()

        # --- Fields based on velocity

        uv_space_fields = {'speed': speed, 'vorticity': vorticity}

        # --- Recover Hessian and construct metric(s)

        # Combine the two velocity components with elevation
        if adapt_field in ('all_avg', 'all_int'):
            c = adapt_field[-3:]
            adapt_field = "velocity_x__{:s}__velocity_y__{:s}__elevation".format(c, c)

        # The list of fields are averaged/intersected, as appropriate
        # If both are specified, the list of fields are first intersected and then averaged
        for c in ('avg', 'int'):
            if c in adapt_field:
                adapt_fields = adapt_field.split('__{:s}__'.format(c))
                metrics = [self.get_hessian_metric(sol, f, fields=fields) for f in adapt_fields]
                return combine_metrics(*metrics, average=c == 'avg')

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


def vorticity(sol):
    """Fluid vorticity, interpreted as a scalar field."""
    assert sol.function_space().mesh().topological_dimension() == 2
    uv, elev = sol.split()
    return curl(uv)


def speed(sol):
    """Fluid velocity magnitude, i.e. fluid speed."""
    uv, elev = sol.split()
    return sqrt(inner(uv, uv))


def heaviside_approx(H, alpha):
    """C0 continuous approximation to Heaviside function."""
    return 0.5*(H/(sqrt(H**2 + alpha**2))) + 0.5
