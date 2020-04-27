from firedrake import project, curl, inner, sqrt

from adapt_utils.adapt.metric import *
from adapt_utils.adapt.recovery import DoubleL2ProjectorHessian
from adapt_utils.swe.options import ShallowWaterOptions


__all__ = ["get_hessian_metric", "vorticity"]


class ShallowWaterHessianRecoverer():
    """
    Class which allows the repeated recovery of Hessians for shallow water problems.

    Hessians may be recovered from fields which lie in either the elevation function space or the
    scalar version of the velocity space (i.e. the speed space).
    """
    def __init__(self, function_space, op=ShallowWaterOptions()):
        self.op = op

        # FunctionSpaces
        self.function_space = function_space
        self.mesh = function_space.mesh()
        uv_space, self.elev_space = function_space.split()
        self.speed_space = FunctionSpace(self.mesh, uv_space.ufl_element())

        # Projectors
        self.speed_projector = DoubleL2ProjectorHessian(speed_space, op=op)
        self.elev_projector = DoubleL2ProjectorHessian(elev_space, op=op)

    def get_hessian_metric(sol, adapt_field, fields={}, **kwargs):
        kwargs.setdefault('noscale', False)
        op = self.op
        kwargs['op'] = op
        u, eta = sol.split()

        # Gather fields
        elev_space_fields = {
            'elevation': eta,
        }
        if 'bathymetry' in fields:
            elev_space_fields['bathymetry'] = fields.pop('bathymetry')
        uv_space_fields = {
            'velocity_x': u[0],
            'velocity_y': u[1],
            'speed': sqrt(inner(u, u)),
        }
        if 'inflow' in fields:
            uv_space_fields['inflow'] = inner(u, fields.pop('inflow'))

        # TODO: TESTME
        if adapt_field == 'elevation':
            M = steady_metric(eta, projector=self.speed_projector, **kwargs)

        # TODO: TESTME
        elif adapt_field in uv_space_fields:
            f = project(uv_space_fields[adapt_field], self.speed_space)
            M = steady_metric(f, projector=self.speed_projector, **kwargs)

        # TODO: TESTME
        elif adapt_field in elev_space_fields:
            f = project(elev_space_fields[adapt_field], self.elev_space)
            M = steady_metric(f, projector=self.elev_projector, **kwargs)

        # TODO: TESTME
        elif adapt_field in ('all_avg', 'all_int'):
            u = project(uv_space_fields['velocity_x'], self.speed_space)
            v = project(uv_space_fields['velocity_y'], self.speed_space)
            eta = elev_space_fields['elevation']
            M_u = steady_metric(u, projector=self.speed_projector, **kwargs)
            M_v = steady_metric(v, projector=self.speed_projector, **kwargs)
            M_eta = steady_metric(eta, projector=self.elev_projector, **kwargs)
            M = combine_metrics(M_u, M_v, M_eta, average=op.adapt_field == 'all_avg')

        # TODO: TESTME int only
        # TODO: TESTME int and avg
        elif 'int' in adapt_field:
            adapt_fields = adapt_field.split('int')
            metrics = [self.get_hessian_metric(sol, field, fields=fields) for field in adapt_fields]
            M = metric_intersection(*metrics)

        # TODO: TESTME
        elif 'avg' in adapt_field:
            adapt_fields = adapt_field.split('avg')
            metrics = [self.get_hessian_metric(sol, field, fields=fields) for field in adapt_fields]
            M = metric_average(*metrics)

        else:
            raise ValueError("Adaptation field {:s} not recognised.".format(adapt_field))
        return M


# TODO: USEME
def vorticity(sol):
    uv, elev = sol.split()
    assert uv.function_space().mesh().topological_dimension() == 2
    return curl(uv)
