from thetis import *

from adapt.options import DefaultOptions


__all__ = ["adapt_mesh"]


def adapt_mesh(solver_obj, error_indicator=None, op=DefaultOptions()):
    """
    Adapt mesh based on an error indicator or field of interest.

    :param solver_obj: Thetis ``FlowSolver2d`` object.
    :param error_indicator: optional error indicator upon which to adapt.
    :param op: `AdaptOptions` parameter class.
    :return: adapted mesh.
    """
    mesh2d = solver_obj.mesh2d()
    op.target_vertices = mesh2d.num_vertices() * op.rescaling
    P1 = FunctionSpace(mesh2d, "CG", 1)

    if op.approach == 'HessianBased':
        if solver_obj.options.tracer_only:
            M = steady_metric(project(solver_obj.fields.tracer_2d, P1), op=op)
        else:
            uv_2d, elev_2d = solver_obj.fields.solution_2d.split()
            if op.adapt_field != 'f':       # Metric for fluid speed
                M = steady_metric(project(sqrt(inner(uv_2d, uv_2d)), P1), op=op)
            if op.adapt_field != 's':       # Metric for free surface
                M2 = steady_metric(project(elev_2d, P1), op=op)
            if op.adapt_field == 'b':       # Intersect metrics for fluid speed and free surface
                M = metric_intersection(M, M2)
            elif op.adapt_field == 'f':
                M = M2

    elif op.approach in ('DWP', 'DWR'):
        assert(error_indicator is not None)

        # Compute metric field
        M = isotropic_metric(error_indicator, invert=False, op=op)
        if op.gradate:
            bdy = 'on_boundary'  # TODO: Use boundary tags to gradate to individual boundaries
            H0 = project(CellSize(mesh2d), P1)
            M_ = isotropic_metric(H0, bdy=bdy, op=op)  # Initial boundary metric
            M = metric_intersection(M, M_, bdy=bdy)
            M = gradate_metric(M, op=op)
    mesh2d = AnisotropicAdaptation(mesh2d, M).adapted_mesh
    print("Number of elements after mesh adaptation: {:d}".format(mesh2d.num_cells()))

    return mesh2d

