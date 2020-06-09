from firedrake import *

from adapt_utils.tracer.options import TracerOptions


__all__ = ["ALEAdvectionOptions"]


class ALEAdvectionOptions(TracerOptions):
    def __init__(self, n=40, approach='ale', *args, **kwargs):
        super(ALEAdvectionOptions, self).__init__(*args, approach=approach, **kwargs)
        self.solve_swe = False
        self.solve_tracer = True
        if self.tracer_family == 'cg':
            self.stabilisation = 'SUPG'
        elif self.tracer_family == 'dg':
            self.stabilisation = None
        self.num_adapt = 1
        self.nonlinear_method = 'relaxation'

        lx, ly = 10, 10
        self.default_mesh = PeriodicRectangleMesh(n, n, lx, ly, direction='x')
        self.periodic = True
        self.dt = 0.2
        self.dt_per_export = 1
        self.dt_per_remesh = 1
        self.end_time = 10.0

        # self.base_diffusivity = 1.0e-8
        self.base_diffusivity = 0.0
        self.base_velocity = [1.0, 0.0]

    def set_source(self, fs):  # TODO
        self.source = Function(fs, name="Tracer source")
        return self.source

    def set_boundary_conditions(self, prob, i):
        boundary_conditions['tracer'] = {
            1: {'diff_flux': Constant(0.0)},
            2: {'diff_flux': Constant(0.0)},
        }
        return boundary_conditions

    def set_initial_condition_tracer(self, prob):
        x, y = SpatialCoordinate(prob.meshes[0])
        x0, y0 = 5.0, 5.0
        prob.fwd_solutions_tracer[0].interpolate(exp(-((x-x0)**2 + (y-y0)**2)))
