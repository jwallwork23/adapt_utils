from firedrake import *

from adapt_utils.options import CoupledOptions


__all__ = ["CosinePrescribedVelocityOptions"]


class CosinePrescribedVelocityOptions(CoupledOptions):
    def __init__(self, n=40, *args, **kwargs):
        super(CosinePrescribedVelocityOptions, self).__init__(*args, **kwargs)
        self.solve_swe = False
        self.solve_tracer = True
        if self.tracer_family == 'cg':
            self.stabilisation_tracer = 'supg'
        elif self.tracer_family == 'dg':
            self.stabilisation_tracer = 'lax_friedrichs'

        lx, ly = 10, 10
        self.default_mesh = PeriodicRectangleMesh(n, n, lx, ly, direction='x')
        self.periodic = True
        self.dt = 0.2
        self.dt_per_export = 1
        self.end_time = 10.0

        # self.base_diffusivity = 1.0e-8
        self.base_diffusivity = Constant(0.0)
        self.base_velocity = [1.0, 0.0]
        self.characteristic_speed = Constant(1.0)

    def set_boundary_conditions(self, prob, i):
        boundary_conditions = {
            'tracer': {
                1: {'diff_flux': Constant(0.0)},
                2: {'diff_flux': Constant(0.0)},
            }
        }
        return boundary_conditions

    def set_initial_condition_tracer(self, prob):
        x, y = SpatialCoordinate(prob.meshes[0])
        x0, y0 = 5.0, 5.0
        prob.fwd_solutions_tracer[0].interpolate(exp(-((x-x0)**2 + (y-y0)**2)))

    def get_update_forcings(self, prob, i, adjoint=False):
        x, y = SpatialCoordinate(prob.meshes[i])
        u, eta = prob.fwd_solutions[i].split()

        def update_forcings(t):
            u.interpolate(as_vector([1.0, cos(2*pi*t/self.end_time)]))

        return update_forcings
