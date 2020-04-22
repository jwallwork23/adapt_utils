from thetis import *
from thetis.configuration import *

from adapt_utils.tracer.options import TracerOptions


__all__ = ["BubbleOptions"]


# TODO: 3d version
class BubbleOptions(TracerOptions):
    # TODO: doc
    def __init__(self, n=1, **kwargs):
        super(BubbleOptions, self).__init__(**kwargs)
        self.default_mesh = UnitSquareMesh(40*2**n, 40*2**n)
        if self.family in ('CG', 'cg', 'Lagrange'):
            self.stabilisation = 'SUPG'
        elif self.family in ('DG', 'dg', 'Discontinuous Lagrange'):
            self.stabilisation = 'no'
        else:
            raise NotImplementedError
        self.num_adapt = 1
        self.nonlinear_method = 'relaxation'

        # Source / receiver
        self.source_loc = [(0.5, 0.85, 0.1)]
        # self.region_of_interest = []
        self.base_diffusivity = 0.

        # Time integration
        self.period = 6.0
        self.dt = 0.015
        # self.end_time = self.period/4 + self.dt
        self.end_time = self.period/2 + self.dt
        # self.dt_per_export = 10
        self.dt_per_export = 1
        self.dt_per_remesh = 10

    def set_boundary_conditions(self, fs):
        zero = Constant(0.0, domain=fs.mesh())
        for i in range(1, 5):
            self.boundary_conditions[i] = {i: {'value': zero}}
            self.adjoint_boundary_conditions[i] = {i: {'diff_flux': zero}}
        return self.boundary_conditions

    def set_diffusivity(self, fs):
        self.diffusivity = Constant(self.base_diffusivity)
        return self.diffusivity

    def set_velocity(self, fs, t=0.0):
        x, y = SpatialCoordinate(fs.mesh())
        T = self.period
        if not hasattr(self, 'fluid_velocity') or self.fluid_velocity is None or \
           isinstance(self.fluid_velocity, Constant) or fs != self.fluid_velocity.function_space():
            self.fluid_velocity = Function(fs)
        self.fluid_velocity.interpolate(
            as_vector([2*sin(pi*x)*sin(pi*x)*sin(2*pi*y)*cos(2*pi*t/T),
                       -2*sin(2*pi*x)*sin(pi*y)*sin(pi*y)*cos(2*pi*t/T)]))
        return self.fluid_velocity

    def set_source(self, fs):
        self.source = Constant(0.)
        return self.source

    def set_initial_condition(self, fs):
        self.initial_value = self.ball(fs, source=True)
        return self.initial_value

    def get_update_forcings(self, solver_obj=None):

        def update_forcings(t):
            # self.set_velocity(solver_obj.function_spaces.Q_2d, t=t)
            self.set_velocity(self.fluid_velocity.function_space(), t=t)

        return update_forcings

    def set_qoi_kernel(self, fs):
        pass  # TODO
