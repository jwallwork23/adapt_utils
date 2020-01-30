from thetis import *
from thetis.configuration import *

from adapt_utils.tracer.options import TracerOptions


__all__ = ["BubbleOptions"]


# TODO: 3d version
class BubbleOptions(TracerOptions):
    # TODO: doc
    def __init__(self, n=0, **kwargs):
        super(BubbleOptions, self).__init__(**kwargs)
        self.default_mesh = UnitSquareMesh(40*2**n, 40*2**n)

        # Source / receiver
        self.source_loc = [(0.5, 0.35, 0.35)]
        # self.region_of_interest = []
        self.base_diffusivity = 0.

        # Time integration
        self.dt = 0.015
        self.end_time = 1.5 + self.dt
        self.dt_per_export = 10
        self.dt_per_remesh = 10
        self.period = 6.0

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
        if not hasattr(self, 'fluid_velocity'):
            self.fluid_velocity = Function(fs)
        self.fluid_velocity.interpolate(
                as_vector((2*sin(pi*x)*sin(pi*x)*sin(2*pi*y)*cos(2*pi*t/T),
                           -2*sin(2*pi*x)*sin(pi*y)*sin(pi*y)*cos(2*pi*t/T))))
        return self.fluid_velocity

    def set_source(self, fs):
        self.source = Constant(0.)
        return self.source

    def set_initial_condition(self, fs):
        self.initial_value = interpolate(self.ball(source=True), fs)
        return self.initial_value

    def get_update_forcings(self, solver_obj):

        def update_forcings(t):
            self.set_velocity(solver_obj.function_spaces.Q_2d, t=t)

        return update_forcings
