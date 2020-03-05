from firedrake import *

from adapt_utils.tracer.options import TracerOptions


__all__ = ["ALEAdvectionOptions"]


class ALEAdvectionOptions(TracerOptions):
    def __init__(self, n=40, approach='ale', *args, **kwargs):
        super(ALEAdvectionOptions, self).__init__(*args, approach=approach, **kwargs)
        self.family = 'CG'
        self.stabilisation = 'SUPG'

        lx, ly = 10, 10
        self.default_mesh = PeriodicRectangleMesh(n, n, lx, ly, direction='x')
        self.periodic = True
        self.dt = 0.2
        self.dt_per_export = 1
        self.end_time = 10.0

        self.base_diffusivity = 1.0e-8
        self.base_velocity = [1.0, 0.0]

        self.params = {
            "ksp_type": "gmres",
            "pc_type": "sor",
            "ksp_monitor": None,
            "ksp_converged_reason": None,
        }

    def set_velocity(self, fs):
        self.fluid_velocity = Constant(as_vector(self.base_velocity))
        return self.fluid_velocity

    def set_diffusivity(self, fs):
        self.diffusivity = Constant(self.base_diffusivity)
        return self.diffusivity

    def set_source(self, fs):
        self.source = Function(fs, name="Tracer source")
        return self.source

    def set_boundary_conditions(self, fs):
        self.boundary_conditions = {1: {'diff_flux': Constant(0.0)}, 2: {'diff_flux': Constant(0.0)}}
        return self.boundary_conditions

    def set_initial_condition(self, fs):
        self.initial_value = Function(fs)
        x, y = SpatialCoordinate(fs.mesh())
        x0, y0 = 5.0, 5.0
        self.initial_value.interpolate(exp(-((x-x0)**2 + (y-y0)**2)))
        return self.initial_value
