from firedrake import *

from adapt_utils.tracer.options import TracerOptions


__all__ = ["ALEAdvectionOptions"]


class ALEAdvectionOptions(TracerOptions):
    def __init__(self, n=40, prescribed_velocity='fluid', approach='ale', *args, **kwargs):
        self.prescribed_velocity = prescribed_velocity
        super(ALEAdvectionOptions, self).__init__(*args, approach=approach, **kwargs)
        self.family = 'CG'
        self.stabilisation = 'SUPG'

        lx, ly = 30, 10
        self.default_mesh = PeriodicRectangleMesh(3*n, n, lx, ly, direction='x')
        self.dt = 0.1
        self.end_time = 5.0

        self.base_diffusivity = 1.0e-8
        self.base_velocity = [1.0, 0.0]

        self.params = {
            "ksp_type": "gmres",
            "pc_type": "sor",
            "ksp_monitor": None,
            "ksp_converged_reason": None,
        }

    def set_velocity(self, fs):
        self.velocity = Constant(as_vector(self.base_velocity))
        return self.velocity

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

    def get_mesh_velocity(self):
        if self.prescribed_velocity == "constant":
            self.mesh_velocity = lambda mesh: Constant(as_vector([0.0, 0.0]))  # Fixed mesh
        elif self.prescribed_velocity == "fluid":
            self.mesh_velocity = lambda mesh: self.velocity  # Prescribed velocity: that of the fluid
        else:
            raise NotImplementedError
        return self.mesh_velocity
