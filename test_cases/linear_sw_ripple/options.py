from firedrake import *

from adapt_utils.swe.options import ShallowWaterOptions


__all__ = ["RippleOptions"]


class RippleOptions(ShallowWaterOptions):
    def __init__(self, nx=16, nt=25, shelf=False, **kwargs):
        super(RippleOptions, self).__init__(**kwargs)
        self.shelf = shelf
        lx = 4.0
        self.dt = 0.05
        self.end_time = nt*self.dt

        # Mesh: 3d approach
        self.default_mesh = BoxMesh(nx, nx, nt, lx, lx, self.end_time)
        self.t_init_tag = 5
        self.t_final_tag = 6

        # Discretisation
        self.stabilisation = 'no'
        self.degree = 1
        # self.family = 'taylor-hood'

        # Initial condition
        self.source_loc = [(lx/2, lx/2, 0.0, 0.1)]

        # Physical parameters
        self.g.assign(9.81)
        self.depth = 0.1

        # Solver parameters
        self.params = {
            # 'mat_type': 'matfree',
            'mat_type': 'aij',
            'ksp_type': 'preonly',
            # 'pc_type': 'python',
            # 'pc_python_type': 'firedrake.AssembledPC',
            # 'assembled_pc_type': 'lu'
            'pc_type': 'lu',
            'pc_factor_mat_solver_type': 'mumps',
            'ksp_monitor': None,
        }

    def set_bathymetry(self, fs):
        if self.shelf:
            x, y, z = SpatialCoordinate(fs.mesh())
            self.bathymetry = interpolate(conditional(le(x, 0.5), self.depth/10, self.depth), fs)
        else:
            self.bathymetry = Constant(self.depth)
        return self.bathymetry

    def set_initial_condition(self, fs):
        self.initial_value = Function(fs)
        u, eta = self.initial_value.split()
        u.interpolate(as_vector([0.0, 0.0]))
        # eta.interpolate(self.gaussian(fs.sub(1), source=True, spacetime=True))
        x, y, z = SpatialCoordinate(fs.mesh())
        x0 = 2.0
        y0 = 2.0
        eta.interpolate(0.001*exp(-((x-x0)*(x-x0) + (y-y0)*(y-y0))/0.04))
        return self.initial_value

    def set_coriolis(self, fs):
        self.coriolis = Constant(0.0)
        return self.coriolis

    def set_viscosity(self, fs):
        self.viscosity = Constant(0.0)
        return self.viscosity

    def set_boundary_conditions(self, fs):
        if not hasattr(self, 'initial_value'):
            self.set_initial_condition(fs)
        u, eta = self.initial_value.split()
        self.boundary_conditions = {1: {'un': Constant(0.0)}, 2: {'un': Constant(0.0)},
                                    3: {'un': Constant(0.0)}, 4: {'un': Constant(0.0)},
                                    5: {'uv': u, 'elev': eta}, 6: {}}
