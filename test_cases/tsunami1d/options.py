from firedrake import *

import numpy as np

from adapt_utils.swe.options import ShallowWaterOptions


__all__ = ["Tsunami1dOptions"]


class Tsunami1dOptions(ShallowWaterOptions):
    def __init__(self, nx=2000, dt=1.0, **kwargs):
        super(Tsunami1dOptions, self).__init__(**kwargs)
        self.dt = dt
        self.end_time = 4200.0
        nt = int(np.round(self.end_time/self.dt))

        # Mesh: 2d space-time approach
        self.default_mesh = RectangleMesh(nx, nt, 400.0e+3, self.end_time)
        self.t_init_tag = 3
        self.t_final_tag = 4

        # Discretisation
        self.stabilisation = 'no'
        self.degree = 1
        # self.family = 'taylor-hood'

        # Initial/final condition
        self.source_loc = [(125.0e+3, 0.0, 25.0e+3)]
        self.region_of_interest = [(17.5e+3, self.end_time, 7.5e+3)]

        # Physical parameters
        self.g.assign(9.81)
        self.depth = 4000.0

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
        x, t = SpatialCoordinate(fs.mesh())
        self.bathymetry = interpolate(conditional(le(x, 50.0e+3), self.depth/20, self.depth), fs)
        return self.bathymetry

    def set_initial_condition(self, fs):
        self.initial_value = Function(fs)
        u, eta = self.initial_value.split()
        u.assign(0.0)
        x, t = SpatialCoordinate(fs.mesh())
        x0, t0, r = self.source_loc[0]
        tol = 1.0e-8
        bump = 0.4*sin(pi*(x-x0+r)/(2*r))
        # eta.interpolate(conditional(le(abs(x-x0), r), conditional(le(abs(t-t0), tol), bump, 0.0), 0.0))
        eta.interpolate(conditional(le(abs(x-x0), r), bump, 0.0))
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
        freeslip = {'un': Constant(0.0)}
        self.boundary_conditions = {1: freeslip, 2: freeslip, 3: {'uv': u, 'elev': eta}, 4: {}}
        return self.boundary_conditions

    def set_qoi_kernel(self, fs):
        x, t = SpatialCoordinate(fs)
        self.kernel = Function(fs)
        ku, ke = self.kernel.split()
        ku.assign(0.0)
        x0, t0, r = self.region_of_interest[0]
        tol = 1.0e-8
        ke.interpolate(conditional(le(abs(x-x0), r), conditional(le(abs(t-t0), tol, 1.0, 0.0)), 0.0))
        # ke.interpolate(conditional(le(abs(x-x0), r), conditional(le(abs(t-t0), tol, 0.4, 0.0)), 0.0))
        return self.kernel
