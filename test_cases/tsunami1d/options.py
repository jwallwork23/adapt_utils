from firedrake import *

import numpy as np
import warnings

from adapt_utils.swe.options import ShallowWaterOptions


__all__ = ["Tsunami1dOptions"]


class Tsunami1dOptions(ShallowWaterOptions):
    def __init__(self, nx=2000, dt=1.0, horizontal_length_scale=1000.0, time_scale=10.0, quads=False, **kwargs):
        super(Tsunami1dOptions, self).__init__(**kwargs)
        self.L = horizontal_length_scale
        self.T = time_scale
        self.dt = dt/self.T
        self.start_time = 0.0/self.T
        self.end_time = 4200.0/self.T
        nt = int(np.round(self.end_time/self.dt))

        # Mesh: 2d space-time approach
        lx = 400.0e+3/self.L
        if quads:
            try:
                assert self.approach in ('fixed_mesh', 'uniform')
            except AssertionError:
                warnings.warn("Quadrilaterals not allowed for adaptation with pragmatic. Reverting to triangles.")
                quads = False
        self.default_mesh = RectangleMesh(nx, nt, lx, self.end_time, quadrilateral=quads)
        self.t_init_tag = 3
        self.t_final_tag = 4

        # Discretisation
        self.stabilisation = 'no'
        self.degree = 1
        # self.family = 'taylor-hood'

        # Initial/final condition
        self.source_loc = [(125.0e+3/self.L, self.start_time, 25.0e+3/self.L)]
        self.region_of_interest = [(17.5e+3/self.L, self.end_time, 7.5e+3/self.L)]

        # Physical parameters
        self.g.assign(9.81)
        self.depth = 4000.0
        self.shelf_break_loc = 50.0e+3/self.L

        # Solver parameters
        direct_params = {  # Just whack it with a full LU preconditioner
            'mat_type': 'aij',
            'ksp_type': 'preonly',
            'pc_type': 'lu',
            'pc_factor_mat_solver_type': 'mumps',
        }
        mres_params = {  # Whack it with an assembled LU preconditioner  NOTE: Doesn't work in serial
            'mat_type': 'matfree',
            'snes_type': 'ksponly',
            'ksp_type': 'gmres',
            'pc_type': 'python',
            'pc_python_type': 'firedrake.AssembledPC',
            'assembled_pc_type': 'lu',
            'snes_lag_preconditioner': -1,
            'snes_lag_preconditioner_persists': None,
        }
        firedrake_fluids_params = {  # Use a "physics-based" method
            'ksp_type': 'gmres',
            'pc_type': 'fieldsplit',
            'pc_fieldsplit_type': 'schur',
            'pc_fieldsplit_fact_type': 'FULL',
            'fieldsplit_0_ksp_type': 'preonly',
            'fieldsplit_0_pc_type': 'ilu',
            'fieldsplit_1_ksp_type': 'preonly',
            'fieldsplit_1_pc_type': 'ilu',
        }
        self.params = direct_params
        self.params['ksp_monitor'] = None
        self.params['ksp_converged_reason'] = None
        self.params['ksp_monitor_true_residual'] = None
        self.adjoint_params = self.params

    def set_bathymetry(self, fs):
        x, t = SpatialCoordinate(fs.mesh())
        self.bathymetry = interpolate(conditional(le(x, self.shelf_break_loc), self.depth/20, self.depth), fs)
        return self.bathymetry

    def set_initial_condition(self, fs):
        self.initial_value = Function(fs)
        u, eta = self.initial_value.split()
        u.assign(0.0)
        x, t = SpatialCoordinate(fs.mesh())
        x0, t0, r = self.source_loc[0]
        tol = self.dt/2
        amplitude = 0.4
        bump = amplitude*sin(pi*(x-x0+r)/(2*r))
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
        x, t = SpatialCoordinate(fs.mesh())
        self.kernel = Function(fs)
        ku, ke = self.kernel.split()
        ku.assign(0.0)
        x0, t0, r = self.region_of_interest[0]
        tol = self.dt/2
        # amplitude = 0.4
        amplitude = 1.0
        bump = amplitude*exp(1 - 1/(1 - ((x-x0)/r)**2))
        # ke.interpolate(conditional(le(abs(x-x0), r), conditional(le(abs(t-t0), tol), amplitude, 0.0), 0.0))
        # ke.interpolate(conditional(le(abs(x-x0), r), amplitude, 0.0))
        ke.interpolate(conditional(lt(abs(x-x0), r), bump, 0.0))
        return self.kernel

    def evaluate_qoi(self, sol):
        eta = sol.split()[1]
        x, t = SpatialCoordinate(sol.function_space().mesh())
        x0, t0, r = self.region_of_interest[0]
        # amplitude = 0.4
        amplitude = 1.0
        bump = amplitude*exp(1 - 1/(1 - ((x-x0)/r)**2))
        kernel = conditional(lt(abs(x-x0), r), bump, 0.0)
        return assemble(kernel*eta*ds(self.t_final_tag, degree=12))
