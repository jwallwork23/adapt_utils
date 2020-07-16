from firedrake import *

from adapt_utils.swe.options import ShallowWaterOptions


__all__ = ["RippleOptions"]


class RippleOptions(ShallowWaterOptions):
    def __init__(self, nx=16, nt=25, shelf=False, **kwargs):
        super(RippleOptions, self).__init__(**kwargs)
        self.solve_swe = True
        self.solve_tracer = False
        self.shelf = shelf
        lx = 4.0
        self.dt = 0.05
        self.end_time = nt*self.dt

        # Mesh: 3d space-time approach
        self.default_mesh = BoxMesh(nx, nx, nt, lx, lx, self.end_time)
        self.t_init_tag = 5
        self.t_final_tag = 6

        # Discretisation
        self.stabilisation = None
        self.degree = 1
        # self.family = 'cg-cg'

        # Initial condition
        self.source_loc = [(lx/2, lx/2, 0.0, 0.1)]

        # Physical parameters
        self.g.assign(9.81)
        self.base_bathymetry = 0.1
        self.base_viscosity = 0.0

        # Solver parameters
        self.solver_parameters = {
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
        self.adjoint_solver_parameters.update(self.solver_parameters)

    def set_bathymetry(self, fs):
        b = base_bathymetry
        if self.shelf:
            x, y, t = SpatialCoordinate(fs.mesh())
            return interpolate(conditional(le(x, 0.5), b/10, b), fs)
        else:
            return Constant(b)

    def set_initial_condition(self, prob):
        args = prob.fwd_solutions[0].split()
        if len(args) == 3:
            udiv = args[2]
            udiv.assign(0.0)
        u, eta = args[:2]
        u.assign(0.0)
        x, y, t = SpatialCoordinate(fs.mesh())
        x0, y0, t0, r = self.source_loc[0]  # TODO: we haven't used r
        eta.interpolate(0.001*exp(-((x-x0)*(x-x0) + (y-y0)*(y-y0))/0.04))

    def set_boundary_conditions(self, prob, i):
        args = self.set_initial_condition(prob.V[i]).split()
        u, eta = args[:2]
        # udiv = args[2]
        boundary_conditions = {
            1: {'un': Constant(0.0)},
            2: {'un': Constant(0.0)},
            3: {'un': Constant(0.0)},
            4: {'un': Constant(0.0)},
            5: {'uv': u, 'elev': eta},
            6: {},
        }
        return boundary_conditions
