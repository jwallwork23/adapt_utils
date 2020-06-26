from thetis import *

import numpy as np
import os


class DesalinationParameters():
    def __init__(self, n, solve_swe=False):

        # Domain geometry
        # self.domain_length, self.domain_width = 3000.0, 1000.0
        self.domain_length, self.domain_width = 100.0, 20.0
        # nx, ny = 60*2**n, 20*2**n
        nx, ny = 100*2**n, 20*2**n
        self.default_mesh = RectangleMesh(nx, ny, self.domain_length, self.domain_width)
        P0 = FunctionSpace(self.default_mesh, "DG", 0)
        P1 = FunctionSpace(self.default_mesh, "CG", 1)

        # Outlet pipe / source term
        self.outlet_x, self.outlet_y = self.domain_length/2, self.domain_width/2
        # self.outlet_r = 30.0
        self.outlet_r = 0.5
        self.outlet_rate = 0.1

        # Physics
        self.tracer = {
            'horizontal_diffusivity': Constant(0.1),
            'use_lax_friedrichs_tracer': True,
        }
        self.shallow_water = {
            'horizontal_viscosity': Constant(3.0),
            'quadratic_drag_coefficient': Constant(0.0025),
            'element_family': 'dg-cg',
        }
        # self.depth = 50.0
        self.depth = 5.0
        self.bathymetry = Function(P1).assign(self.depth)
        self.gravity = Constant(9.81)

        # Tidal forcing
        # self.T_tide = 12.42*60*60
        self.T_tide = 1.24*60*60
        self.T_amp = Constant(0.5)

        # Boundary conditions
        self.boundary_conditions = {}
        self.boundary_conditions['tracer'] = {
            1: {'value': Constant(0.0)},  # Upwind
            2: {'value': Constant(0.0)},  # Upwind
            # 3: {'open': None},            # Open
            # 4: {'open': None},            # Open
        }

        # Outputs
        abspath = os.path.dirname(__file__)
        self.di = create_directory(os.path.join(abspath, 'outputs'))
        for setup in (1, 2):
            qoi = Function(P0, name="Region of interest")
            qoi.interpolate(self.quantity_of_interest_kernel(self.default_mesh, setup))
            File(os.path.join(self.di, 'region_of_interest_{:d}.pvd'.format(setup))).write(qoi)

    def ball(self, mesh, triple, scaling=1.0, eps=1.0e-10):
        x, y = SpatialCoordinate(mesh)
        expr = lt((x-triple[0])**2 + (y-triple[1])**2, triple[2]**2 + eps)
        return conditional(expr, scaling, 0.0)

    def source(self, fs):
        triple = (self.outlet_x, self.outlet_y, self.outlet_r)
        area = assemble(self.ball(fs.mesh(), triple)*dx)
        area_exact = pi*triple[2]**2
        scaling = 1.0 if np.allclose(area, 0.0) else area_exact/area
        # scaling *= 0.5*self.outlet_rate
        scaling *= self.outlet_rate
        return self.ball(fs.mesh(), triple, scaling=scaling)

    def quantity_of_interest_kernel(self, mesh, setup):

        # Inlet pipe parametrisation
        inlet_x = 0.2*self.domain_length
        inlet_y = 0.5*self.domain_width if setup == 1 else 0.75*self.domain_width
        # inlet_r = 30.0
        inlet_r = 0.5

        # Corresponding QoI
        triple = (inlet_x, inlet_y, inlet_r)
        area = assemble(self.ball(mesh, triple)*dx)
        area_exact = pi*triple[2]**2
        scaling = 1.0 if np.allclose(area, 0.0) else area_exact/area
        return self.ball(mesh, triple, scaling=scaling)

    def quantity_of_interest(self, sol, setup):
        kernel = self.quantity_of_interest_kernel(sol.function_space().mesh(), setup)
        return assemble(inner(kernel, sol)*dx(degree=12))

    def quantities_of_interest(self, sol):
        return [self.quantity_of_interest(sol, 1), self.quantity_of_interest(sol, 2)]


class InstantaneousQoICallback(callback.DiagnosticCallback):
    variable_names = ["spatial integral at current timestep"]

    def __init__(self, functional_callback, solver_obj, **kwargs):
        """
        :arg functional_callback: Python function that returns a list of values for various
            functionals.
        :arg solver_obj: Thetis solver object
        :arg **kwargs: any additional keyword arguments, see DiagnosticCallback
        """
        kwargs.setdefault('export_to_hdf5', False)
        kwargs.setdefault('append_to_log', True)
        kwargs.setdefault('separator', '   ')  # Choose '\n' for separate lines
        self.sep = kwargs.pop('separator')
        super(InstantaneousQoICallback, self).__init__(solver_obj, **kwargs)
        self.callback = functional_callback

    def __call__(self):
        return [self.callback(), ]  # Evaluate functionals

    def message_str(self, *args):
        f = args[0]
        msg = '{:s} {:d} value {:11.4e}'
        return self.sep.join([msg.format(self.name, i, f[i]) for i in range(len(f))])


class QoICallback(InstantaneousQoICallback):
    name = "qoi"

    def __init__(self, solver_obj, params, **kwargs):
        qoi = lambda: params.quantities_of_interest(solver_obj.fields.tracer_2d)
        super(QoICallback, self).__init__(qoi, solver_obj, **kwargs)


def solve_tracer(n, solve_swe=False, **model_options):

    # Set up parameter class
    params = DesalinationParameters(n, solve_swe=solve_swe)
    mesh2d = params.default_mesh
    x, y = SpatialCoordinate(mesh2d)

    # Pass parameters to solver object
    solver_obj = solver2d.FlowSolver2d(mesh2d, params.bathymetry)
    options = solver_obj.options
    options.output_directory = params.di
    options.timestep = 4.0*0.5**n
    options.simulation_end_time = 3.0*params.T_tide
    options.simulation_export_time = 30.0
    options.fields_to_export = ['uv_2d', 'elev_2d', 'tracer_2d']
    options.solve_tracer = True
    options.update(params.tracer)
    if solve_swe:
        options.update(params.shallow_water)
    else:
        options.tracer_only = True
    options.update(model_options)

    # Set initial conditions
    solver_obj.create_function_spaces()
    options.tracer_source_2d = params.source(solver_obj.function_spaces.P1_2d)
    uv, eta = Function(solver_obj.function_spaces.V_2d).split()
    uv.interpolate(as_vector([1.0e-08, 0.0]))
    eta.interpolate(params.T_amp*(0.5*params.domain_length - x)/params.domain_length)
    solver_obj.assign_initial_conditions(uv=uv, elev=eta)

    # Account for tidal forcing
    tc = Constant(0.0)
    omega = Constant(2*pi/params.T_tide)
    L = Constant(params.domain_length)
    g = params.gravity
    T_amp = params.T_amp
    if solve_swe:
        elev_in = Function(solver_obj.function_spaces.H_2d)
        elev_out = Function(solver_obj.function_spaces.H_2d)

        # Sinusoidal variance in elevation at opposing boundaries
        def update_forcings(t):
            tc.assign(t)
            elev_in.assign(T_amp*cos(omega*tc))
            elev_in.assign(T_amp*cos(omega*tc+pi))

        update_forcings(0.0)
        solver_obj.bnd_functions['shallow_water'] = {1: {'elev': elev_in}, 2: {'elev': elev_out}}
    else:
        # Simple linear variation in free surface:
        #  eta = T_amp - 2*T_amp/L * cos(omega*t) * x
        #  => deta/dx = -2*T_amp/L * cos(omega*t)

        # Linear SWE momentum du/dt = -g * deta/dx
        #  => du/dx = 2*T_amp/(g*L) * cos(omega*t)
        #  => u = 2*T_amp/(g*omega*L) * sin(omega*t)
        def update_forcings(t):
            tc.assign(t)
            solver_obj.fields.elev_2d.interpolate(T_amp - 2.0*T_amp*cos(omega*tc)/L * x)
            solver_obj.fields.uv_2d.interpolate(as_vector([2.0*T_amp*sin(omega*tc)/(g*omega*L), 0.0]))

        update_forcings(0.0)
    solver_obj.bnd_functions['tracer'] = params.boundary_conditions['tracer']

    # Callback for quantities of interest
    cb = QoICallback(solver_obj, params, append_to_log=True)
    solver_obj.add_callback(cb, 'timestep')

    # Solve
    solver_obj.iterate(update_forcings=update_forcings)


if __name__ == "__main__":
    solve_tracer(1, False, timestepper_type='CrankNicolson')
