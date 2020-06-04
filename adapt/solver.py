from thetis import *
from firedrake.petsc import PETSc

import os
import numpy as np

from adapt_utils.swe.utils import *
from adapt_utils.adapt.metric import *
from adapt_utils.adapt.adaptation import pragmatic_adapt
from adapt_utils.swe.utils import ShallowWaterHessianRecoverer


__all__ = ["AdaptiveProblem"]


class AdaptiveProblem():
    """
    Solver object for adaptive mesh simulations with a number of meshes which is known a priori.
    In the steady state case, the number of meshes is clearly known to be one. In the unsteady
    case, it is likely that we seek to use more than one mesh.

    The philosophy here is to separate the PDE solution from the mesh adaptation, in the sense that
    the forward (and possibly adjoint) equations are solved over the whole time period before any
    mesh adaptation is performed. This means that the solver object is based upon a sequence of
    meshes, as opposed to a single mesh which is updated on-the-fly. Whilst this approach has
    increased memory requirements compared with the on-the-fly strategy, it is beneficial for
    goal-oriented mesh adaptation, where an outer loop is required.
    """

    # --- Setup

    def __init__(self, op, meshes=None, discrete_adjoint=True):
        op.print_debug(op.indent + "{:s} initialisation begin".format(self.__class__.__name__))

        # Read args and kwargs
        self.op = op
        self.stabilisation = op.stabilisation
        self.discrete_adjoint = discrete_adjoint
        self.approach = op.approach

        # Sub options
        self.timestepping_options = {
            'timestep': op.dt,
            'simulation_export_time': op.dt*op.dt_per_export,
            'timestepper_type': op.timestepper,
        }
        self.num_timesteps = int(np.floor(op.end_time/op.dt))
        self.num_meshes = op.num_meshes
        try:
            assert self.num_timesteps % op.num_meshes == 0
        except AssertionError:
            raise ValueError("Number of meshes should divide total number of timesteps.")
        self.dt_per_mesh = self.num_timesteps//op.num_meshes
        try:
            assert self.dt_per_mesh % op.dt_per_export == 0
        except AssertionError:
            raise ValueError("Timesteps per export should divide timesteps per mesh iteration.")
        self.io_options = {
            'output_directory': op.di,
            'fields_to_export': ['uv_2d', 'elev_2d'] if op.plot_pvd else [],
            'fields_to_export_hdf5': ['uv_2d', 'elev_2d'] if op.save_hdf5 else [],
            'no_exports': True,  # TODO: TEMPORARY
        }
        self.shallow_water_options = {
            'use_nonlinear_equations': True,
            'element_family': op.family,
            'polynomial_degree': op.degree,
            'use_grad_div_viscosity_term': op.grad_div_viscosity,
            'use_grad_depth_viscosity_term': op.grad_depth_viscosity,
            'use_automatic_sipg_parameter': op.use_automatic_sipg_parameter,
            'use_wetting_and_drying': op.wetting_and_drying,
            'wetting_and_drying_alpha': op.wetting_and_drying_alpha,
            # 'check_volume_conservation_2d': True,
        }
        if hasattr(op, 'sipg_parameter') and op.sipg_parameter is not None:
            self.shallow_water_options['sipg_parameter'] = op.sipg_parameter
        self.tracer_options = {  # TODO
            'solve_tracer': op.solve_tracer,
            'tracer_only': not op.solve_swe,
        }
        physical_constants['g_grav'].assign(op.g)

        # Setup problem
        op.print_debug(op.indent + "SETUP: Building meshes...")
        self.set_meshes(meshes)
        op.print_debug(op.indent + "SETUP: Creating function spaces...")
        self.set_finite_element()
        self.create_function_spaces()
        op.print_debug(op.indent + "SETUP: Creating solutions...")
        self.create_solutions()
        op.print_debug(op.indent + "SETUP: Creating fields...")
        self.set_fields()
        op.print_debug(op.indent + "SETUP: Setting stabilisation parameters...")
        self.set_stabilisation()
        op.print_debug(op.indent + "SETUP: Setting boundary conditions...")
        self.set_boundary_conditions()
        self.callbacks = [{} for mesh in self.meshes]

        # Lists of objects to be populated
        self.fwd_solvers = [None for mesh in self.meshes]
        self.adj_solvers = [None for mesh in self.meshes]
        self.kernels = [None for mesh in self.meshes]

        # Outputs
        self.di = create_directory(self.op.di)
        self.solution_file = File(os.path.join(self.di, 'solution.pvd'))
        # self.solution_fpath_hdf5 = os.path.join(self.di, 'solution.hdf5')
        self.adjoint_solution_file = File(os.path.join(self.di, 'adjoint_solution.pvd'))
        # self.adjoint_solution_fpath_hdf5 = os.path.join(self.di, 'adjoint_solution.hdf5')
        self.bathymetry_file = File(os.path.join(self.di, 'bathymetry.pvd'))
        self.indicator_file = File(os.path.join(self.di, 'indicator.pvd'))
        self.kernel_file = File(os.path.join(self.di, 'kernel.pvd'))

        # Storage for diagnostics over mesh adaptation loop
        self.num_cells = [[mesh.num_cells() for mesh in self.meshes], ]
        self.num_vertices = [[mesh.num_vertices() for mesh in self.meshes], ]
        self.dofs = [[np.array(V.dof_count).sum() for V in self.V], ]
        self.indicators = [{} for mesh in self.meshes]
        self.estimators = [{} for mesh in self.meshes]
        self.qois = []
        self.st_complexities = [np.nan]

    # TODO: AdaptiveMesh
    def set_meshes(self, meshes):
        """
        Build an class:`AdaptiveMesh` object associated with each mesh.
        """
        self.meshes = meshes or [self.op.default_mesh for i in range(self.num_meshes)]
        msg = self.op.indent + "SETUP: Mesh {:d} has {:d} elements"
        for i, mesh in enumerate(self.meshes):
            self.op.print_debug(msg.format(i, mesh.num_cells()))

    def set_finite_element(self):
        p = self.op.degree
        assert p >= 0
        family = self.op.family
        if family == 'taylor-hood':
            u_element = VectorElement("Lagrange", triangle, p+1)
            eta_element = FiniteElement("Lagrange", triangle, p, variant='equispaced')
        elif family == 'dg-dg':
            u_element = VectorElement("DG", triangle, p)
            eta_element = FiniteElement("DG", triangle, p, variant='equispaced')
        elif family == 'dg-cg':
            u_element = VectorElement("DG", triangle, p)
            eta_element = FiniteElement("Lagrange", triangle, p+1, variant='equispaced')
        else:
            raise NotImplementedError("Cannot build element {:s} of order {:d}".format(family, p))
        self.finite_element = u_element*eta_element

    def create_function_spaces(self):  # NOTE: Keep minimal
        """
        Build the finite element space, `V`, for the prognostic solution, along with various other
        useful spaces and test functions and trial functions based upon them.
        """
        self.P0 = [FunctionSpace(mesh, "DG", 0) for mesh in self.meshes]
        self.P1 = [FunctionSpace(mesh, "CG", 1) for mesh in self.meshes]
        self.P1_vec = [VectorFunctionSpace(mesh, "CG", 1) for mesh in self.meshes]
        self.P1_ten = [TensorFunctionSpace(mesh, "CG", 1) for mesh in self.meshes]
        self.P1DG = [FunctionSpace(mesh, "DG", 1) for mesh in self.meshes]
        # self.P1DG_vec = [VectorFunctionSpace(mesh, "DG", 1) for mesh in self.meshes]
        # self.P2 = [FunctionSpace(mesh, "CG", 2) for mesh in self.meshes]
        # self.P2_vec = [VectorFunctionSpace(mesh, "CG", 2) for mesh in self.meshes]

        # Shallow water space
        self.V = [FunctionSpace(mesh, self.finite_element) for mesh in self.meshes]

        # Tracer space
        self.Q = self.P1DG

    def create_solutions(self):
        """
        Set up `Function`s in the prognostic space to hold the forward and adjoint solutions.
        """
        if self.tracer_options['tracer_only']:
            self.fwd_solutions = None
            self.adj_solutions = None
            self.fwd_solutions_old = None
            self.adj_solutions_old = None
        else:
            self.fwd_solutions = []
            self.adj_solutions = []
            for V in self.V:
                fwd = Function(V, name='Forward solution')
                u, eta = fwd.split()
                u.rename("Fluid velocity")
                eta.rename("Elevation")
                self.fwd_solutions.append(fwd)

                adj = Function(V, name='Adjoint solution')
                z, zeta = adj.split()
                z.rename("Adjoint fluid velocity")
                zeta.rename("Adjoint elevation")
                self.adj_solutions.append(adj)
            self.fwd_solutions_old = [fwd.copy(deepcopy=True) for fwd in self.fwd_solutions]
            self.adj_solutions_old = [adj.copy(deepcopy=True) for adj in self.adj_solutions]

        if self.tracer_options['solve_tracer']:
            self.fwd_tracer_solutions = [Function(Q, name="Forward tracer solution") for Q in self.Q]
            self.adj_tracer_solutions = [Function(Q, name="Adjoint tracer solution") for Q in self.Q]
            self.fwd_tracer_solutions_old = [fwd.copy(deepcopy=True) for fwd in self.fwd_tracer_solutions]
            self.adj_tracer_solutions_old = [adj.copy(deepcopy=True) for adj in self.adj_tracer_solutions]
        else:
            self.fwd_tracer_solutions = None
            self.adj_tracer_solutions = None
            self.fwd_tracer_solutions_old = None
            self.adj_tracer_solutions_old = None

    def set_fields(self):
        """Set velocity field, viscosity, etc (on each mesh)."""
        self.fields = []
        for P1 in self.P1:
            self.fields.append({
                'horizontal_viscosity': self.op.set_viscosity(P1),
                'horizontal_diffusivity': self.op.set_diffusivity(P1),
                'coriolis_frequency': self.op.set_coriolis(P1),
                'quadratic_drag_coefficient': self.op.set_quadratic_drag_coefficient(P1),
                'manning_drag_coefficient': self.op.set_manning_drag_coefficient(P1),
            })
        self.inflow = [self.op.set_inflow(P1_vec) for P1_vec in self.P1_vec]
        self.bathymetry = [self.op.set_bathymetry(P1DG) for P1DG in self.P1DG]
        self.depth = [None for bathymetry in self.bathymetry]
        for i, bathymetry in enumerate(self.bathymetry):
            self.depth[i] = DepthExpression(
                bathymetry,
                use_nonlinear_equations=self.shallow_water_options['use_nonlinear_equations'],
                use_wetting_and_drying=self.shallow_water_options['use_wetting_and_drying'],
                wetting_and_drying_alpha=self.shallow_water_options['wetting_and_drying_alpha'],
            )

    # TODO: Allow different / mesh dependent stabilisation parameters
    # TODO: Tracer stabilisation
    def set_stabilisation(self):
        """ Set stabilisation mode and parameter(s) on each mesh."""
        self.stabilisation = self.stabilisation or 'no'
        try:
            assert self.stabilisation in ('no', 'lax_friedrichs')
        except AssertionError:
            msg = "Stabilisation method {:s} not recognised for {:s}"
            raise ValueError(msg.format(self.stabilisation, self.__class__.__name__))
        self.stabilisation_parameters = []
        for i in range(self.num_meshes):
            self.stabilisation_parameters.append(self.op.stabilisation_parameter)

    def set_boundary_conditions(self):
        self.boundary_conditions = [self.op.set_boundary_conditions(V) for V in self.V]

    def set_initial_condition(self):
        """Apply initial condition for forward solution on first mesh."""
        self.fwd_solutions[0].assign(self.op.set_initial_condition(self.V[0]))

    def set_final_condition(self):
        """Apply final time condition for adjoint solution on final mesh."""
        self.adj_solutions[-1].assign(self.op.set_final_condition(self.V[-1]))

    # --- Helper functions

    def project(self, f, i, j):
        """Project field `f` from mesh `i` onto mesh `j`."""
        if f[i] is None or isinstance(f[i], Constant):
            return
        elif f[i].function_space() == f[j].function_space():
            f[j].assign(f[i])
        else:
            for fik, fjk in zip(f[i].split(), f[j].split()):
                fjk.project(fik)

    def project_forward_solution(self, i, j):
        """Project forward solution from mesh `i` to mesh `j`."""
        self.project(self.fwd_solutions, i, j)

    def project_adjoint_solution(self, i, j):
        """Project adjoint solution from mesh `i` to mesh `j`."""
        self.project(self.adj_solutions, i, j)

    def transfer_solution(self, i, adjoint=False):
        if adjoint:
            self.transfer_adjoint_solution()
        else:
            self.transfer_forward_solution()

    def transfer_forward_solution(self, i):
        if i == 0:
            self.set_initial_condition()
        else:
            self.project_forward_solution(i-1, i)

    def transfer_adjoint_solution(self, i):
        if i == self.num_meshes - 1:
            self.set_final_condition()
        else:
            self.project_adjoint_solution(i+1, i)

    def store_plexes(self, di=None):
        """Save meshes to disk using DMPlex format."""
        di = di or os.path.join(self.di, self.approach)
        fname = os.path.join(di, 'plex_{:d}.h5')
        for i, mesh in enumerate(self.meshes):
            assert os.path.isdir(di)
            viewer = PETSc.Viewer().createHDF5(fname.format(i), 'w')
            viewer(mesh._plex)

    def load_plexes(self, fname):
        """Load meshes in DMPlex format."""
        for i in range(self.num_meshes):
            newplex = PETSc.DMPlex().create()
            newplex.createFromFile('_'.join([fname, '{:d}.h5'.format(i)]))
            self.meshes[i] = Mesh(newplex)

    # --- Solvers

    def setup_solver_forward(self, i, extra_setup=None):
        """Setup forward solver on mesh `i`."""
        op = self.op
        if extra_setup is not None:
            self.extra_setup = extra_setup

        # Create solver object
        self.fwd_solvers[i] = solver2d.FlowSolver2d(self.meshes[i], self.bathymetry[i])
        options = self.fwd_solvers[i].options
        options.update(self.io_options)

        # Timestepping
        options.simulation_end_time = (i+1)*op.end_time/self.num_meshes - 0.5*op.dt
        options.update(self.timestepping_options)
        if hasattr(options.timestepper_options, 'implicitness_theta'):
            options.timestepper_options.implicitness_theta = op.implicitness_theta
        if hasattr(options.timestepper_options, 'use_automatic_timestep'):
            options.timestepper_options.use_automatic_timestep = op.use_automatic_timestep

        # Solver parameters
        if op.params != {}:
            options.timestepper_options.solver_parameters = op.params
        if not self.shallow_water_options['use_nonlinear_equations']:
            options.timestepper_options.solver_parameters['snes_type'] = 'ksponly'
        op.print_debug(options.timestepper_options.solver_parameters)

        # Parameters
        options.estimate_error = op.estimate_error
        options.update(self.fields[i])
        options.update(self.shallow_water_options)
        options.update(self.tracer_options)
        options.use_lax_friedrichs_velocity = self.stabilisation == 'lax_friedrichs'
        options.lax_friedrichs_velocity_scaling_factor = self.stabilisation_parameters[i]
        if op.solve_tracer:
            raise NotImplementedError  # TODO

        # Boundary conditions
        self.fwd_solvers[i].bnd_functions['shallow_water'] = self.boundary_conditions[i]

        # NOTE: Extra setup must be done *before* setting initial condition
        if hasattr(self, 'extra_setup'):
            self.extra_setup(i)

        # Initial conditions
        uv, elev = self.fwd_solutions[i].split()
        self.fwd_solvers[i].assign_initial_conditions(uv=uv, elev=elev)

        # NOTE: Callbacks must be added *after* setting initial condition
        if hasattr(self, 'add_callbacks'):
            self.add_callbacks(i)

        # Ensure time level matches solver iteration
        self.fwd_solvers[i].i_export = i*self.dt_per_mesh//op.dt_per_export
        self.fwd_solvers[i].next_export_t = i*op.dt*self.dt_per_mesh
        self.fwd_solvers[i].iteration = i*self.dt_per_mesh
        self.fwd_solvers[i].simulation_time = i*op.dt*self.dt_per_mesh
        for e in self.fwd_solvers[i].exporters.values():
            e.set_next_export_ix(self.fwd_solvers[i].i_export)

        # For later use
        self.lhs = self.fwd_solvers[i].timestepper.F
        assert self.fwd_solutions[i].function_space() == self.fwd_solvers[i].function_spaces.V_2d

    def setup_solver_adjoint(self, i, **kwargs):
        """Setup adjoint solver on mesh `i`."""
        raise NotImplementedError("Should be implemented in derived class.")

    def solve_forward_step(self, i, update_forcings=None, export_func=None):
        """
        Solve forward PDE on mesh `i`.

        :kwarg update_forcings: a function which takes simulation time as an argument and is
            evaluated at the start of every timestep.
        :kwarg export_func: a function with no arguments which is evaluated at every export step.
        """
        update_forcings = update_forcings or self.op.get_update_forcings(self.fwd_solvers[i])
        export_func = export_func or self.op.get_export_func(self.fwd_solvers[i])

        def wrapped_export_func():
            """Extract forward solution and wrap the user-provided export function."""
            self.fwd_solutions[i].assign(self.fwd_solvers[i].fields.solution_2d)
            if export_func is not None:
                export_func()

        self.fwd_solvers[i].iterate(update_forcings=update_forcings, export_func=wrapped_export_func)
        self.fwd_solutions[i].assign(self.fwd_solvers[i].fields.solution_2d)

    def solve_adjoint_step(self, i, **kwargs):
        """Solve adjoint PDE on mesh `i`."""
        raise NotImplementedError("Should be implemented in derived class.")

    def solve(self, adjoint=False, **kwargs):
        """
        Solve the forward or adjoint problem (as specified by the `adjoint` boolean kwarg) on the
        full sequence of meshes.

        NOTE: The implementation contains a very simple checkpointing scheme, in the sense that
            the final solution computed on mesh `i` is stored in `self.fwd_solvers[i]` or
            `self.adj_solutions[i]`, as appropriate.
        """
        if adjoint:
            self.solve_adjoint(**kwargs)
        else:
            self.solve_forward(**kwargs)

    def solve_forward(self, reverse=False, **kwargs):
        """Solve forward problem on the full sequence of meshes."""
        R = range(self.num_meshes-1, -1, -1) if reverse else range(self.num_meshes)
        for i in R:
            self.transfer_forward_solution(i)
            self.setup_solver_forward(i)
            self.solve_forward_step(i, **kwargs)

    def solve_adjoint(self, reverse=True, **kwargs):
        """Solve adjoint problem on the full sequence of meshes."""
        R = range(self.num_meshes-1, -1, -1) if reverse else range(self.num_meshes)
        for i in R:
            self.transfer_adjoint_solution(i)
            self.setup_solver_adjoint(i)
            self.solve_adjoint_step(i, **kwargs)

    # --- Run scripts

    def run(self, **kwargs):
        """
        Run simulation using mesh adaptation approach specified by `self.approach`.

        For metric-based approaches, a fixed point iteration loop is used.
        """
        run_scripts = {

            # Non-adaptive
            'fixed_mesh': self.solve_forward,

            # Metric-based, no adjoint
            'hessian': self.run_hessian_based,

            # Metric-based with adjoint
            'dwp': self.run_dwp,
            'dwr': self.run_dwr,

            # Mesh movement
            'monge_ampere': self.run_moving_mesh,
        }
        try:
            run_scripts[self.approach](**kwargs)
        except KeyError:
            raise ValueError("Approach '{:s}' not recognised".format(self.approach))

    def run_hessian_based(self, **kwargs):
        """
        Adaptation loop for Hessian based approach.

        Field for adaptation is specified by `op.adapt_field`.

        Multiple fields can be combined using double-understrokes and either 'avg' for metric
        average or 'int' for metric intersection. We assume distributivity of intersection over
        averaging.

        For example, `adapt_field = 'elevation__avg__velocity_x__int__bathymetry'` would imply
        first intersecting the Hessians recovered from the x-component of velocity and bathymetry
        and then averaging the result with the Hessian recovered from the elevation.

        Stopping criteria:
          * iteration count > self.op.num_adapt;
          * relative change in element count < self.op.element_rtol;
          * relative change in quantity of interest < self.op.qoi_rtol.
        """
        op = self.op
        if op.adapt_field in ('all_avg', 'all_int'):
            c = op.adapt_field[-3:]
            op.adapt_field = "velocity_x__{:s}__velocity_y__{:s}__elevation".format(c, c)
        adapt_fields = ('__int__'.join(op.adapt_field.split('__avg__'))).split('__int__')
        if op.hessian_time_combination not in ('integrate', 'intersect'):
            msg = "Hessian time combination method '{:s}' not recognised."
            raise ValueError(msg.format(op.hessian_time_combination))

        for n in range(op.num_adapt):

            # Arrays to hold Hessians for each field on each window
            H_windows = [[Function(P1_ten) for P1_ten in self.P1_ten] for f in adapt_fields]

            if hasattr(self, 'hessian_func'):
                delattr(self, 'hessian_func')
            update_forcings = None
            export_func = None
            for i in range(self.num_meshes):

                # Transfer the solution from the previous mesh / apply initial condition
                self.transfer_forward_solution(i)

                if n < op.num_adapt-1:

                    # Create double L2 projection operator which will be repeatedly used
                    kwargs = {
                        'enforce_constraints': False,
                        'normalise': False,
                        'noscale': True,
                    }
                    recoverer = ShallowWaterHessianRecoverer(
                        self.V[i], op=op,
                        constant_fields={'bathymetry': self.bathymetry[i]}, **kwargs,
                    )

                    def hessian(sol, adapt_field):
                        fields = {'adapt_field': adapt_field, 'fields': self.fields[i]}
                        return recoverer.get_hessian_metric(sol, **fields, **kwargs)

                    # Array to hold time-integrated Hessian UFL expression
                    H_window = [0 for f in adapt_fields]

                    def update_forcings(t):
                        """Time-integrate Hessian using Trapezium Rule."""
                        it = self.fwd_solvers[i].iteration
                        if it % op.hessian_timestep_lag != 0:
                            return
                        first_ts = it == i*self.dt_per_mesh
                        final_ts = it == (i+1)*self.dt_per_mesh
                        dt = op.dt*op.hessian_timestep_lag
                        for j, f in enumerate(adapt_fields):
                            H = hessian(self.fwd_solvers[i].fields.solution_2d, f)
                            if op.hessian_time_combination == 'integrate':
                                H_window[j] += (0.5 if first_ts or final_ts else 1.0)*dt*H
                            elif f == 'bathymetry':
                                H_window[j] = H
                            else:
                                H_window[j] = H if first_ts else metric_intersection(H, H_window[j])

                    def export_func():
                        """
                        Extract time-averaged Hessian.

                        NOTE: We only care about the final export in each mesh iteration
                        """
                        if self.fwd_solvers[i].iteration == (i+1)*self.dt_per_mesh:
                            for j, H in enumerate(H_window):
                                if op.hessian_time_combination:
                                    H_window[j] *= op.dt*self.dt_per_mesh
                                H_windows[j][i].interpolate(H_window[j])

                # Solve step for current mesh iteration
                print_output("Solving forward equation for iteration {:d}".format(i))
                self.setup_solver_forward(i)
                self.solve_forward_step(i, export_func=export_func, update_forcings=update_forcings)

                # Delete objects to free memory
                if n < op.num_adapt-1:
                    del H_window
                    del recoverer

            # --- Convergence criteria

            # Check QoI convergence
            qoi = self.quantity_of_interest()
            print_output("Quantity of interest {:d}: {:.4e}".format(n+1, qoi))
            self.qois.append(qoi)
            if len(self.qois) > 1:
                if np.abs(self.qois[-1] - self.qois[-2]) < op.qoi_rtol*self.qois[-2]:
                    print_output("Converged quantity of interest!")
                    break

            # Check maximum number of iterations
            if n == op.num_adapt - 1:
                break

            # --- Time normalise metrics

            for j in range(len(adapt_fields)):
                space_time_normalise(H_windows[j], op=op)

            # Combine metrics (if appropriate)
            metrics = [Function(P1_ten, name="Hessian metric") for P1_ten in self.P1_ten]
            for i in range(self.num_meshes):
                H_window = [H_windows[j][i] for j in range(len(adapt_fields))]
                if 'int' in op.adapt_field:
                    if 'avg' in op.adapt_field:
                        raise NotImplementedError  # TODO: mixed case
                    metrics[i].assign(metric_intersection(*H_window))
                elif 'avg' in op.adapt_field:
                    metrics[i].assign(metric_average(*H_window))
                else:
                    try:
                        assert len(adapt_fields) == 1
                    except AssertionError:
                        msg = "Field for adaptation '{:s}' not recognised"
                        raise ValueError(msg.format(op.adapt_field))
                    metrics[i].assign(H_window[0])
            del H_windows

            # metric_file = File(os.path.join(self.di, 'metric.pvd'))
            complexities = []
            for i, M in enumerate(metrics):
                # metric_file.write(M)
                complexities.append(metric_complexity(M))
            self.st_complexities.append(sum(complexities)*op.end_time/op.dt)

            # --- Adapt meshes

            print_output("\nStarting mesh adaptation for iteration {:d}...".format(n+1))
            for i, M in enumerate(metrics):
                print_output("Adapting mesh {:d}/{:d}...".format(i+1, self.num_meshes))
                self.meshes[i] = pragmatic_adapt(self.meshes[i], M, op=op)
            del metrics
            self.num_cells.append([mesh.num_cells() for mesh in self.meshes])
            self.num_vertices.append([mesh.num_vertices() for mesh in self.meshes])
            print_output("Done!")

            # ---  Setup for next run / logging

            self.set_meshes(self.meshes)
            self.create_function_spaces()
            self.dofs.append([np.array(V.dof_count).sum() for V in self.V])
            self.create_solutions()
            self.set_fields()
            self.set_stabilisation()
            self.set_boundary_conditions()
            self.callbacks = [{} for mesh in self.meshes]

            print_output("\nResulting meshes")
            msg = "  {:2d}: complexity {:8.1f} vertices {:7d} elements {:7d}"
            for i, c in enumerate(complexities):
                print_output(msg.format(i, c, self.num_vertices[n+1][i], self.num_cells[n+1][i]))
            msg = "  total:            {:8.1f}          {:7d}          {:7d}\n"
            print_output(msg.format(
                self.st_complexities[-1],
                sum(self.num_vertices[n+1])*self.dt_per_mesh,
                sum(self.num_cells[n+1])*self.dt_per_mesh,
            ))

            # Check convergence of *all* element counts
            converged = True
            for i, num_cells_ in enumerate(self.num_cells[n-1]):
                if np.abs(self.num_cells[n][i] - num_cells_) > op.element_rtol*num_cells_:
                    converged = False
            if converged:
                print_output("Converged number of mesh elements!")
                break

    def get_checkpoints(self):
        """
        Run the forward model for the entire time period in order to get checkpoints.

        For mesh 0, the checkpoint is just the initial condition. For all other meshes, it is
        the final solution tuple on the previous mesh (which will need to be interpolated).
        """
        self.solution_file.__init__(self.solution_file.filename)
        for i in range(self.num_meshes):
            self.solution_file._topology = None
            proj = Function(self.P1[i], name="Projected elevation")

            def export_func():
                if self.op.family != 'taylor-hood':
                    proj.project(self.fwd_solvers[i].fields.elev_2d)
                    self.solution_file.write(proj)

            self.transfer_forward_solution(i)
            self.setup_solver_forward(i)
            self.fwd_solvers[i].export_initial_state = i == 0
            self.solve_forward_step(i, export_func=export_func)

    def run_dwp(self, **kwargs):
        r"""
        The "dual weighted primal" approach, first used (not under this name) in [1]. For shallow
        water tsunami propagation problems with a quantity of interest of the form

      ..math::
            J(u, \eta) = \int_{t_0}^{t_f} \int_R \eta \;\mathrm dx\;\mathrm dt,

        where :math:`eta` is free surface displacement and :math:`R\subset\Omega` is a spatial
        region of interest, it can be shown [1] that

      ..math::
            \int_R q(x, t=t_0) \cdot \hat q(x, t=t_0) \;\mathrm dx = \int_R q(x, t=t_f) \cdot \hat q(x, t=t_f) \;\mathrm dx

        under certain boundary condition assumptions. Here :math:`q=(u,\eta)` and :math:`\hat q`
        denotes the adjoint solution. Note that the choice of :math:`[t_0, t_f] \subseteq [0, T]`
        is arbitrary, so the above holds at all time levels.

        This motivates using error indicators of the form :math:`|q \cdot \hat q|`.

        [1] B. Davis & R. LeVeque, "Adjoint Methods for Guiding Adaptive Mesh Refinement in
            Tsunami Modelling", Pure and Applied Geophysics, 173, Springer International
            Publishing (2016), p.4055--4074, DOI 10.1007/s00024-016-1412-y.
        """
        op = self.op
        for n in range(op.num_adapt):

            # --- Solve forward to get checkpoints

            self.get_checkpoints()

            # --- Convergence criteria

            # Check QoI convergence
            qoi = self.quantity_of_interest()
            print_output("Quantity of interest {:d}: {:.4e}".format(n+1, qoi))
            self.qois.append(qoi)
            if len(self.qois) > 1:
                if np.abs(self.qois[-1] - self.qois[-2]) < op.qoi_rtol*self.qois[-2]:
                    print_output("Converged quantity of interest!")
                    break

            # Check maximum number of iterations
            if n == op.num_adapt - 1:
                break

            # --- Loop over mesh windows *in reverse*

            for i, P1 in enumerate(self.P1):
                self.indicators[i]['dwp'] = Function(P1, name="DWP indicator")
            metrics = [Function(P1_ten, name="Metric") for P1_ten in self.P1_ten]
            for i in range(self.num_meshes-1, -1, -1):
                fwd_solutions_step = []
                adj_solutions_step = []

                # --- Solve forward on current window

                def export_func():
                    fwd_solutions_step.append(self.fwd_solutions[i].copy(deepcopy=True))

                self.transfer_forward_solution(i)
                self.setup_solver_forward(i)
                self.solve_forward_step(i, export_func=export_func)

                # --- Solve adjoint on current window

                def export_func():
                    adj_solutions_step.append(self.adj_solutions[i].copy(deepcopy=True))

                self.transfer_adjoint_solution(i)
                self.setup_solver_adjoint(i)
                self.solve_adjoint_step(i, export_func=export_func)

                # --- Assemble indicators and metrics

                n_fwd = len(fwd_solutions_step)
                n_adj = len(adj_solutions_step)
                if n_fwd != n_adj:
                    msg = "Mismatching number of indicators ({:d} vs {:d})"
                    raise ValueError(msg.format(n_fwd, n_adj))
                I = 0
                op.print_debug("DWP indicators on mesh {:2d}".format(i))
                for j, solutions in enumerate(zip(fwd_solutions_step, reversed(adj_solutions_step))):
                    scaling = 0.5 if j in (0, n_fwd-1) else 1.0  # Trapezium rule  # TODO: Other integrators
                    fwd_dot_adj = abs(inner(*solutions))
                    op.print_debug("    ||<q, q*>||_L2 = {:.4e}".format(assemble(fwd_dot_adj*fwd_dot_adj*dx)))
                    I += op.dt*self.dt_per_mesh*scaling*fwd_dot_adj
                self.indicators[i]['dwp'].interpolate(I)
                metrics[i].assign(isotropic_metric(self.indicators[i]['dwp'], normalise=False))

            # --- Normalise metrics

            space_time_normalise(metrics, op=op)

            # Output to .pvd and .vtu
            # metric_file = File(os.path.join(self.di, 'metric.pvd'))
            complexities = []
            for i, M in enumerate(metrics):
                self.indicator_file._topology = None
                self.indicator_file.write(self.indicators[i]['dwp'])
                # metric_file._topology = None
                # metric_file.write(M)
                complexities.append(metric_complexity(M))
            self.st_complexities.append(sum(complexities)*op.end_time/op.dt)

            # --- Adapt meshes

            print_output("\nStarting mesh adaptation for iteration {:d}...".format(n+1))
            for i, M in enumerate(metrics):
                print_output("Adapting mesh {:d}/{:d}...".format(i+1, self.num_meshes))
                self.meshes[i] = pragmatic_adapt(self.meshes[i], M, op=op)
            del metrics
            self.num_cells.append([mesh.num_cells() for mesh in self.meshes])
            self.num_vertices.append([mesh.num_vertices() for mesh in self.meshes])
            print_output("Done!")

            # ---  Setup for next run / logging

            self.set_meshes(self.meshes)
            self.create_function_spaces()
            self.dofs.append([np.array(V.dof_count).sum() for V in self.V])
            self.create_solutions()
            self.set_fields()
            self.set_stabilisation()
            self.set_boundary_conditions()
            self.callbacks = [{} for mesh in self.meshes]

            print_output("\nResulting meshes")
            msg = "  {:2d}: complexity {:8.1f} vertices {:7d} elements {:7d}"
            for i, c in enumerate(complexities):
                print_output(msg.format(i, c, self.num_vertices[n+1][i], self.num_cells[n+1][i]))
            msg = "  total:            {:8.1f}          {:7d}          {:7d}\n"
            print_output(msg.format(
                self.st_complexities[-1],
                sum(self.num_vertices[n+1])*self.dt_per_mesh,
                sum(self.num_cells[n+1])*self.dt_per_mesh,
            ))

            # Check convergence of *all* element counts
            converged = True
            for i, num_cells_ in enumerate(self.num_cells[n-1]):
                if np.abs(self.num_cells[n][i] - num_cells_) > op.element_rtol*num_cells_:
                    converged = False
            if converged:
                print_output("Converged number of mesh elements!")
                break

    def run_dwr(self, **kwargs):
        # TODO: doc
        op = self.op
        for n in range(op.num_adapt):

            # --- Solve forward to get checkpoints

            self.get_checkpoints()

            # --- Convergence criteria

            # Check QoI convergence
            qoi = self.quantity_of_interest()
            print_output("Quantity of interest {:d}: {:.4e}".format(n+1, qoi))
            self.qois.append(qoi)
            if len(self.qois) > 1:
                if np.abs(self.qois[-1] - self.qois[-2]) < op.qoi_rtol*self.qois[-2]:
                    print_output("Converged quantity of interest!")
                    break

            # Check maximum number of iterations
            if n == op.num_adapt - 1:
                break

            # --- Setup problem on enriched space

            same_mesh = True
            for mesh in self.meshes:
                if mesh != self.meshes[0]:
                    same_mesh = False
                    break
            if same_mesh:
                print_output("All meshes are identical so we use an identical hierarchy.")
                hierarchy = MeshHierarchy(self.meshes[0], 1)
                refined_meshes = [hierarchy[1] for mesh in self.meshes]
            else:
                print_output("Meshes differ so we create separate hierarchies.")
                hierarchies = [MeshHierarchy(mesh, 1) for mesh in self.meshes]
                refined_meshes = [hierarchy[1] for hierarchy in hierarchies]
            ep = type(self)(op, refined_meshes, discrete_adjoint=self.discrete_adjoint)

            # --- Loop over mesh windows *in reverse*

            for i, P1 in enumerate(self.P1):
                self.indicators[i]['dwr'] = Function(P1, name="DWR indicator")
            metrics = [Function(P1_ten, name="Metric") for P1_ten in self.P1_ten]
            for i in range(self.num_meshes-1, -1, -1):
                fwd_solutions_step = []
                fwd_solutions_step_old = []
                adj_solutions_step = []
                enriched_adj_solutions_step = []
                tm = dmhooks.get_transfer_manager(self.meshes[i]._plex)

                # --- Setup forward solver for enriched problem

                # TODO: Need to transfer fwd sol in nonlinear case
                ep.setup_solver_forward(i)
                ets = ep.fwd_solvers[i].timestepper

                # --- Solve forward on current window

                def export_func():
                    fwd_solutions_step.append(ts.solution.copy(deepcopy=True))
                    fwd_solutions_step_old.append(ts.solution_old.copy(deepcopy=True))
                    # TODO: Also need store fields at each export (in general case)

                self.transfer_forward_solution(i)
                self.setup_solver_forward(i)
                ts = self.fwd_solvers[i].timestepper
                self.solve_forward_step(i, export_func=export_func)

                # --- Solve adjoint on current window

                def export_func():
                    adj_solutions_step.append(self.adj_solutions[i].copy(deepcopy=True))

                self.transfer_adjoint_solution(i)
                self.setup_solver_adjoint(i)
                self.solve_adjoint_step(i, export_func=export_func)

                # --- Solve adjoint on current window in enriched space

                def export_func():
                    enriched_adj_solutions_step.append(ep.adj_solutions[i].copy(deepcopy=True))

                ep.transfer_adjoint_solution(i)
                ep.setup_solver_adjoint(i)
                ep.solve_adjoint_step(i, export_func=export_func)

                # --- Assemble indicators and metrics

                n_fwd = len(fwd_solutions_step)
                n_adj = len(adj_solutions_step)
                if n_fwd != n_adj:
                    msg = "Mismatching number of indicators ({:d} vs {:d})"
                    raise ValueError(msg.format(n_fwd, n_adj))
                adj_solutions_step = list(reversed(adj_solutions_step))
                enriched_adj_solutions_step = list(reversed(enriched_adj_solutions_step))
                I = 0
                op.print_debug("DWR indicators on mesh {:2d}".format(i))
                indicator_enriched = Function(ep.P0[i])
                fwd_proj = Function(ep.V[i])
                fwd_old_proj = Function(ep.V[i])
                adj_error = Function(ep.V[i])
                bcs = self.fwd_solvers[i].bnd_functions['shallow_water']
                ets.setup_error_estimator(fwd_proj, fwd_old_proj, adj_error, bcs)

                # Loop over exported timesteps
                for j in range(len(fwd_solutions_step)):
                    scaling = 0.5 if j in (0, n_fwd-1) else 1.0  # Trapezium rule  # TODO: Other integrators

                    # Prolong forward solution at current and previous timestep
                    tm.prolong(fwd_solutions_step[j], fwd_proj)
                    tm.prolong(fwd_solutions_step_old[j], fwd_old_proj)

                    # Approximate adjoint error in enriched space
                    tm.prolong(adj_solutions_step[j], adj_error)
                    adj_error *= -1
                    adj_error += enriched_adj_solutions_step[j]

                    # Compute dual weighted residual
                    indicator_enriched.interpolate(abs(ets.error_estimator.weighted_residual()))

                    # Time-integrate
                    I += op.dt*self.dt_per_mesh*scaling*indicator_enriched
                indicator_enriched_cts = interpolate(I, ep.P1[i])

                ep.fwd_solvers[i] = None

                tm.inject(indicator_enriched_cts, self.indicators[i]['dwr'])
                metrics[i].assign(isotropic_metric(self.indicators[i]['dwr'], normalise=False))

            del indicator_enriched_cts
            del adj_error
            del indicator_enriched
            del ep
            del refined_meshes
            if same_mesh:
                del hierarchy
            else:
                del hierarchies

            # --- Normalise metrics

            space_time_normalise(metrics, op=op)

            # Output to .pvd and .vtu
            # metric_file = File(os.path.join(self.di, 'metric.pvd'))
            complexities = []
            for i, M in enumerate(metrics):
                self.indicator_file._topology = None
                self.indicator_file.write(self.indicators[i]['dwr'])
                # metric_file._topology = None
                # metric_file.write(M)
                complexities.append(metric_complexity(M))
            self.st_complexities.append(sum(complexities)*op.end_time/op.dt)

            # --- Adapt meshes

            print_output("\nStarting mesh adaptation for iteration {:d}...".format(n+1))
            for i, M in enumerate(metrics):
                print_output("Adapting mesh {:d}/{:d}...".format(i+1, self.num_meshes))
                self.meshes[i] = pragmatic_adapt(self.meshes[i], M, op=op)
            del metrics
            self.num_cells.append([mesh.num_cells() for mesh in self.meshes])
            self.num_vertices.append([mesh.num_vertices() for mesh in self.meshes])
            print_output("Done!")

            # ---  Setup for next run / logging

            self.set_meshes(self.meshes)
            self.create_function_spaces()
            self.dofs.append([np.array(V.dof_count).sum() for V in self.V])
            self.create_solutions()
            self.set_fields()
            self.set_stabilisation()
            self.set_boundary_conditions()
            self.callbacks = [{} for mesh in self.meshes]

            print_output("\nResulting meshes")
            msg = "  {:2d}: complexity {:8.1f} vertices {:7d} elements {:7d}"
            for i, c in enumerate(complexities):
                print_output(msg.format(i, c, self.num_vertices[n+1][i], self.num_cells[n+1][i]))
            msg = "  total:            {:8.1f}          {:7d}          {:7d}\n"
            print_output(msg.format(
                self.st_complexities[-1],
                sum(self.num_vertices[n+1])*self.dt_per_mesh,
                sum(self.num_cells[n+1])*self.dt_per_mesh,
            ))

            # Check convergence of *all* element counts
            converged = True
            for i, num_cells_ in enumerate(self.num_cells[n-1]):
                if np.abs(self.num_cells[n][i] - num_cells_) > op.element_rtol*num_cells_:
                    converged = False
            if converged:
                print_output("Converged number of mesh elements!")
                break

    def run_moving_mesh(self, **kwargs):
        try:
            assert hasattr(self, 'monitor')
        except AssertionError:
            raise AttributeError("Cannot perform mesh movement without a monitor function.")
        raise NotImplementedError  # TODO

    # --- Metric

    def get_hessian_metric(self, adjoint=False, **kwargs):
        kwargs.setdefault('normalise', True)
        kwargs['op'] = self.op
        self.metrics = []
        solutions = self.adj_solutions if adjoint else self.fwd_solutions
        for i, sol in enumerate(solutions):
            fields = {'bathymetry': self.bathymetry[i], 'inflow': self.inflow[i]}
            self.metrics.append(get_hessian_metric(sol, fields=fields, **kwargs))

    # --- Goal-oriented

    def get_qoi_kernels(self, i):
        self.kernels[i] = self.op.set_qoi_kernel(self.V[i])

    def get_bnd_functions(self, i, *args):
        swt = shallowwater_eq.ShallowWaterTerm(self.V[i], self.bathymetry)
        return swt.get_bnd_functions(*args, self.boundary_conditions[i])

    def get_strong_residual_forward(self):
        raise NotImplementedError  # TODO: Use thetis/error_estimation_2d

    def get_dwr_residual_forward(self):
        raise NotImplementedError  # TODO: Use thetis/error_estimation_2d

    def get_dwr_flux_forward(self):
        raise NotImplementedError  # TODO: Use thetis/error_estimation_2d

    def dwr_indication(self, adjoint=False):
        """
        Indicate errors in the quantity of interest by the Dual Weighted Residual method. This is
        inherently problem-dependent.
        """
        raise NotImplementedError("Should be implemented in derived class.")

    def quantity_of_interest(self):
        """Functional of interest which takes the PDE solution as input."""
        raise NotImplementedError("Should be implemented in derived class.")
