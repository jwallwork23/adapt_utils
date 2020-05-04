from thetis import *
from firedrake.petsc import PETSc

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

    def __init__(self, op, meshes=None, discrete_adjoint=True, levels=0, hierarchies=None):
        op.print_debug(op.indent + "{:s} initialisation begin".format(self.__class__.__name__))

        # Read args and kwargs
        self.op = op
        self.stabilisation = op.stabilisation
        self.discrete_adjoint = discrete_adjoint
        self.approach = op.approach
        self.levels = levels
        if levels > 0:
            raise NotImplementedError  # TODO

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
        self.indicator_file = File(os.path.join(self.di, 'indicator.pvd'))

        # Storage for diagnostics over mesh adaptation loop
        self.num_cells = [[mesh.num_cells() for mesh in self.meshes], ]
        self.num_vertices = [[mesh.num_vertices() for mesh in self.meshes], ]
        self.dofs = [[np.array(V.dof_count).sum() for V in self.V], ]
        self.indicators = [{} for mesh in self.meshes]
        self.estimators = [{} for mesh in self.meshes]
        self.qois = []
        self.st_complexities = [np.nan]

    # TODO: AdaptiveMesh
    # TODO: levels > 0
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
        u_element = VectorElement("DG", triangle, p)
        if family == 'dg-dg':
            eta_element = FiniteElement("DG", triangle, p, variant='equispaced')
        elif family == 'dg-cg':
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

        # Shallow water space
        self.V = [FunctionSpace(mesh, self.finite_element) for mesh in self.meshes]

        # Tracer space
        self.Q = self.P1DG

    def create_solutions(self):
        """
        Set up `Function`s in the prognostic space to hold the forward and adjoint solutions.
        """
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
        self.bathymetry = [self.op.set_bathymetry(P1DG) for P1DG in self.P1DG]
        self.inflow = [self.op.set_inflow(P1_vec) for P1_vec in self.P1_vec]

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
        """Solve forward PDE on mesh `i`."""
        update_forcings = update_forcings or self.op.get_update_forcings(self.fwd_solvers[i])
        export_func = export_func or self.op.get_export_func(self.fwd_solvers[i])
        self.fwd_solvers[i].iterate(update_forcings=update_forcings, export_func=export_func)
        self.fwd_solutions[i].assign(self.fwd_solvers[i].fields.solution_2d)

    def solve_adjoint_step(self, i, **kwargs):
        """Solve adjoint PDE on mesh `i`."""
        raise NotImplementedError("Should be implemented in derived class.")

    def solve(self, adjoint=False):
        if adjoint:
            self.solve_adjoint()
        else:
            self.solve_forward()

    def solve_forward(self):
        """Solve forward problem on the full sequence of meshes."""
        for i in range(self.num_meshes):
            self.transfer_forward_solution(i)
            self.setup_solver_forward(i)
            self.solve_forward_step(i)

    def solve_adjoint(self):
        """Solve adjoint problem on the full sequence of meshes."""
        for i in range(self.num_meshes - 1, -1):
            self.transfer_adjoint_solution(i)
            self.setup_solver_adjoint(i)
            self.solve_adjoint_step(i)

    # --- Run scripts

    def run(self):
        if self.approach == 'fixed_mesh':
            self.solve_forward()
        elif self.approach == 'hessian':
            self.run_hessian_based()
        else:
            raise NotImplementedError  # TODO

    def run_hessian_based(self):
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
                        first_timestep = self.fwd_solvers[i].iteration == 0
                        final_timestep = self.fwd_solvers[i].iteration == (i+1)*self.dt_per_mesh
                        weight = (0.5 if first_timestep or final_timestep else 1.0)*op.dt
                        for j, f in enumerate(adapt_fields):
                            H_window[j] += weight*hessian(self.fwd_solvers[i].fields.solution_2d, f)
                        # TODO: Hessian intersection option

                    def export_func():

                        # We only care about the final export in each mesh iteration
                        if self.fwd_solvers[i].iteration != (i+1)*self.dt_per_mesh:
                            return

                        # Extract time averaged Hessian
                        for j, H in enumerate(H_window):
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

            metric_file = File(os.path.join(self.di, 'metric.pvd'))
            complexities = []
            for i, M in enumerate(metrics):
                metric_file.write(M)
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
            print_output("")

            # Check convergence of *all* element counts
            converged = True
            for i, num_cells_ in enumerate(self.num_cells[n-1]):
                if np.abs(self.num_cells[n][i] - num_cells_) > op.element_rtol*num_cells_:
                    converged = False
            if converged:
                print_output("Converged number of mesh elements!")
                break

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
        self.kernels[i] = self.op.set_qoi_kernel(self.fwd_solvers[i])

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
