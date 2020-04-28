from thetis import *
from thetis.physical_constants import *

import numpy as np

from adapt_utils.adapt.adaptation import AdaptiveMesh
from adapt_utils.swe.utils import *


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
        self.num_meshes = op.num_meshes
        op.print_debug(op.indent + "SETUP: Building meshes...")
        self.set_meshes(meshes)
        op.print_debug(op.indent + "SETUP: Creating function spaces...")
        self.create_function_spaces()
        op.print_debug(op.indent + "SETUP: Creating solutions...")
        self.create_solutions()
        op.print_debug(op.indent + "SETUP: Creating fields...")
        self.set_fields()
        op.print_debug(op.indent + "SETUP: Setting stabilisation parameters...")
        self.set_stabilisation()
        op.print_debug(op.indent + "SETUP: Setting boundary conditions...")
        self.boundary_conditions = [op.set_boundary_conditions(V) for V in self.V]
        self.fwd_solvers = [None for mesh in self.meshes]  # To be populated
        self.adj_solvers = [None for mesh in self.meshes]  # To be populated

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

    def set_meshes(self, meshes):  # TODO: levels > 0
        """
        Build an class:`AdaptiveMesh` object associated with each mesh.
        """
        self.meshes = meshes or [self.op.default_mesh for i in range(self.num_meshes)]
        self.ams = [AdaptiveMesh(self.meshes[i], levels=self.levels) for i in range(self.num_meshes)]
        msg = self.op.indent + "SETUP: Mesh {:d} has {:d} elements"
        for i in range(self.num_meshes):
            self.op.print_debug(msg.format(i, self.meshes[i].num_cells()))

    def create_function_spaces(self):  # NOTE: Keep minimal
        """
        Build the finite element space, `V`, for the prognostic solution, along with various other
        useful spaces and test functions and trial functions based upon them.
        """
        self.P0 = [FunctionSpace(mesh, "DG", 0) for mesh in self.meshes]
        self.P1 = [FunctionSpace(mesh, "CG", 1) for mesh in self.meshes]
        self.P1_vec = [VectorFunctionSpace(mesh, "CG", 1) for mesh in self.meshes]
        # self.P1_ten = [TensorFunctionSpace(mesh, "CG", 1) for mesh in self.meshes]
        self.P1DG = [FunctionSpace(mesh, "DG", 1) for mesh in self.meshes]
        # self.P1DG_vec = [VectorFunctionSpace(mesh, "DG", 1) for mesh in self.meshes]
        # self.P2 = [FunctionSpace(mesh, "CG", 2) for mesh in self.meshes]

        # Shallow water space
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
        self.V = [FunctionSpace(mesh, self.finite_element) for mesh in self.meshes]

        # Tracer space
        self.Q = self.P1DG

        # self.test = [TestFunction(V) for V in self.V]
        # self.tests = [TestFunctions(V) for V in self.V]
        # self.trial = [TrialFunction(V) for V in self.V]
        # self.trials = [TrialFunctions(V) for V in self.V]
        # self.p0test = [TestFunction(P0) for P0 in self.P0]
        # self.p0trial = [TrialFunction(P0) for P0 in self.P0]

    def create_solutions(self):
        """
        Set up `Function`s in the prognostic space to hold the forward and adjoint solutions.
        """
        self.fwd_solutions = []
        self.adj_solutions = []
        for i, V in enumerate(self.V):
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
        self.bathymetry = []
        self.inflow = []
        for i in range(self.num_meshes):
            self.fields.append({
                'horizontal_viscosity': self.op.set_viscosity(self.P1[i]),
                'horizontal_diffusivity': self.op.set_diffusivity(self.P1[i]),
                'coriolis_frequency': self.op.set_coriolis(self.P1[i]),
                'quadratic_drag_coefficient': self.op.set_quadratic_drag_coefficient(self.P1[i]),
                'manning_drag_coefficient': self.op.set_manning_drag_coefficient(self.P1[i]),
            })
            self.bathymetry.append(self.op.set_bathymetry(self.P1DG[i]))
            self.inflow.append(self.op.set_inflow(self.P1_vec[i]))

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

    def set_initial_condition(self):
        """Apply initial condition for forward solution on first mesh."""
        self.fwd_solutions[0].assign(self.op.set_start_condition(self.V[0], adjoint=False))

    def set_final_condition(self):
        """Apply final time condition for adjoint solution on final mesh."""
        self.adj_solutions[-1].assign(self.op.set_start_condition(self.V[-1], adjoint=True))

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

    # --- Solvers

    def setup_solver_forward(self, i, extra_setup=None):
        """Setup forward solver on mesh `i`."""
        op = self.op
        if extra_setup is not None:
            self.extra_setup = extra_setup

        # Create solver object
        solver_obj = solver2d.FlowSolver2d(self.meshes[i], self.bathymetry[i])
        self.fwd_solvers[i] = solver_obj
        options = solver_obj.options
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
        solver_obj.bnd_functions['shallow_water'] = self.boundary_conditions[i]

        # NOTE: Extra setup must be done *before* setting initial condition
        if hasattr(self, 'extra_setup'):
            self.extra_setup(i)

        # Initial conditions
        uv, elev = self.fwd_solutions[i].split()
        solver_obj.assign_initial_conditions(uv=uv, elev=elev)

        # NOTE: Callbacks must be added *after* setting initial condition
        if hasattr(self, 'add_callbacks'):
            self.add_callbacks(i)

        # Ensure time level matches solver iteration
        solver_obj.i_export = i
        solver_obj.next_export_t = i*op.dt*op.dt_per_remesh
        solver_obj.iteration = i*op.dt_per_remesh
        solver_obj.simulation_time = i*op.dt*op.dt_per_remesh
        for e in solver_obj.exporters.values():
            e.set_next_export_ix(solver_obj.i_export)

        # For later use
        self.lhs = solver_obj.timestepper.F
        # TODO: Check
        assert self.fwd_solutions[i].function_space() == solver_obj.function_spaces.V_2d

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
            if i == 0:
                self.set_initial_condition()
            else:
                self.project_forward_solution(i-1, i)
            self.setup_solver_forward(i)
            self.solve_forward_step(i)

    def solve_adjoint(self):
        """Solve adjoint problem on the full sequence of meshes."""
        for i in range(self.num_meshes - 1, -1):
            if i == self.num_meshes - 1:
                self.set_final_condition()
            else:
                self.project_adjoint_solution(i+1, i)
            self.setup_solver_adjoint(i)
            self.solve_adjoint_step(i)

    # --- Metric

    def get_hessian_metric(self, adjoint=False, **kwargs):
        kwargs.setdefault('noscale', False)
        kwargs['op'] = self.op
        self.metrics = []
        for i in range(self.num_meshes):
            sol = self.adj_solutions[i] if adjoint else self.fwd_solutions[i]
            fields = {'bathymetry': self.bathymetry[i], 'inflow': self.inflow[i]}
            self.metrics.append(get_hessian_metric(sol, fields=fields, **kwargs))

    # --- Goal-oriented

    def get_qoi_kernels(self):
        self.kernels = [self.op.set_qoi_kernel(solver) for solver in self.fwd_solvers]

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
