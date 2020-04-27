from thetis import *
from thetis.physical_constants import *

from adapt_utils.adapt.solver import AdaptiveProblem
from adapt_utils.swe.utils import *


__all__ = ["AdaptiveShallowWaterProblem"]


# TODO: Account for tracers
class AdaptiveShallowWaterProblem(AdaptiveProblem):
    """
    General solver object for adaptive shallow water problems.
    """

    # --- Setup

    def __init__(self, op, mesh=None, **kwargs):
        p = op.degree
        u_element = VectorElement("DG", triangle, p)
        if op.family == 'dg-dg' and p >= 0:
            fe = u_element*FiniteElement("DG", triangle, p, variant='equispaced')
        elif op.family == 'dg-cg' and p >= 0:
            fe = u_element*FiniteElement("Lagrange", triangle, p+1, variant='equispaced')
        else:
            raise NotImplementedError
        super(AdaptiveShallowWaterProblem, self).__init__(op, mesh, fe, **kwargs)

        # Physical parameters
        physical_constants['g_grav'].assign(op.g)

        # Classification
        self.shallow_water_options['use_nonlinear_equations'] = True

    def set_fields(self):
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

    def create_solutions(self):
        super(AdaptiveShallowWaterProblem, self).create_solutions()
        for i in range(self.num_meshes):
            u, eta = self.fwd_solutions[i].split()
            u.rename("Fluid velocity")
            eta.rename("Elevation")
            z, zeta = self.adj_solutions[i].split()
            z.rename("Adjoint fluid velocity")
            zeta.rename("Adjoint elevation")

    def set_stabilisation(self):  # TODO: Allow different / mesh dependent stabilisation parameters
        self.stabilisation = self.stabilisation or 'no'
        try:
            assert self.stabilisation in ('no', 'lax_friedrichs')
        except AssertionError:
            msg = "Stabilisation method {:s} not recognised for {:s}"
            raise ValueError(msg.format(self.stabilisation, self.__class__.__name__))
        self.stabilisation_parameters = []
        for i in range(self.num_meshes):
            self.stabilisation_parameters.append(self.op.stabilisation_parameter)

    # --- Solvers

    def setup_solver_forward(self, i, extra_setup=None):
        """
        Create a Thetis FlowSolver2d object for solving the shallow water equations on mesh `i`.
        """
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

        # Parameters  # TODO: Put parameters in op.shallow_water/tracer and use update
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

    def solve_forward_step(self, i, update_forcings=None, export_func=None):
        update_forcings = update_forcings or self.op.get_update_forcings(self.fwd_solvers[i])
        export_func = export_func or self.op.get_export_func(self.fwd_solvers[i])
        self.fwd_solvers[i].iterate(update_forcings=update_forcings, export_func=export_func)
        self.fwd_solutions[i].assign(self.fwd_solvers[i].fields.solution_2d)

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
        self.kernels = []
        for i in range(self.num_meshes):
            self.kernels.append(self.op.set_qoi_kernel(self.fwd_solvers[i]))

    def get_bnd_functions(self, i, *args):
        swt = shallowwater_eq.ShallowWaterTerm(self.V[i], self.bathymetry)
        return swt.get_bnd_functions(*args, self.boundary_conditions[i])

    def get_strong_residual_forward(self):
        raise NotImplementedError  # TODO: Use thetis/error_estimation_2d

    def get_dwr_residual_forward(self):
        raise NotImplementedError  # TODO: Use thetis/error_estimation_2d

    def get_dwr_flux_forward(self):
        raise NotImplementedError  # TODO: Use thetis/error_estimation_2d
