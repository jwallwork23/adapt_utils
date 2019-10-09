from thetis import *
from thetis.physical_constants import *
from firedrake.petsc import PETSc
import math

from adapt_utils.solver import UnsteadyProblem


__all__ = ["UnsteadyShallowWaterProblem"]


class UnsteadyShallowWaterProblem(UnsteadyProblem):
    # TODO: doc
    def __init__(self, op, mesh=None, discrete_adjoint=True):
        if op.family == 'dg-dg' and op.degree in (1, 2):
            element = VectorElement("DG", triangle, 1)*FiniteElement("DG", triangle, op.degree)
        elif op.family == 'dg-cg':
            element = VectorElement("DG", triangle, 1)*FiniteElement("Lagrange", triangle, 2)
        else:
            raise NotImplementedError
        if mesh is None:
            mesh = op.default_mesh
        super(UnsteadyShallowWaterProblem, self).__init__(mesh, op, element, discrete_adjoint)

        # Stabilisation
        if self.stab is not None:
            try:
                assert self.stab == 'lax_friedrichs'
            except:
                raise NotImplementedError

        # Classification
        self.nonlinear = True

        # Set ICs
        self.solution = Function(self.V)
        self.solution.assign(op.set_initial_condition(self.V))

        # Gravitational constant
        physical_constants['g_grav'].assign(op.g)


    def solve_step(self, **kwargs):
        op = self.op
        solver_obj = solver2d.FlowSolver2d(self.mesh, op.bathymetry)
        options = solver_obj.options
        options.use_nonlinear_equations = True
        options.check_volume_conservation_2d = True

        # Timestepping
        options.timestep = op.dt
        options.simulation_export_time = op.dt*op.dt_per_export
        options.simulation_end_time = self.step_end - 0.5*op.dt
        options.timestepper_type = op.timestepper
        #options.timestepper_options.solver_parameters = op.params
        PETSc.Sys.Print(options.timestepper_options.solver_parameters)

        # Outputs
        options.output_directory = self.di
        if op.plot_pvd:
            options.fields_to_export = ['uv_2d', 'elev_2d']
        else:
            options.no_exports = True

        # Parameters
        options.use_grad_div_viscosity_term = op.symmetric_viscosity
        options.element_family = op.family
        options.horizontal_viscosity = op.viscosity
        options.quadratic_drag_coefficient = op.drag_coefficient
        options.coriolis_frequency = op.set_coriolis(self.P1)
        options.use_lax_friedrichs_velocity = self.stab == 'lax_friedrichs'
        options.use_grad_depth_viscosity_term = False
        #options.use_grad_depth_viscosity_term = True
        #options.compute_residuals = True

        # Boundary conditions
        solver_obj.bnd_functions['shallow_water'] = op.set_bcs()

        # Initial conditions
        uv, elev = self.solution.split()
        solver_obj.assign_initial_conditions(uv=uv, elev=elev)

        # Ensure correct iteration count
        solver_obj.i_export = self.remesh_step
        solver_obj.next_export_t = self.remesh_step*op.dt*op.dt_per_remesh
        solver_obj.iteration = self.remesh_step*op.dt_per_remesh
        solver_obj.simulation_time = self.remesh_step*op.dt*op.dt_per_remesh
        for e in solver_obj.exporters.values():
            e.set_next_export_ix(solver_obj.i_export)

        # Solve
        solver_obj.iterate()
        self.solution.assign(solver_obj.fields.solution_2d)
        self.ts = solver_obj.timestepper
