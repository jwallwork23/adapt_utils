from thetis import *
from thetis.configuration import *

from adapt_utils.swe.solver import UnsteadyShallowWaterProblem
# from adapt_utils.swe.tsunami.qois import InundationCallback


__all__ = ["TsunamiProblem"]


class TsunamiProblem(UnsteadyShallowWaterProblem):
    """
    For general tsunami propagation problems.
    """
    def __init__(self, *args, extension=None, **kwargs):
        self.extension = extension
        super().__init__(*args, **kwargs)
        # self.nonlinear = False  # TODO

    def set_fields(self):
        self.fields = {}
        self.fields['viscosity'] = self.op.set_viscosity(self.P1)
        self.fields['diffusivity'] = self.op.set_diffusivity(self.P1)
        self.fields['bathymetry'] = self.op.set_bathymetry()
        self.fields['coriolis'] = self.op.set_coriolis(self.P1)
        self.fields['quadratic_drag_coefficient'] = self.op.set_quadratic_drag_coefficient(self.P1DG)
        self.fields['manning_drag_coefficient'] = self.op.set_manning_drag_coefficient(self.P1)
        self.fields['source'] = self.op.source

    def extra_setup(self):
        op = self.op

        # Don't bother plotting velocity
        # self.solver_obj.options.fields_to_export = ['elev_2d'] if op.plot_pvd else []  # TODO
        self.solver_obj.options.fields_to_export = ['elev_2d', 'uv_2d'] if op.plot_pvd else []
        self.solver_obj.options.fields_to_export_hdf5 = ['elev_2d'] if op.save_hdf5 else []

        # Set callbacks to save gauge timeseries to HDF5
        self.callbacks = {}
        locs = [op.gauges[g]["coords"] for g in op.gauges]
        names = list(op.gauges.keys())
        fname = "gauges"
        if self.extension is not None:
            fname = '_'.join([fname, self.extension])
        fname = '_'.join([fname, str(self.num_cells[-1])])
        for g in op.gauges:
            self.callbacks[g] = callback.DetectorsCallback(
                self.solver_obj, locs, ['elev_2d'], fname, names)
            self.solver_obj.add_callback(self.callbacks[g], 'export')

        # TODO: QoI
        # # Set callback for QoI evaluation
        # self.callbacks["qoi"] = InundationCallback(self.solver_obj)
        # self.solver_obj.add_callback(self.callbacks["qoi"])

    # TODO: Reduce duplication (this was copied and modified from swe/solver)
    def setup_solver_forward(self):
        if not hasattr(self, 'remesh_step'):
            self.remesh_step = 0
        op = self.op

        # Interpolate bathymetry from data
        self.fields['bathymetry'] = self.op.set_bathymetry()

        # Create new solver
        self.solver_obj = solver2d.FlowSolver2d(self.mesh, self.fields['bathymetry'])
        self.solver_obj.export_initial_state = self.remesh_step == 0
        options = self.solver_obj.options
        options.use_nonlinear_equations = self.nonlinear
        options.check_volume_conservation_2d = True
        if hasattr(options, 'use_lagrangian_formulation'):  # TODO: Temporary
            options.use_lagrangian_formulation = op.approach == 'ale'

        # Timestepping
        options.timestep = op.dt
        options.simulation_export_time = op.dt*op.dt_per_export
        options.simulation_end_time = self.step_end - 0.5*op.dt
        options.timestepper_type = op.timestepper
        if op.params != {}:
            options.timestepper_options.solver_parameters = op.params
        if self.nonlinear:
            options.timestepper_options.solver_parameters['snes_type'] = 'ksponly'  # TODO: Check
        if op.debug:
            # options.timestepper_options.solver_parameters['snes_monitor'] = None
            # options.timestepper_options.solver_parameters['snes_converged_reason'] = None
            # options.timestepper_options.solver_parameters['ksp_monitor'] = None
            # options.timestepper_options.solver_parameters['ksp_converged_reason'] = None
            print_output(options.timestepper_options.solver_parameters)
        if hasattr(options.timestepper_options, 'implicitness_theta'):
            options.timestepper_options.implicitness_theta = op.implicitness_theta

        # Outputs
        options.output_directory = self.di

        # Parameters
        options.use_grad_div_viscosity_term = op.grad_div_viscosity
        options.element_family = op.family
        options.horizontal_viscosity = self.fields['viscosity']
        options.horizontal_diffusivity = self.fields['diffusivity']
        options.quadratic_drag_coefficient = self.fields['quadratic_drag_coefficient']
        options.manning_drag_coefficient = self.fields['manning_drag_coefficient']
        options.coriolis_frequency = self.fields['coriolis']
        options.use_lax_friedrichs_velocity = self.stabilisation == 'lax_friedrichs'
        options.lax_friedrichs_velocity_scaling_factor = self.stabilisation_parameter
        options.use_grad_depth_viscosity_term = op.grad_depth_viscosity
        options.use_automatic_sipg_parameter = op.sipg_parameter is None
        options.use_wetting_and_drying = op.wetting_and_drying
        options.wetting_and_drying_alpha = op.wetting_and_drying_alpha
        options.solve_tracer = op.solve_tracer

        # Boundary conditions
        self.solver_obj.bnd_functions['shallow_water'] = op.set_boundary_conditions(self.V)

        # Initial conditions
        u_interp, eta_interp = self.solution.split()
        self.solver_obj.assign_initial_conditions(uv=u_interp, elev=eta_interp)

        if hasattr(self, 'extra_setup'):
            self.extra_setup()

        # Ensure correct iteration count
        self.solver_obj.i_export = self.remesh_step
        self.solver_obj.next_export_t = self.remesh_step*op.dt*op.dt_per_remesh
        self.solver_obj.iteration = self.remesh_step*op.dt_per_remesh
        self.solver_obj.simulation_time = self.remesh_step*op.dt*op.dt_per_remesh
        if hasattr(self.solver_obj, 'exporters'):
            for e in self.solver_obj.exporters.values():
                e.set_next_export_ix(self.solver_obj.i_export)
