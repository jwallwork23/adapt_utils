from thetis import *
from firedrake.petsc import PETSc

from adapt_utils.solver import *  # TODO: Temporary
from adapt_utils.swe.solver import *
from adapt_utils.turbine.options import *
from adapt_utils.adapt.recovery import *
from adapt_utils.adapt.metric import *


__all__ = ["SteadyTurbineProblem", "UnsteadyTurbineProblem"]


class SteadyTurbineProblem(SteadyShallowWaterProblem):
    """
    General solver object for stationary tidal turbine problems.
    """
    def extra_setup(self):
        """
        We haven't meshed the turbines with separate ids, so define a farm everywhere and make it
        have a density of 1/D^2 inside the DxD squares where the turbines are and 0 outside.
        """
        op = self.op
        num_turbines = len(op.region_of_interest)
        scaling = num_turbines/assemble(op.bump(self.P1)*dx)
        self.turbine_density = op.bump(self.P1, scale=scaling)
        self.farm_options = TidalTurbineFarmOptions()
        self.farm_options.turbine_density = self.turbine_density
        self.farm_options.turbine_options.diameter = op.turbine_diameter
        self.farm_options.turbine_options.thrust_coefficient = op.thrust_coefficient

        A_T = pi*(op.turbine_diameter/2.0)**2
        self.C_D = op.thrust_coefficient*A_T*self.turbine_density/2.0

        # Turbine drag is applied everywhere (where the turbine density isn't zero)
        self.solver_obj.options.tidal_turbine_farms["everywhere"] = self.farm_options

        # Callback that computes average power
        self.cb = turbines.TurbineFunctionalCallback(self.solver_obj)
        self.solver_obj.add_callback(self.cb, 'timestep')

    def extra_residual_terms(self, u, eta, z, zeta):
        H = self.op.bathymetry + eta
        return -self.C_D*sqrt(dot(u, u))*inner(z, u)/H

    def get_callbacks(self, cb):
        self.qoi = cb.average_power

    def get_qoi_kernel(self):
        self.kernel = Function(self.V)
        u = self.solution.split()[0]
        k_u = self.kernel.split()[0]
        k_u.interpolate(Constant(1/3)*self.turbine_density*sqrt(inner(u, u))*u)

    def quantity_of_interest(self):
        return self.qoi

    def quantity_of_interest_form(self):
        return self.C_D*pow(inner(split(self.solution)[0], split(self.solution)[0]), 1.5)*dx

    def custom_adapt(self):
        if self.approach == 'vorticity':
            self.indicator = Function(self.P1, name='vorticity')
            self.indicator.interpolate(curl(self.solution.split()[0]))
            self.get_isotropic_metric()


# TODO: make a subclass of UnsteadyShallowWaterProblem
class UnsteadyTurbineProblem(UnsteadyProblem):
    # TODO: doc
    def __init__(self,
                 op=UnsteadyTurbineOptions(),
                 mesh=None,
                 discrete_adjoint=True):
        if op.family == 'dg-dg' and op.degree in (1, 2):
            element = VectorElement("DG", triangle, 1)*FiniteElement("DG", triangle, op.degree)
        elif op.family == 'dg-cg':
            element = VectorElement("DG", triangle, 1)*FiniteElement("Lagrange", triangle, 2)
        else:
            raise NotImplementedError
        if mesh is None:
            mesh = op.default_mesh
        super(UnsteadyTurbineProblem, self).__init__(mesh, op, element, discrete_adjoint)

        # Stabilisation
        if self.stab is not None:
            try:
                assert self.stab == 'lax_friedrichs'
            except:
                raise NotImplementedError

        # Physical fields
        self.set_fields()

        # Parameters for adjoint computation
        self.gradient_field = self.op.bathymetry
        z, zeta = self.adjoint_solution.split()
        z.rename("Adjoint fluid velocity")
        zeta.rename("Adjoint elevation")

        # Classification
        self.nonlinear = True

        # Set ICs
        self.solution = op.set_initial_condition(self.V)

    def set_fields(self):
        self.viscosity = self.op.set_viscosity(self.P1)
        self.drag_coefficient = Constant(self.op.drag_coefficient)
        self.op.set_boundary_surface(self.V.sub(1))

    def solve_step(self):
        self.set_fields()
        op = self.op
        solver_obj = solver2d.FlowSolver2d(self.mesh, op.bathymetry)
        options = solver_obj.options
        options.use_nonlinear_equations = True
        options.check_volume_conservation_2d = True

        # Timestepping
        options.timestep = op.dt
        options.simulation_export_time = op.dt*op.dt_per_export
        options.simulation_end_time = self.step_end-0.5*op.dt
        options.timestepper_type = op.timestepper
        options.timestepper_options.solver_parameters = op.params
        # PETSc.Sys.Print(options.timestepper_options.solver_parameters)
        # options.timestepper_options.implicitness_theta = 1.0

        # Outputs
        options.output_directory = self.di
        options.fields_to_export = ['uv_2d', 'elev_2d']

        # Parameters
        options.use_grad_div_viscosity_term = op.symmetric_viscosity
        options.element_family = op.family
        options.horizontal_viscosity = op.viscosity
        options.quadratic_drag_coefficient = self.drag_coefficient
        options.use_lax_friedrichs_velocity = self.stab == 'lax_friedrichs'
        # options.use_grad_depth_viscosity_term = False
        options.use_grad_depth_viscosity_term = True
        options.compute_residuals = True

        # Boundary conditions
        op.set_bcs()
        solver_obj.bnd_functions['shallow_water'] = op.boundary_conditions
        def update_forcings(t):
            op.elev_in.assign(op.hmax*cos(op.omega*(t-op.T_ramp)))
            op.elev_out.assign(op.hmax*cos(op.omega*(t-op.T_ramp)+pi))
        update_forcings(0.)

        # Tidal farm
        num_turbines = len(op.region_of_interest)
        if num_turbines > 0:
            # We haven't meshed the turbines with separate ids, so define a farm everywhere
            # and make it have a density of 1/D^2 inside the DxD squares where the turbines are
            # and 0 outside
            scaling = num_turbines/assemble(op.bump(self.P1)*dx)
            self.turbine_density = op.bump(self.P1, scale=scaling)
            # File(self.di+'Bump.pvd').write(turbine_density)
            farm_options = TidalTurbineFarmOptions()
            farm_options.turbine_density = self.turbine_density
            farm_options.turbine_options.diameter = op.turbine_diameter
            farm_options.turbine_options.thrust_coefficient = op.thrust_coefficient
            # Turbine drag is applied everywhere (where the turbine density isn't zero)
            options.tidal_turbine_farms["everywhere"] = farm_options

            # Callback that computes average power
            cb = turbines.TurbineFunctionalCallback(solver_obj)
            solver_obj.add_callback(cb, 'timestep')

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
        solver_obj.iterate(update_forcings=update_forcings)
        self.solution.assign(solver_obj.fields.solution_2d)
        if num_turbines > 0:
            self.qoi = cb.average_power
        self.ts = solver_obj.timestepper

    def quantity_of_interest(self):
        return self.qoi

    # TODO: update
    def get_hessian_metric(self, adjoint=False):
        sol = self.adjoint_solution if adjoint else self.solution
        u, eta = sol.split()
        if self.op.adapt_field in ('fluid_speed', 'both'):
            spd = Function(self.P1).interpolate(sqrt(inner(u, u)))
            self.M = steady_metric(spd, op=self.op)
        elif self.op.adapt_field == 'elevation':
            self.M = steady_metric(eta, op=self.op)
        if self.op.adapt_field == 'both':
            M = steady_metric(eta, op=self.op)
            self.M = metric_intersection(self.M, M)

    def dwr_estimation(self):  # TODO: Different flavours of DWR
        cell_res = self.ts.cell_residual(self.adjoint_solution)
        edge_res = self.ts.edge_residual(self.adjoint_solution)
        self.indicator = Function(self.P0)
        self.indicator.project(cell_res + edge_res)

    def custom_adapt(self):
        if self.approach == 'vorticity':
            self.indicator = Function(self.P1, name='vorticity')
            self.indicator.interpolate(curl(self.solution.split()[0]))
            self.get_isotropic_metric()

    def interpolate_solution(self):
        """
        Interpolate solution onto the new mesh after a mesh adaptation.
        """
        interpolated_solution = Function(FunctionSpace(self.mesh, self.V.ufl_element()))
        uv_i, elev_i = interpolated_solution.split()
        uv, elev = self.solution.split()
        uv_i.project(uv)
        name = uv.dat.name
        uv_i.rename(name)
        elev_i.project(elev)
        name = elev.dat.name
        elev_i.rename(name)
        self.solution = interpolated_solution

    def interpolate_adjoint_solution(self):
        """
        Interpolate adjoint solution onto the new mesh after a mesh adaptation.
        """
        self.interpolated_adjoint_solution = Function(FunctionSpace(self.mesh, self.V.ufl_element()))
        z_i, zeta_i = self.interpolated_adjoint_solution.split()
        z, zeta = self.adjoint_solution
        z_i.project(z)
        zeta_i.project(zeta)
