from thetis_adjoint import *
import pyadjoint
import math

from adapt_utils.solver import *
from adapt_utils.turbine.options import *
from adapt_utils.adapt.recovery import *
from adapt_utils.adapt.metric import *
from adapt_utils.adapt.interpolation import *


__all__ = ["SteadyTurbineProblem", "UnsteadyTurbineProblem"]


class SteadyTurbineProblem(SteadyProblem):
    """
    General solver object for stationary tidal turbine problems.
    """
    # TODO: Documentation
    def __init__(self,
                 mesh=None,
                 discrete_adjoint=True,
                 op=Steady2TurbineOptions(),
                 prev_solution=None):
        if op.family == 'dg-dg' and op.degree in (1, 2):
            element = VectorElement("DG", triangle, 1)*FiniteElement("DG", triangle, op.degree)
        elif op.family == 'dg-cg':
            element = VectorElement("DG", triangle, 1)*FiniteElement("Lagrange", triangle, 2)
        else:
            raise NotImplementedError
        if mesh is None:
            mesh = op.default_mesh
        super(SteadyTurbineProblem, self).__init__(mesh, op, element, discrete_adjoint, prev_solution)

        # Stabilisation
        if self.stab is not None:
            try:
                assert self.stab == 'lax_friedrichs'
            except:
                raise NotImplementedError
        self.prev_solution = prev_solution
        if prev_solution is not None:
            self.interpolate_solution()

        # Physical fields
        self.set_fields()

        # Parameters for adjoint computation
        self.gradient_field = self.op.bathymetry
        z, zeta = self.adjoint_solution.split()
        z.rename("Adjoint fluid velocity")
        zeta.rename("Adjoint elevation")

        # Classification
        self.nonlinear = True

    def set_fields(self):
        self.viscosity = self.op.set_viscosity()
        self.inflow = self.op.set_inflow(self.P1_vec)
        self.drag_coefficient = Constant(self.op.drag_coefficient)

    def solve(self):
        """
        Create a Thetis FlowSolver2d object for solving the (modified) shallow water equations and solve.
        """
        op = self.op
        solver_obj = solver2d.FlowSolver2d(self.mesh, op.bathymetry)
        options = solver_obj.options
        options.use_nonlinear_equations = True
        options.check_volume_conservation_2d = True

        # Timestepping
        options.timestep = op.dt
        options.simulation_export_time = op.dt
        options.simulation_end_time = op.end_time
        options.timestepper_type = 'SteadyState'
        options.timestepper_options.solver_parameters = op.params
        print(options.timestepper_options.solver_parameters)
        # options.timestepper_options.implicitness_theta = 1.0

        # Outputs
        options.output_directory = self.di
        options.fields_to_export = ['uv_2d', 'elev_2d']

        # Parameters
        options.use_grad_div_viscosity_term = op.symmetric_viscosity
        options.element_family = op.family
        options.horizontal_viscosity = self.viscosity
        options.quadratic_drag_coefficient = self.drag_coefficient
        options.use_lax_friedrichs_velocity = self.stab == 'lax_friedrichs'
        #options.use_grad_depth_viscosity_term = False
        options.use_grad_depth_viscosity_term = True
        options.compute_residuals = True

        # Boundary conditions
        op.set_bcs()
        solver_obj.bnd_functions['shallow_water'] = op.boundary_conditions

        # We haven't meshed the turbines with separate ids, so define a farm everywhere
        # and make it have a density of 1/D^2 inside the DxD squares where the turbines are
        # and 0 outside
        num_turbines = len(op.region_of_interest)
        scaling = num_turbines/assemble(op.bump(self.P1)*dx)
        self.turbine_density = op.bump(self.P1, scale=scaling)
        #File(self.di+'Bump.pvd').write(turbine_density)
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
        if self.prev_solution is not None:
            solver_obj.assign_initial_conditions(uv=self.interpolated_solution)
        else:
            solver_obj.assign_initial_conditions(uv=self.inflow)

        # Solve
        solver_obj.iterate()
        self.solution.assign(solver_obj.fields.solution_2d)
        self.objective = cb.average_power
        self.ts = solver_obj.timestepper

    def get_objective_kernel(self):
        self.kernel = Function(self.V)
        u = self.solution.split()[0]
        k_u = self.kernel.split()[0]
        k_u.interpolate(Constant(1/3)*self.turbine_density*sqrt(inner(u, u))*u)

    def objective_functional(self):
        return self.objective

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

    def explicit_estimation(self):
        with pyadjoint.stop_annotating():
            cell_res = self.ts.cell_residual()
            self.cell_residuals = [Function(self.P1), Function(self.P1), Function(self.P1)]
            self.cell_residuals[0].project(abs(cell_res[0]))
            self.cell_residuals[1].project(abs(cell_res[1]))
            self.cell_residuals[2].project(abs(cell_res[2]))
            if self.approach == 'explicit':
                self.indicator = Function(self.P1)
                cell_res_dot = self.cell_residuals[0]*self.cell_residuals[0]
                cell_res_dot += self.cell_residuals[1]*self.cell_residuals[1]
                cell_res_dot += self.cell_residuals[2]*self.cell_residuals[2]
                self.indicator.interpolate(cell_res_dot)
            #edge_res = self.ts.edge_residual()  # TODO
            #self.edge_residuals = [Function(self.P1), Function(self.P1), Function(self.P1)]
            #self.edge_residuals[0].project(abs(cell_res[0]))
            #self.edge_residuals[1].project(abs(cell_res[1]))
            #self.edge_residuals[2].project(abs(cell_res[2]))
            #if self.approach == 'explicit':
            #    self.indicator = Function(self.P1)
            #    edge_res_dot = self.edge_residuals[0]*self.edge_residuals[0]
            #    edge_res_dot += self.edge_residuals[1]*self.edge_residuals[1]
            #    edge_res_dot += self.edge_residuals[2]*self.edge_residuals[2]
            #self.indicator.project(cell_res + edge_res)

    def explicit_estimation_adjoint(self):
        raise NotImplementedError  # TODO

    def dwr_estimation(self):  # TODO: Different flavours of DWR
        with pyadjoint.stop_annotating():
            cell_res = self.ts.cell_residual(self.adjoint_solution)
            edge_res = self.ts.edge_residual(self.adjoint_solution)
            self.indicator = Function(self.P0)
            if self.op.dwr_approach == 'error_representation':
                self.indicator.project(cell_res + edge_res)
            elif self.op.dwr_approach == 'AO97':
                self.indicator.project(self.h*cell_res + 0.5*self.h*self.h*edge_res)
            else:
                raise NotImplementedError  # TODO

    def dwr_estimation_adjoint(self):
        raise NotImplementedError  # TODO

    def get_anisotropic_metric(self, adjoint=False, relax=True, superpose=False):
        assert not (relax and superpose)

        u, eta = self.solution.split()
        z, zeta = self.adjoint_solution.split()
        z0_diff = Function(self.P1_vec).interpolate(abs(construct_gradient(z[0], mesh=self.mesh)))
        z1_diff = Function(self.P1_vec).interpolate(abs(construct_gradient(z[1], mesh=self.mesh)))
        zeta_diff = Function(self.P1_vec).interpolate(construct_gradient(zeta))
        z_p1 = Function(self.P1_vec).interpolate(abs(z))
        b = self.op.bathymetry
        H = eta + b
        g = 9.81
        nu = self.viscosity
        C_b = self.drag_coefficient
        C_t = self.turbine_density
        normu = sqrt(inner(u, u))
        normu3 = normu**3
        F1 = [0, 0, 0]
        F2 = [0, 0, 0]
        f = [0, 0]
        if adjoint:
            raise NotImplementedError  # TODO
        else:
            F1[0] = H*u[0]*u[0] + 0.5*g*eta*eta - nu*H*u[0].dx(0) + C_b*normu3/3.
            F1[1] = H*u[0]*u[1] - nu*H*u[1].dx(0)
            F1[2] = H*u[0]
            F2[0] = H*u[0]*u[1] - nu*H*u[0].dx(1)
            F2[1] = H*u[1]*u[1] + 0.5*g*eta*eta - nu*H*u[1].dx(1) + C_b*normu3/3.
            F2[2] = H*u[1]
            # f[0] = g*eta*b.dx(0) - C_t*normu*u[0]
            # f[1] = g*eta*b.dx(1) - C_t*normu*u[1]
            f[0] = -C_t*normu*u[0]
            f[1] = -C_t*normu*u[1]

        H1 = [0, 0, 0]
        H2 = [0, 0, 0]
        Hf = [0, 0]

        # Construct Hessians
        for i in range(3):
            H1[i] = construct_hessian(F1[i], mesh=self.mesh, op=self.op)
            H2[i] = construct_hessian(F2[i], mesh=self.mesh, op=self.op)
        Hf[0] = construct_hessian(f[0], mesh=self.mesh, op=self.op)
        Hf[1] = construct_hessian(f[1], mesh=self.mesh, op=self.op)

        # Form metric
        self.M = Function(self.P1_ten)
        for i in range(len(self.M.dat.data)):
            self.M.dat.data[i][:,:] += H1[0].dat.data[i]*z0_diff.dat.data[i][0]
            self.M.dat.data[i][:,:] += H1[1].dat.data[i]*z1_diff.dat.data[i][0]
            self.M.dat.data[i][:,:] += H1[2].dat.data[i]*zeta_diff.dat.data[i][0]
            self.M.dat.data[i][:,:] += H2[0].dat.data[i]*z0_diff.dat.data[i][1]
            self.M.dat.data[i][:,:] += H2[1].dat.data[i]*z1_diff.dat.data[i][0]
            self.M.dat.data[i][:,:] += H2[2].dat.data[i]*zeta_diff.dat.data[i][0]
            if relax:
                self.M.dat.data[i][:,:] += Hf[0].dat.data[i]*z_p1.dat.data[i][0]
                self.M.dat.data[i][:,:] += Hf[1].dat.data[i]*z_p1.dat.data[i][1]
        self.M = steady_metric(None, H=self.M, op=self.op)

        # Source term contributions
        if superpose:
            Mf = Function(self.P1_ten)
            for i in range(len(Mf.dat.data)):
                Mf.dat.data[i][:,:] += Hf[0].dat.data[i]*z_p1.dat.data[i][0]
                Mf.dat.data[i][:,:] += Hf[1].dat.data[i]*z_p1.dat.data[i][1]
            Mf = steady_metric(None, H=Mf, op=self.op)
            self.M = metric_intersection(self.M, Mf)

        # TODO: Account for flux terms contributed by DG scheme

        # TODO: boundary contributions

    def custom_adapt(self):
        if self.approach == 'vorticity':
            self.indicator = Function(self.P1, name='vorticity')
            self.indicator.interpolate(curl(self.solution.split()[0]))
            self.get_isotropic_metric()
        elif self.approach == 'Power':
            self.explicit_estimation()
            z, zeta = self.adjoint_solution.split()
            #spd = sqrt(inner(z,z))
            #H1 = construct_hessian(spd, mesh=self.mesh, op=self.op)
            H1 = construct_hessian(z[0], mesh=self.mesh, op=self.op)  # TODO: should take abs
            H2 = construct_hessian(z[1], mesh=self.mesh, op=self.op)
            H3 = construct_hessian(zeta, mesh=self.mesh, op=self.op)
            self.M = Function(self.P1_ten)
            for i in range(self.mesh.num_vertices()):  # TODO: use pyop2
                #self.M.dat.data[i][:, :] += self.indicator.dat.data[i]*H1.dat.data[i]
                self.M.dat.data[i][:, :] += self.cell_residuals[0].dat.data[i]*H1.dat.data[i]
                self.M.dat.data[i][:, :] += self.cell_residuals[1].dat.data[i]*H2.dat.data[i]
                self.M.dat.data[i][:, :] += self.cell_residuals[2].dat.data[i]*H3.dat.data[i]
            # TODO: What about edge residuals?
            self.M = steady_metric(None, H=self.M, mesh=self.mesh, op=self.op)
            File('outputs/test.pvd').write(self.M)

    def interpolate_solution(self):
        """
        Here we only need interpolate the velocity.
        """
        print("Interpolating solution across meshes...")
        #self.interpolated_solution = Function(self.V.sub(0))
        #self.interpolated_solution.project(self.prev_solution.split()[0])
        self.interpolated_solution = interp(self.mesh, self.prev_solution.split()[0])


class UnsteadyTurbineProblem(UnsteadyProblem):
    # TODO: doc
    def __init__(self,
                 op=Unsteady2TurbineOptions(),
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
        self.viscosity = self.op.set_viscosity()
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
        print(options.timestepper_options.solver_parameters)
        #options.timestepper_options.implicitness_theta = 1.0

        # Outputs
        options.output_directory = self.di
        options.fields_to_export = ['uv_2d', 'elev_2d']

        # Parameters
        options.use_grad_div_viscosity_term = op.symmetric_viscosity
        options.element_family = op.family
        options.horizontal_viscosity = op.viscosity
        options.quadratic_drag_coefficient = self.drag_coefficient
        options.use_lax_friedrichs_velocity = self.stab == 'lax_friedrichs'
        #options.use_grad_depth_viscosity_term = False
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
            #File(self.di+'Bump.pvd').write(turbine_density)
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
            self.objective = cb.average_power
        self.ts = solver_obj.timestepper

    def objective_functional(self):
        return self.objective

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

    def explicit_estimation(self):
        with pyadjoint.stop_annotating():
            cell_res = self.ts.cell_residual()
            #edge_res = self.ts.edge_residual()
            self.residuals = [Function(self.P1), Function(self.P1), Function(self.P1)]
            self.residuals[0].project(abs(cell_res[0]))
            self.residuals[1].project(abs(cell_res[1]))
            self.residuals[2].project(abs(cell_res[2]))
            if self.approach == 'explicit':
                self.indicator = Function(self.P1)
                res_dot = self.residuals[0]*self.residuals[0]
                res_dot += self.residuals[1]*self.residuals[1]
                res_dot += self.residuals[2]*self.residuals[2]
                self.indicator.interpolate(res_dot)
            #self.indicator.project(cell_res + edge_res)

    def explicit_estimation_adjoint(self):
        raise NotImplementedError  # TODO

    def dwr_estimation(self):  # TODO: Different flavours of DWR
        with pyadjoint.stop_annotating():
            cell_res = self.ts.cell_residual(self.adjoint_solution)
            edge_res = self.ts.edge_residual(self.adjoint_solution)
            self.indicator = Function(self.P0)
            self.indicator.project(cell_res + edge_res)

    def dwr_estimation_adjoint(self):
        raise NotImplementedError  # TODO

    def get_anisotropic_metric(self, adjoint=False, relax=True, superpose=False):
        raise NotImplementedError  # TODO

    def custom_adapt(self):
        if self.approach == 'vorticity':
            self.indicator = Function(self.P1, name='vorticity')
            self.indicator.interpolate(curl(self.solution.split()[0]))
            self.get_isotropic_metric()
        elif self.approach == 'Power':
            self.explicit_estimation()
            self.indicator = Function(self.P0)
            self.indicator.interpolate(sqrt(self.residuals[0]*self.residuals[0]+self.residuals[1]*self.residuals[1]))
            z, zeta = self.adjoint_solution.split()
            #spd = sqrt(inner(z,z))
            #H1 = construct_hessian(spd, mesh=self.mesh, op=self.op)
            H1 = construct_hessian(z[0], mesh=self.mesh, op=self.op)  # TODO: should take abs
            H2 = construct_hessian(z[1], mesh=self.mesh, op=self.op)
            H3 = construct_hessian(zeta, mesh=self.mesh, op=self.op)
            self.M = Function(self.P1_ten)
            for i in range(self.mesh.num_vertices()):
                #self.M.dat.data[i][:, :] += self.indicator.dat.data[i]*H1.dat.data[i]
                self.M.dat.data[i][:, :] += self.residuals[0].dat.data[i]*H1.dat.data[i]
                self.M.dat.data[i][:, :] += self.residuals[1].dat.data[i]*H2.dat.data[i]
                self.M.dat.data[i][:, :] += self.residuals[2].dat.data[i]*H3.dat.data[i]
            self.M = steady_metric(None, H=self.M, mesh=self.mesh, op=self.op)

    def interpolate_solution(self):
        """
        Interpolate solution onto the new mesh after a mesh adaptation.
        """
        with pyadjoint.stop_annotating():
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
        with pyadjoint.stop_annotating():
            self.interpolated_adjoint_solution = Function(FunctionSpace(self.mesh, self.V.ufl_element()))
            z_i, zeta_i = self.interpolated_adjoint_solution.split()
            z, zeta = self.adjoint_solution
            z_i.project(z)
            zeta_i.project(zeta)
