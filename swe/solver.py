from thetis import *
from thetis.physical_constants import *
from firedrake.petsc import PETSc
import math

from adapt_utils.solver import SteadyProblem, UnsteadyProblem


__all__ = ["SteadyShallowWaterProblem", "UnsteadyShallowWaterProblem"]


class SteadyShallowWaterProblem(SteadyProblem):
    """
    General solver object for stationary shallow water problems.
    """
    # TODO: Documentation
    def __init__(self,
                 mesh=None,
                 discrete_adjoint=True,
                 op=SteadyShallowWaterOptions(),
                 prev_solution=None):
        if op.family == 'dg-dg' and op.degree in (1, 2):
            element = VectorElement("DG", triangle, 1)*FiniteElement("DG", triangle, op.degree)
        elif op.family == 'dg-cg':
            element = VectorElement("DG", triangle, 1)*FiniteElement("Lagrange", triangle, 2)
        else:
            raise NotImplementedError
        if mesh is None:
            mesh = op.default_mesh
        super(SteadyShallowWaterProblem, self).__init__(mesh, op, element, discrete_adjoint, prev_solution)

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
        self.gradient_field = self.op.bathymetry  # For pyadjoint gradient computation
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
        Create a Thetis FlowSolver2d object for solving the shallow water equations and solve.
        """
        op = self.op
        solver_obj = solver2d.FlowSolver2d(self.mesh, op.bathymetry)
        options = solver_obj.options
        options.use_nonlinear_equations = self.nonlinear
        options.check_volume_conservation_2d = True

        # Timestepping
        options.timestep = op.dt
        options.simulation_export_time = op.dt
        options.simulation_end_time = op.end_time
        options.timestepper_type = 'SteadyState'
        options.timestepper_options.solver_parameters = op.params
        PETSc.Sys.Print(options.timestepper_options.solver_parameters)
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

        # Boundary conditions
        solver_obj.bnd_functions['shallow_water'] = op.set_bcs()

        if hasattr(self, 'modified_setup'):
            self.modified_setup(solver_obj)  # TODO for turbine

        # Initial conditions
        if self.prev_solution is not None:
            solver_obj.assign_initial_conditions(uv=self.interpolated_solution)
        else:
            solver_obj.assign_initial_conditions(uv=self.inflow)

        # Solve
        solver_obj.iterate()
        self.solution.assign(solver_obj.fields.solution_2d)

    def get_qoi_kernel(self):
        pass

    def quantity_of_interest(self):
        pass

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

    def get_strong_residual(self):
        i = self.p0test
        u, eta = self.solution.split()
        z, zeta = self.adjoint_solution.split()
        b = self.bathymetry
        nu = self.nu
        f = self.coriolis
        C_d = self.drag_coefficient
        H = b + eta
        g = self.g

        F = -g*inner(grad(eta), z)                  # ExternalPressureGradient
        F += -div(H*u)*zeta                         # HUDiv
        F += -inner(dot(u, nabla_grad(u)), z)       # HorizontalAdvection
        F += -inner(div(nu*grad(u)), z)             # HorizontalViscosity
        F += -inner(f*as_vector((-u[1], u[0])), z)  # Coriolis
        F += C_d*sqrt(dot(u, u))*inner(u, z)/H      # QuadraticDrag
        if hasattr(self, 'extra_residual_terms'):   # TODO for turbine
            F += self.extra_residual_terms()

        # TODO: what about optional viscosity terms?

        self.estimators['dwr_cell'] = assemble(F*dx)
        self.indicators['dwr_cell'] = assemble(i*F*dx)

    def get_flux_terms(self):
        i = self.p0test
        u, eta = self.solution.split()
        z, zeta = self.adjoint_solution.split()
        b = self.bathymetry
        nu = self.nu
        f = self.coriolis
        H = b + eta
        g = self.g
        n = self.n

        # ExternalPressureGradient
        eta_star = avg(eta) + 0.5*sqrt(avg(H)/g)*jump(u, n)
        loc = i*g*dot(z, n)
        flux_terms = eta_star*(loc('+') + loc('-'))*dS
        loc = -i*g*eta*dot(z, n)
        flux_terms += (loc('+') + loc('-'))*dS + loc*ds  # Term arising from IBP

        # HUDiv
        if self.op.family == 'dg-dg':
            u_rie = avg(u) + sqrt(g/avg(H))*jump(eta, n)
            loc = i*n*zeta
            flux_terms += dot(h*u_rie, loc('+') + loc('-'))*dS
        loc = -i*dot(H*u, n)*zeta
        flux_terms += (loc('+') + loc('-'))*dS + loc*ds  # Term arising from IBP

        # HorizontalAdvection
        un_av = dot(avg(u), n('-'))
        u_up = avg(u)
        loc = i*z
        flux_terms += jump(u, n)*dot(u_up, loc('+') + loc('-'))*dS
        loc = -i*inner(outer(u, n), outer(u, z))
        flux_terms += (loc('+') + loc('-'))*dS + loc*ds  # Term arising from IBP
        # TODO: Lax-Friedrichs

        # HorizontalViscosity
        stress = nu*grad(u)
        stress_jump = avg(nu)*tensor_jump(u, n)
        alpha = 1.5 if p == 0 else 5*p*(p+1)
        loc = i*outer(z, n)
        flux_terms += alpha/avg(self.h)*inner(loc('+') + loc('-'), stress_jump)*dS
        flux_terms += -inner(loc('+') + loc('-'), avg(stress))*dS
        loc = i*grad(z)
        flux_terms += -0.5*inner(loc('+') + loc('-'), stress_jump)*dS
        loc = i*inner(stress, outer(z, n))
        flux_terms += (loc('+') + loc('-'))*dS + loc*ds  # Term arising from IBP

        bcs = self.boundary_conditions
        for j in bcs.keys():
            raise NotImplementedError  # TODO!

        if hasattr(self, 'extra_flux_terms'):
            flux_terms += self.extra_flux_terms()

        # TODO: what about optional viscosity terms?

        # Solve auxiliary finite element problem to get traces on particular element
        mass_term = i*self.p0trial*dx
        res = Function(self.P0)
        solve(mass_term == flux_terms, res)
        self.estimators['dwr_flux'] = assemble(res*dx)
        self.indicators['dwr_flux'] = res

    def explicit_estimation(self):
        raise NotImplementedError  # TODO

    def explicit_estimation_adjoint(self):
        raise NotImplementedError  # TODO

    def dwr_estimation(self):  # TODO: Different flavours of DWR
        raise NotImplementedError  # TODO

    def dwr_estimation_adjoint(self):
        raise NotImplementedError  # TODO

    def get_anisotropic_metric(self, adjoint=False):
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

        H1 = [0, 0, 0]
        H2 = [0, 0, 0]
        Hf = [0, 0]

        # Construct Hessians
        for i in range(3):
            H1[i] = steady_metric(F1[i], mesh=self.mesh, noscale=True, op=self.op)
            H2[i] = steady_metric(F2[i], mesh=self.mesh, noscale=True, op=self.op)

        # Form metric
        self.M = Function(self.P1_ten)
        for i in range(len(self.M.dat.data)):
            self.M.dat.data[i][:,:] += H1[0].dat.data[i]*z0_diff.dat.data[i][0]
            self.M.dat.data[i][:,:] += H1[1].dat.data[i]*z1_diff.dat.data[i][0]
            self.M.dat.data[i][:,:] += H1[2].dat.data[i]*zeta_diff.dat.data[i][0]
            self.M.dat.data[i][:,:] += H2[0].dat.data[i]*z0_diff.dat.data[i][1]
            self.M.dat.data[i][:,:] += H2[1].dat.data[i]*z1_diff.dat.data[i][0]
            self.M.dat.data[i][:,:] += H2[2].dat.data[i]*zeta_diff.dat.data[i][0]
        self.M = steady_metric(None, H=self.M, op=self.op)

        # TODO: Account for flux terms contributed by DG scheme
        # TODO: boundary contributions

    def custom_adapt(self):
        if self.approach == 'vorticity':
            self.indicator = Function(self.P1, name='vorticity')
            self.indicator.interpolate(curl(self.solution.split()[0]))
            self.get_isotropic_metric()

    def plot(self):
        """
        Plot current mesh and indicator field, if available.
        """
        File(self.di + 'mesh.pvd').write(self.mesh.coordinates)
        if hasattr(self, 'indicator'):
            name = self.indicator.dat.name
            self.indicator.rename(name + ' indicator')
            File(self.di + 'indicator.pvd').write(self.indicator)
        if hasattr(self, 'adjoint_solution'):
            z, zeta = self.adjoint_solution.split()
            self.adjoint_solution_file.write(z, zeta)

    def interpolate_solution(self):
        """
        Here we only need interpolate the velocity.
        """
        PETSc.Sys.Print("Interpolating solution across meshes...")
        self.interpolated_solution = Function(self.V.sub(0))
        self.interpolated_solution.project(self.prev_solution.split()[0])
        #self.interpolated_solution = interp(self.mesh, self.prev_solution.split()[0])


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
        options.use_nonlinear_equations = self.nonlinear
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
