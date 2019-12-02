from thetis import *
from thetis.physical_constants import *
from firedrake.petsc import PETSc
import math

from adapt_utils.swe.options import ShallowWaterOptions
from adapt_utils.solver import SteadyProblem, UnsteadyProblem
from adapt_utils.adapt.metric import *
from adapt_utils.adapt.p0_metric import *


__all__ = ["SteadyShallowWaterProblem", "UnsteadyShallowWaterProblem"]


class SteadyShallowWaterProblem(SteadyProblem):
    """
    General solver object for stationary shallow water problems.
    """
    def __init__(self,
                 mesh=None,
                 discrete_adjoint=True,
                 op=ShallowWaterOptions(),
                 prev_solution=None):
        if op.family == 'dg-dg' and op.degree in (1, 2):
            element = VectorElement("DG", triangle, op.degree)*FiniteElement("DG", triangle, op.degree)
        elif op.family == 'dg-cg':
            element = VectorElement("DG", triangle, op.degree)*FiniteElement("Lagrange", triangle, op.degree+1)
        else:
            raise NotImplementedError
        if mesh is None:
            mesh = op.default_mesh
        super(SteadyShallowWaterProblem, self).__init__(mesh, op, element, discrete_adjoint, prev_solution)

        self.prev_solution = prev_solution
        if prev_solution is not None:
            self.interpolate_solution()

        # Physical fields
        self.set_fields()
        physical_constants['g_grav'].assign(op.g)

        # BCs
        self.boundary_conditions = op.set_bcs(self.V)

        # Parameters for adjoint computation
        z, zeta = self.adjoint_solution.split()
        z.rename("Adjoint fluid velocity")
        zeta.rename("Adjoint elevation")

        # Classification
        self.nonlinear = True

    def set_fields(self):
        self.op.set_viscosity(self.P1)
        self.inflow = self.op.set_inflow(self.P1_vec)

    def setup_solver(self):
        """
        Create a Thetis FlowSolver2d object for solving the shallow water equations.
        """
        op = self.op
        self.solver_obj = solver2d.FlowSolver2d(self.mesh, op.bathymetry)
        options = self.solver_obj.options
        options.use_nonlinear_equations = self.nonlinear
        options.check_volume_conservation_2d = True

        # Timestepping
        options.timestep = op.dt
        options.simulation_export_time = op.dt
        options.simulation_end_time = op.end_time
        options.timestepper_type = op.timestepper
        options.timestepper_options.solver_parameters = op.params
        PETSc.Sys.Print(options.timestepper_options.solver_parameters)
        # options.timestepper_options.implicitness_theta = 1.0

        # Outputs
        options.output_directory = self.di
        options.fields_to_export = ['uv_2d', 'elev_2d']

        # Parameters
        options.use_grad_div_viscosity_term = op.grad_div_viscosity
        options.element_family = op.family
        options.polynomial_degree = op.degree
        options.horizontal_viscosity = op.viscosity
        options.quadratic_drag_coefficient = op.drag_coefficient
        options.use_lax_friedrichs_velocity = op.lax_friedrichs
        options.lax_friedrichs_velocity_scaling_factor = op.lax_friedrichs_scaling_factor
        options.use_grad_depth_viscosity_term = op.grad_depth_viscosity
        options.use_automatic_sipg_parameter = True
        self.solver_obj.create_equations()
        self.sipg_parameter = options.sipg_parameter

        # Boundary conditions
        self.solver_obj.bnd_functions['shallow_water'] = self.boundary_conditions

        if hasattr(self, 'extra_setup'):
            self.extra_setup()

        # Initial conditions
        if self.prev_solution is not None:
            interp = self.interpolated_solution
            u_interp, eta_interp = self.interpolated_solution.split()
        else:
            interp = Function(self.V)
            u_interp, eta_interp = interp.split()
            u_interp.interpolate(self.inflow)
        self.solver_obj.assign_initial_conditions(uv=u_interp, elev=eta_interp)
        self.lhs = self.solver_obj.timestepper.F
        self.solution = self.solver_obj.fields.solution_2d

    def solve(self):
        if not hasattr(self, 'solver_obj'):
            self.setup_solver()
        self.solver_obj.iterate()
        self.solution = self.solver_obj.fields.solution_2d

        if hasattr(self, 'cb'):
            self.get_callbacks(self.cb)

    def solve_discrete_adjoint(self):
        dFdu = derivative(self.lhs, self.solution, TrialFunction(self.V))
        dFdu_form = adjoint(dFdu)
        dJdu = derivative(self.quantity_of_interest_form(), self.solution, TestFunction(self.V))
        solve(dFdu_form == dJdu, self.adjoint_solution, solver_parameters=self.op.adjoint_params)

    def get_qoi_kernel(self):
        pass

    def quantity_of_interest(self):
        pass

    def get_hessian_metric(self, noscale=False, degree=1, adjoint=False):
        field = self.op.adapt_field
        sol = self.adjoint_solution if adjoint else self.solution
        u, eta = sol.split()

        def elevation():
            return steady_metric(eta, noscale=noscale, degree=degree, op=self.op)

        def velocity_x():
            s = Function(self.P1).interpolate(u[0])
            return steady_metric(s, noscale=noscale, degree=degree, op=self.op)

        def velocity_y():
            s = Function(self.P1).interpolate(u[1])
            return steady_metric(s, noscale=noscale, degree=degree, op=self.op)

        def speed():
            spd = Function(self.P1).interpolate(sqrt(inner(u, u)))
            return steady_metric(spd, noscale=noscale, degree=degree, op=self.op)

        def inflow():
            v = Function(self.P1).interpolate(inner(u, self.op.inflow))
            return steady_metric(v, noscale=noscale, degree=degree, op=self.op)

        def bathymetry():
            b = Function(self.P1).interpolate(self.op.bathymetry)
            return steady_metric(b, noscale=noscale, degree=degree, op=self.op)

        def viscosity():
            nu = Function(self.P1).interpolate(self.op.viscosity)
            return steady_metric(nu, noscale=noscale, degree=degree, op=self.op)

        metrics = {'elevation': elevation, 'velocity_x': velocity_x, 'velocity_y': velocity_y,
                   'speed': speed, 'inflow': inflow,
                   'bathymetry': bathymetry, 'viscosity': viscosity}

        self.M = Function(self.P1_ten)
        if field in metrics:
            self.M = metrics[field]()
        elif field == 'all_avg':
            self.M += metrics['velocity_x']()/3.0
            self.M += metrics['velocity_y']()/3.0
            self.M += metrics['elevation']()/3.0
        elif field == 'all_int':
            self.M = metric_intersection(metrics['velocity_x'](), metrics['velocity_y']())
            self.M = metric_intersection(self.M, metrics['elevation']())
        elif 'avg' in field and 'int' in field:
            raise NotImplementedError  # TODO
        elif 'avg' in field:
            fields = field.split('_avg_')
            num_fields = len(fields)
            for i in range(num_fields):
                self.M += metrics[fields[i]]()/num_fields
        elif 'int' in field:
            fields = field.split('_int_')
            self.M = metrics[fields[0]]()
            for i in range(1, len(fields)):
                self.M = metric_intersection(self.M, metrics[fields[i]]())
        else:
            raise ValueError("Adaptation field {:s} not recognised.".format(field))

    def get_bdy_functions(self, eta_in, u_in, bdy_id):
        b = self.op.bathymetry
        bdy_len = self.mesh.boundary_len[bdy_id]
        funcs = self.boundary_conditions.get(bdy_id)
        if 'elev' in funcs and 'uv' in funcs:
            eta = funcs['elev']
            u = funcs['uv']
        elif 'elev' in funcs and 'un' in funcs:
            eta = funcs['elev']
            u = funcs['un']*self.n
        elif 'elev' in funcs and 'flux' in funcs:
            eta = funcs['elev']
            H = eta + b
            area = H*bdy_len
            u = funcs['flux']/area*self.n
        elif 'elev' in funcs:
            eta = funcs['elev']
            u = u_in
        elif 'uv' in funcs:
            eta = eta_in
            u = funcs['uv']
        elif 'un' in funcs:
            eta = eta_in
            u = funcs['un']*self.n
        elif 'flux' in funcs:
            eta = eta_in
            H = eta + b
            area = H*bdy_len
            u = funcs['flux']/area*self.n
        else:
            raise Exception('Unsupported boundary type {:}'.format(funcs.keys()))
        return eta, u

    def get_strong_residual(self, sol, adjoint_sol, adjoint=False):
        assert not adjoint  # FIXME
        assert sol.function_space() == self.solution.function_space()
        assert adjoint_sol.function_space() == self.adjoint_solution.function_space()
        u, eta = sol.split()
        z, zeta = adjoint_sol.split()

        op = self.op
        i = self.p0test
        b = op.bathymetry
        nu = op.viscosity
        f = None if not hasattr(op, 'coriolis') else op.coriolis
        C_d = None if not hasattr(op, 'drag_coefficient') else op.drag_coefficient
        H = b + eta

        F = -op.g*inner(z, grad(eta))                        # ExternalPressureGradient
        F += -zeta*div(H*u)                                  # HUDiv
        F += -inner(z, dot(u, nabla_grad(u)))                # HorizontalAdvection
        if f is not None:
            F += -inner(z, f*as_vector((-u[1], u[0])))       # Coriolis
        if C_d is not None:
            F += -C_d*sqrt(dot(u, u))*inner(z, u)/H          # QuadraticDrag

        # HorizontalViscosity
        stress = 2*nu*sym(grad(u)) if op.grad_div_viscosity else nu*grad(u)
        F += inner(z, div(stress))
        if op.grad_depth_viscosity:
            F += inner(z, dot(grad(H)/H, stress))

        if hasattr(self, 'extra_residual_terms'):
            F += self.extra_residual_terms(u, eta, z, zeta)

        self.estimators['dwr_cell'] = assemble(F*dx)
        self.indicators['dwr_cell'] = assemble(i*F*dx)

    def get_flux_terms(self, sol, adjoint_sol, adjoint=False):
        assert not adjoint  # FIXME
        assert sol.function_space() == self.solution.function_space()
        assert adjoint_sol.function_space() == self.adjoint_solution.function_space()
        u, eta = sol.split()
        z, zeta = adjoint_sol.split()

        op = self.op
        i = self.p0test
        b = op.bathymetry
        nu = op.viscosity
        H = b + eta
        g = op.g
        n = self.n

        # HorizontalAdvection
        u_up = avg(u)
        loc = -i*z[0]
        flux_terms = jump(u[0], n[0])*dot(u_up[0], loc('+') + loc('-'))*dS
        flux_terms += jump(u[1], n[1])*dot(u_up[0], loc('+') + loc('-'))*dS
        loc = -i*z[1]
        flux_terms += jump(u[0], n[0])*dot(u_up[1], loc('+') + loc('-'))*dS
        flux_terms += jump(u[1], n[1])*dot(u_up[1], loc('+') + loc('-'))*dS
        if op.lax_friedrichs:
            gamma = 0.5*abs(dot(u_up, n('-')))*op.lax_friedrichs_scaling_factor
            loc = -i*z
            flux_terms += gamma*dot(loc('+') + loc('-'), jump(u))*dS

        # NOTE: The following is an influential term for steady turbine...
        # loc = i*inner(outer(u, z), outer(u, n))
        # loc = i*inner(u, z)*inner(u, n)
        # loc = i*inner(z, u*dot(u, n))
        loc = i*inner(dot(outer(u, z), u), n)
        flux_terms += (loc('+') + loc('-'))*dS + loc*ds  # Term arising from IBP

        # ExternalPressureGradient
        if self.op.family != 'dg-cg':
            eta_star = avg(eta) + 0.5*sqrt(avg(H)/g)*jump(u, n)
            loc = -i*g*dot(z, n)
            flux_terms += eta_star*(loc('+') + loc('-'))*dS
            loc = i*g*eta*dot(z, n)
            flux_terms += (loc('+') + loc('-'))*dS + loc*ds  # Term arising from IBP

        # HUDiv
        if self.op.family != 'dg-cg':
            u_rie = avg(u) + sqrt(g/avg(H))*jump(eta, n)
            loc = -i*zeta*n
            flux_terms += dot(avg(H)*u_rie, loc('+') + loc('-'))*dS
        loc = i*zeta*dot(H*u, n)
        flux_terms += (loc('+') + loc('-'))*dS + loc*ds  # Term arising from IBP
        # NOTE: This ^^^ is an influential term for steady turbine

        # HorizontalViscosity
        if op.grad_div_viscosity:
            stress = 2*nu*sym(grad(u))
            stress_jump = 2*avg(nu)*sym(tensor_jump(u, n))
        else:
            stress = nu*grad(u)
            stress_jump = avg(nu)*tensor_jump(u, n)
        alpha = self.sipg_parameter
        assert alpha is not None
        loc = i*outer(z, n)
        flux_terms += -alpha/avg(self.h)*inner(loc('+') + loc('-'), stress_jump)*dS
        flux_terms += inner(loc('+') + loc('-'), avg(stress))*dS
        loc = i*grad(z)
        flux_terms += 0.5*inner(loc('+') + loc('-'), stress_jump)*dS
        # loc = -i*inner(outer(z, n), stress)
        loc = -i*inner(dot(z, stress), n)
        flux_terms += (loc('+') + loc('-'))*dS + loc*ds  # Term arising from IBP

        bcs = self.boundary_conditions
        for j in bcs:
            funcs = bcs.get(j)

            if funcs is not None:
                eta_ext, u_ext = self.get_bdy_functions(eta, u, j)

                # ExternalPressureGradient
                un_jump = inner(u - u_ext, n)
                eta_rie = 0.5*(eta + eta_ext) + sqrt(H/g)*un_jump
                if self.op.family == 'dg-cg':
                    flux_terms += -i*g*eta_rie*dot(z, n)*ds(j)
                else:
                    flux_terms += -i*g*(eta_rie - eta)*dot(z, n)*ds(j)

                # HUDiv
                H_ext = eta_ext + b
                H_av = 0.5*(H + H_ext)
                eta_jump = eta - eta_ext
                un_rie = 0.5*inner(u + u_ext, n) + sqrt(g/H_av)*eta_jump
                un_jump = inner(u - u_ext, n)
                eta_rie = 0.5*(eta + eta_ext) + sqrt(H_av/g)*un_jump
                H_rie = b + eta_rie
                flux_terms += -i*H_rie*un_rie*zeta*ds(j)

                # HorizontalAdvection
                eta_jump = eta - eta_ext
                un_rie = 0.5*inner(u + u_ext, n) + sqrt(g/H)*eta_jump
                flux_terms += -i*dot(u, z)*un_rie*ds(j)

                # HorizontalViscosity
                if 'un' in funcs:
                    delta_u = (dot(u, n) - funcs['un'])*n
                else:
                    if u_ext is u:
                        continue
                    delta_u = u - u_ext
                    if op.grad_div_viscosity:
                        stress_jump = 2*nu*sym(outer(delta_u, n))
                    else:
                        stress_jump = nu*outer(delta_u, n)
                flux_terms += -i*alpha/self.h*inner(outer(z, n), stress_jump)*ds(j)
                flux_terms += i*inner(grad(z), stress_jump)*ds(j)
                flux_terms += i*inner(outer(z, n), stress)*ds(j)

            if self.op.family != 'dg-cg' and (funcs is None or 'symm' in funcs):

                # ExternalPressureGradient
                un_jump = inner(u, n)
                head_rie = eta + sqrt(H/g)*un_jump
                flux_terms += -i*g*head_rie*dot(z, n)*ds(j)

            if funcs is None:

                # HorizontalAdvection
                if self.op.lax_friedrichs:
                    u_ext = u - 2*dot(u, n)*n
                    gamma = 0.5*abs(dot(u_old, n))*op.lax_friedrichs_scaling_factor
                    flux_terms += -i*gamma*dot(z, u - u_ext)*ds(j)


        if hasattr(self, 'extra_flux_terms'):
            flux_terms += self.extra_flux_terms()

        # Solve auxiliary finite element problem to get traces on particular element
        mass_term = i*self.p0trial*dx
        res = Function(self.P0)
        solve(mass_term == flux_terms, res)
        self.estimators['dwr_flux'] = assemble(res*dx)
        self.indicators['dwr_flux'] = res

    def explicit_indication(self):
        raise NotImplementedError  # TODO

    def explicit_indication_adjoint(self):
        raise NotImplementedError  # TODO

    def dwr_indication(self, adjoint=False):
        label = 'dwr'
        if adjoint:
            label += '_adjoint'
        self.get_strong_residual(self.solution, self.adjoint_solution, adjoint=adjoint)
        self.get_flux_terms(self.solution, self.adjoint_solution, adjoint=adjoint)
        self.indicator = Function(self.P1, name=label)
        self.indicator.interpolate(abs(self.indicators['dwr_cell'] + self.indicators['dwr_flux']))
        self.estimators[label] = self.estimators['dwr_cell'] + self.estimators['dwr_flux']
        self.indicators[label] = self.indicator

    def dwr_indication_adjoint(self):
        self.dwr_indication(adjoint=True)

    def get_anisotropic_metric(self, sol, adjoint_sol, adjoint=False):
        assert sol.function_space() == self.solution.function_space()
        assert adjoint_sol.function_space() == self.adjoint_solution.function_space()
        u, eta = sol.split()
        z, zeta = adjoint_sol.split()

        z0_diff = Function(self.P1_vec).interpolate(abs(construct_gradient(z[0], mesh=self.mesh)))
        z1_diff = Function(self.P1_vec).interpolate(abs(construct_gradient(z[1], mesh=self.mesh)))
        zeta_diff = Function(self.P1_vec).interpolate(construct_gradient(zeta))
        z_p1 = Function(self.P1_vec).interpolate(abs(z))

        op = self.op
        b = self.op.bathymetry
        H = eta + b
        g = op.g
        nu = op.viscosity
        C_b = op.drag_coefficient
        normu = sqrt(inner(u, u))
        normu3 = normu**3
        F1 = [0, 0, 0]
        F2 = [0, 0, 0]

        if adjoint:
            raise NotImplementedError  # TODO
        else:
            F1[0] = H*u[0]*u[0] + 0.5*g*eta*eta - nu*H*u[0].dx(0) + C_b*normu3/3.
            F1[1] = H*u[0]*u[1] - nu*H*u[1].dx(0)
            F1[2] = H*u[0]
            F2[0] = H*u[0]*u[1] - nu*H*u[0].dx(1)
            F2[1] = H*u[1]*u[1] + 0.5*g*eta*eta - nu*H*u[1].dx(1) + C_b*normu3/3.
            F2[2] = H*u[1]
            # FIXME: doesn't use non-conservative form

        H1 = [0, 0, 0]
        H2 = [0, 0, 0]

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
        self.interpolated_solution = Function(self.V)
        u_interp, eta_interp = self.interpolated_solution.split()
        u_, eta_ = self.prev_solution.split()
        PETSc.Sys.Print("Interpolating solution across meshes...")
        u_interp.project(u_)
        eta_interp.project(eta_)


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
        options.use_grad_div_viscosity_term = op.grad_div_viscosity
        options.element_family = op.family
        options.horizontal_viscosity = op.viscosity
        options.quadratic_drag_coefficient = op.drag_coefficient
        options.coriolis_frequency = op.set_coriolis(self.P1)
        options.use_lax_friedrichs_velocity = self.stab == 'lax_friedrichs'
        options.use_grad_depth_viscosity_term = op.grad_depth_viscosity
        options.use_automatic_sipg_parameter = True

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
