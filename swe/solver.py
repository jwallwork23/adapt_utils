from thetis import *
from thetis.physical_constants import *

from adapt_utils.solver import SteadyProblem, UnsteadyProblem
from adapt_utils.adapt.metric import *
from adapt_utils.swe.utils import *


__all__ = ["SteadyShallowWaterProblem", "UnsteadyShallowWaterProblem"]


class SteadyShallowWaterProblem(SteadyProblem):
    """
    General solver object for stationary shallow water problems.
    """
    def __init__(self, op, mesh=None, **kwargs):
        p = op.degree
        if op.family == 'dg-dg' and p >= 0:
            fe = VectorElement("DG", triangle, p)*FiniteElement("DG", triangle, p)
        elif op.family == 'dg-cg' and p >= 0:
            fe = VectorElement("DG", triangle, p)*FiniteElement("Lagrange", triangle, p+1)
        else:
            raise NotImplementedError
        super(SteadyShallowWaterProblem, self).__init__(op, mesh, fe, **kwargs)
        prev_solution = kwargs.get('prev_solution')
        if prev_solution is not None:
            self.interpolate_solution(prev_solution)

        # Physical parameters
        physical_constants['g_grav'].assign(op.g)

        # Classification
        self.nonlinear = True

    def set_fields(self, adapted=False):
        self.fields = {}
        self.fields['viscosity'] = self.op.set_viscosity(self.P1)
        self.fields['diffusivity'] = self.op.set_diffusivity(self.P1)
        self.fields['bathymetry'] = self.op.set_bathymetry(self.P1DG)
        self.fields['inflow'] = self.op.set_inflow(self.P1_vec)
        self.fields['coriolis'] = self.op.set_coriolis(self.P1)
        self.fields['quadratic_drag_coefficient'] = self.op.set_quadratic_drag_coefficient(self.P1)
        self.fields['manning_drag_coefficient'] = self.op.set_manning_drag_coefficient(self.P1)

    def create_solutions(self):
        super(SteadyShallowWaterProblem, self).create_solutions()
        u, eta = self.solution.split()
        u.rename("Fluid velocity")
        eta.rename("Elevation")
        z, zeta = self.adjoint_solution.split()
        z.rename("Adjoint fluid velocity")
        zeta.rename("Adjoint elevation")

    def get_tracer(self):
        return self.solver_obj.fields.tracer_2d

    def project_tracer(self, val, adjoint=False):
        self.project(val, out=self.get_tracer())

    def set_stabilisation(self):
        self.stabilisation = self.stabilisation or 'no'
        if self.stabilisation in ('no', 'lax_friedrichs'):
            self.stabilisation_parameter = self.op.stabilisation_parameter
        else:
            raise ValueError("Stabilisation method {:s} for {:s} not recognised".format(self.stabilisation, self.__class__.__name__))

    def setup_solver_forward(self):
        """
        Create a Thetis FlowSolver2d object for solving the shallow water equations.
        """
        op = self.op
        self.solver_obj = solver2d.FlowSolver2d(self.mesh, self.fields['bathymetry'])
        options = self.solver_obj.options
        options.use_nonlinear_equations = self.nonlinear
        options.check_volume_conservation_2d = True

        # Timestepping  # TODO: Put parameters in op.timestepping and use update
        options.timestep = op.dt
        options.simulation_export_time = op.dt
        options.simulation_end_time = op.end_time
        options.timestepper_type = op.timestepper
        if op.params != {}:
            if op.timestepper == 'PressureProjectionPicard':
                options.timestepper_options.solver_parameters_momentum = op.params  # LU by default
            else:
                options.timestepper_options.solver_parameters = op.params
        if op.debug:
            # options.timestepper_options.solver_parameters['snes_monitor'] = None
            print_output(options.timestepper_options.solver_parameters)
        if hasattr(options.timestepper_options, 'implicitness_theta'):
            options.timestepper_options.implicitness_theta = op.implicitness_theta

        # Outputs
        options.output_directory = self.di
        options.fields_to_export = ['uv_2d', 'elev_2d'] if op.plot_pvd else []
        options.fields_to_export_hdf5 = ['uv_2d', 'elev_2d'] if op.save_hdf5 else []

        # Parameters  # TODO: Put parameters in op.shallow_water/tracer and use update
        options.use_grad_div_viscosity_term = op.grad_div_viscosity
        options.element_family = op.family
        options.polynomial_degree = op.degree
        options.horizontal_viscosity = self.fields['viscosity']
        options.horizontal_diffusivity = self.fields['diffusivity']
        options.coriolis_frequency = self.fields['coriolis']
        options.quadratic_drag_coefficient = self.fields['quadratic_drag_coefficient']
        options.manning_drag_coefficient = self.fields['manning_drag_coefficient']
        options.use_lax_friedrichs_velocity = self.stabilisation == 'lax_friedrichs'
        options.lax_friedrichs_velocity_scaling_factor = self.stabilisation_parameter
        options.use_grad_depth_viscosity_term = op.grad_depth_viscosity
        options.use_automatic_sipg_parameter = op.sipg_parameter is None
        options.use_wetting_and_drying = op.wetting_and_drying
        options.wetting_and_drying_alpha = op.wetting_and_drying_alpha
        if op.solve_tracer:
            raise NotImplementedError  # TODO
        options.solve_tracer = op.solve_tracer

        # Boundary conditions
        self.solver_obj.bnd_functions['shallow_water'] = self.boundary_conditions

        # NOTE: Extra setup must be done *before* setting initial condition
        if hasattr(self, 'extra_setup'):
            self.extra_setup()

        # Initial conditions  # TODO: will this work over mesh iterations?
        if self.prev_solution is not None:
            u_interp, eta_interp = self.solution.split()
        else:
            interp = Function(self.V)
            u_interp, eta_interp = interp.split()
            u_interp.interpolate(self.fields['inflow'])
            eta_interp.assign(0.0)
        self.solver_obj.assign_initial_conditions(uv=u_interp, elev=eta_interp)

        self.lhs = self.solver_obj.timestepper.F
        self.solution = self.solver_obj.fields.solution_2d
        self.sipg_parameter = options.sipg_parameter

    def solve_forward(self):
        self.setup_solver_forward()
        self.solver_obj.iterate()
        self.solution.assign(self.solver_obj.fields.solution_2d)

    def get_bnd_functions(self, *args):
        b = self.op.bathymetry if self.op.solve_tracer else self.fields['bathymetry']
        swt = shallowwater_eq.ShallowWaterTerm(self.V, b)
        return swt.get_bnd_functions(*args, self.boundary_conditions)

    def get_strong_residual_forward(self):
        u, eta = self.solution.split()

        b = self.fields['bathymetry']
        nu = self.fields['viscosity']
        f = self.fields['coriolis']
        H = b + eta

        R1 = -self.op.g*grad(eta)              # ExternalPressureGradient
        R2 = -div(H*u)                         # HUDiv
        R1 += -dot(u, nabla_grad(u))           # HorizontalAdvection
        if f is not None:
            R1 += -f*as_vector((-u[1], u[0]))  # Coriolis

        # QuadraticDrag
        if self.fields['quadratic_drag_coefficient'] is not None:
            C_D = self.fields['quadratic_drag_coefficient']
            R1 += -C_D*sqrt(dot(u, u))*u/H
        elif self.fields['manning_drag_coefficient'] is not None:
            C_D = op.g*self.fields['manning_drag_coefficient']**2/pow(H, 1/3)
            R1 += -C_D*sqrt(dot(u, u))*u/H

        # HorizontalViscosity
        stress = 2*nu*sym(grad(u)) if self.op.grad_div_viscosity else nu*grad(u)
        R1 += div(stress)
        if self.op.grad_depth_viscosity:
            R1 += dot(grad(H)/H, stress)

        if hasattr(self, 'extra_strong_residual_terms_momentum'):
            R1 += self.extra_strong_residual_terms_momentum()
        if hasattr(self, 'extra_strong_residual_terms_continuity'):
            R2 += self.extra_strong_residual_terms_continuity()

        name = 'cell_residual_forward'
        if norm_type == 'L2':
            inner_product = assemble(self.p0test*(inner(R1, R1) + inner(R2, R2))*dx)
            self.indicators[name] = project(sqrt(inner_product), self.P0)
        else:
            raise NotImplementedError
        self.estimate_error(name)
        return name

    def get_dwr_residual_forward(self):
        tpe = self.tp_enriched
        tpe.project_solution(self.solution)  # FIXME: prolong
        u, eta = tpe.solution.split()
        z, zeta = self.adjoint_error.split()

        b = tpe.fields['bathymetry']
        nu = tpe.fields['viscosity']
        f = tpe.fields['coriolis']
        H = b + eta

        dwr = -self.op.g*inner(z, grad(eta))                   # ExternalPressureGradient
        dwr += -zeta*div(H*u)                                  # HUDiv
        dwr += -inner(z, dot(u, nabla_grad(u)))                # HorizontalAdvection
        if f is not None:
            dwr += -inner(z, f*as_vector((-u[1], u[0])))       # Coriolis

        # QuadraticDrag
        if tpe.fields['quadratic_drag_coefficient'] is not None:
            C_D = tpe.fields['quadratic_drag_coefficient']
            dwr += -C_D*sqrt(dot(u, u))*inner(z, u)/H
        elif tpe.fields['manning_drag_coefficient'] is not None:
            C_D = op.g*tpe.fields['manning_drag_coefficient']**2/pow(H, 1/3)
            dwr += -C_D*sqrt(dot(u, u))*inner(z, u)/H

        # HorizontalViscosity
        stress = 2*nu*sym(grad(u)) if self.op.grad_div_viscosity else nu*grad(u)
        dwr += inner(z, div(stress))
        if self.op.grad_depth_viscosity:
            dwr += inner(z, dot(grad(H)/H, stress))

        if hasattr(self, 'extra_residual_terms'):
            dwr += tpe.extra_residual_terms()

        name = 'dwr_cell'
        self.indicators[name] = project(assemble(tpe.p0test*dwr*dx), self.P0)
        self.estimate_error(name)
        return name

    def get_dwr_flux_forward(self):
        tpe = self.tp_enriched
        i = tpe.p0test
        tpe.project_solution(self.solution)  # FIXME: prolong
        u, eta = tpe.solution.split()
        z, zeta = self.adjoint_error.split()

        b = tpe.fields['bathymetry']
        nu = tpe.fields['viscosity']
        H = b + eta
        g = self.op.g
        n = tpe.n
        h = tpe.h

        # HorizontalAdvection
        u_up = avg(u)
        loc = -i*z[0]
        flux_terms = jump(u[0], n[0])*dot(u_up[0], loc('+') + loc('-'))*dS
        flux_terms += jump(u[1], n[1])*dot(u_up[0], loc('+') + loc('-'))*dS
        loc = -i*z[1]
        flux_terms += jump(u[0], n[0])*dot(u_up[1], loc('+') + loc('-'))*dS
        flux_terms += jump(u[1], n[1])*dot(u_up[1], loc('+') + loc('-'))*dS
        if self.op.stabilisation == 'lax_friedrichs':
            gamma = 0.5*abs(dot(u_up, n('-')))*self.op.stabilisation_parameter
            loc = -z
            flux_terms += gamma*dot(i('+')*loc('+') - i('-')*loc('-'), jump(u))*dS  # "Local jump"

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
        if self.op.grad_div_viscosity:
            stress = 2*nu*sym(grad(u))
            stress_jump = 2*avg(nu)*sym(tensor_jump(u, n))
        else:
            stress = nu*grad(u)
            stress_jump = avg(nu)*tensor_jump(u, n)
        alpha = tpe.sipg_parameter
        assert alpha is not None
        loc = i*outer(z, n)
        flux_terms += -avg(alpha/h)*inner(loc('+') + loc('-'), stress_jump)*dS
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
                eta_ext, u_ext = tpe.get_bnd_functions(eta, u, j)

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
                flux_terms += -i*dot(0.5*(u + u_ext), z)*un_rie*ds(j)

                # HorizontalViscosity
                if 'un' in funcs:
                    delta_u = (dot(u, n) - funcs['un'])*n
                else:
                    if u_ext is u:
                        continue
                    delta_u = u - u_ext
                    if self.op.grad_div_viscosity:
                        stress_jump = 2*nu*sym(outer(delta_u, n))
                    else:
                        stress_jump = nu*outer(delta_u, n)
                flux_terms += -i*alpha/h*inner(outer(z, n), stress_jump)*ds(j)
                flux_terms += i*inner(grad(z), stress_jump)*ds(j)
                flux_terms += i*inner(outer(z, n), stress)*ds(j)

            if self.op.family != 'dg-cg' and (funcs is None or 'symm' in funcs):

                # ExternalPressureGradient
                un_jump = inner(u, n)
                head_rie = eta + sqrt(H/g)*un_jump
                flux_terms += -i*g*head_rie*dot(z, n)*ds(j)

            if funcs is None:

                # HorizontalAdvection
                if self.op.stabilisation == 'lax_friedrichs':
                    u_ext = u - 2*dot(u, n)*n
                    gamma = 0.5*abs(dot(u_old, n))*self.op.stabilisation_parameter
                    flux_terms += -i*gamma*dot(z, u - u_ext)*ds(j)

        if hasattr(self, 'extra_flux_terms'):
            flux_terms += tpe.extra_flux_terms()

        # Solve auxiliary finite element problem to get traces on particular element
        name = 'dwr_flux'
        mass_term = i*tpe.p0trial*dx
        res = Function(tpe.P0)
        solve(mass_term == flux_terms, res)
        self.indicators[name] = project(assemble(i*res*dx), self.P0)
        self.estimate_error(name)
        return name

    def custom_adapt(self):
        if self.approach == 'vorticity':
            self.indicator = Function(self.P1, name='vorticity')
            self.indicator.interpolate(vorticity(self.solution))
            self.get_isotropic_metric()

    def plot_solution(self, adjoint=False):
        if adjoint:
            z, zeta = self.adjoint_solution.split()
            z.rename("Adjoint fluid velocity")
            zeta.rename("Adjoint elevation")
            self.adjoint_solution_file.write(z, zeta)
        else:
            u, eta = self.solution.split()
            u.rename("Fluid velocity")
            eta.rename("Elevation")
            self.solution_file.write(u, eta)

    def get_hessian_metric(self, adjoint=False, **kwargs):
        kwargs.setdefault('noscale', False)
        kwargs['op'] = self.op
        sol = self.adjoint_solution if adjoint else self.solution
        self.M = get_hessian_metric(sol, self.op.adapt_field, fields=self.fields, **kwargs)


class UnsteadyShallowWaterProblem(UnsteadyProblem):
    """
    General solver object for time-dependent shallow water problems.
    """
    def __init__(self, op, mesh=None, **kwargs):
        p = op.degree
        if op.family == 'dg-dg' and p >= 0:
            fe = VectorElement("DG", triangle, p)*FiniteElement("DG", triangle, p)
        elif op.family == 'dg-cg' and p >= 0:
            fe = VectorElement("DG", triangle, p)*FiniteElement("Lagrange", triangle, p+1)
        else:
            raise NotImplementedError
        super(UnsteadyShallowWaterProblem, self).__init__(op, mesh, fe, **kwargs)
        prev_solution = kwargs.get('prev_solution')
        if prev_solution is not None:
            self.interpolate_solution(prev_solution)

        # Physical fields
        physical_constants['g_grav'].assign(op.g)

        # Classification
        self.nonlinear = True

    def set_fields(self):
        self.fields = {}
        self.fields['viscosity'] = self.op.set_viscosity(self.P1)
        self.fields['diffusivity'] = self.op.set_diffusivity(self.P1)
        if not self.op.solve_tracer:
            self.fields['bathymetry'] = self.op.set_bathymetry(self.P1DG)
        self.fields['inflow'] = self.op.set_inflow(self.P1_vec)
        self.fields['coriolis'] = self.op.set_coriolis(self.P1)
        self.fields['quadratic_drag_coefficient'] = self.op.set_quadratic_drag_coefficient(self.P1)
        self.fields['manning_drag_coefficient'] = self.op.set_manning_drag_coefficient(self.P1)
        self.fields['source'] = self.op.source

        self.op.set_boundary_surface()

    def create_solutions(self):
        super(UnsteadyShallowWaterProblem, self).create_solutions()
        u, eta = self.solution.split()
        u.rename("Fluid velocity")
        eta.rename("Elevation")
        z, zeta = self.adjoint_solution.split()
        z.rename("Adjoint fluid velocity")
        zeta.rename("Adjoint elevation")

    def get_tracer(self):
        return self.solver_obj.fields.tracer_2d

    def project_tracer(self, val, adjoint=False):
        self.project(val, out=self.get_tracer())

    def set_stabilisation(self):
        self.stabilisation = self.stabilisation or 'no'
        if self.stabilisation in ('no', 'lax_friedrichs'):
            self.stabilisation_parameter = self.op.stabilisation_parameter
        else:
            raise ValueError("Stabilisation method {:s} for {:s} not recognised".format(self.stabilisation, self.__class__.__name__))

    def solve_forward(self, adjoint=False):

        # Create solver object
        if not hasattr(self, 'solver_obj'):
            self.setup_solver_forward()

        # Time integrate
        self.solver_obj.iterate(
            update_forcings=self.op.get_update_forcings(self.solver_obj),
            export_func=self.op.get_export_func(self.solver_obj)
        )

        # Extract solutions
        self.solution = self.solver_obj.fields.solution_2d
        if self.op.solve_tracer:
            self.solution_tracer = self.solver_obj.fields.tracer_2d
        self.solution_old_bathymetry = self.solver_obj.fields.bathymetry_2d

    def setup_solver_forward(self):
        op = self.op

        # Use appropriate bathymetry
        b = self.op.set_bathymetry(self.P1) if self.op.solve_tracer else self.fields['bathymetry']

        # Create solver object
        self.solver_obj = solver2d.FlowSolver2d(self.mesh, b)
        options = self.solver_obj.options
        options.use_nonlinear_equations = self.nonlinear
        options.check_volume_conservation_2d = True

        # Timestepping
        options.timestep = op.dt
        options.simulation_export_time = op.dt*op.dt_per_export
        options.simulation_end_time = self.step_end - 0.5*op.dt
        options.timestepper_type = op.timestepper
        if op.params != {}:
            options.timestepper_options.solver_parameters = op.params
        op.print_debug(options.timestepper_options.solver_parameters)
        if hasattr(options.timestepper_options, 'implicitness_theta'):
            options.timestepper_options.implicitness_theta = op.implicitness_theta

        # Outputs
        options.output_directory = self.di
        prognostic_fields = ['uv_2d', 'elev_2d']
        if op.solve_tracer:
            prognostic_fields.append('tracer_2d')
        options.fields_to_export = prognostic_fields if op.plot_pvd else []
        options.fields_to_export_hdf5 = prognostic_fields if op.save_hdf5 else []

        # Parameters  # TODO: Put parameters in op.shallow_water/tracer and use update
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
        if op.solve_tracer:
            options.use_tracer_conservative_form = op.conservative
            options.tracer_advective_velocity_factor = self.op.corrective_velocity_factor
            if op.depth_integrated and op.conservative:
                options.tracer_depth_integ_source = self.op.depth_int_source
                options.tracer_depth_integ_sink = self.op.depth_int_sink
            else:
                options.tracer_source_2d = self.op.source
                options.tracer_sink_2d = self.op.sink

        # Boundary conditions
        self.solver_obj.bnd_functions['shallow_water'] = op.set_boundary_conditions(self.V)
        if op.solve_tracer:
            self.solver_obj.bnd_functions['tracer'] = op.set_boundary_conditions_tracer(self.V)

        # Initial conditions
        uv, eta = self.solution.split()
        if op.solve_tracer:
            self.uv_d, self.eta_d = self.solution.split()
            op.set_up_suspended(self.mesh, tracer=self.op.tracer_init)
        tracer_init = None
        if op.solve_tracer and self.op.tracer_init is not None:
            tracer_init = self.op.tracer_init
        self.solver_obj.assign_initial_conditions(uv=uv, elev=eta, tracer=tracer_init)

        if hasattr(self, 'extra_setup'):
            self.extra_setup()

    def get_bnd_functions(self, *args):
        b = self.op.bathymetry if self.op.solve_tracer else self.fields['bathymetry']
        swt = shallowwater_eq.ShallowWaterTerm(self.V, b)
        return swt.get_bnd_functions(*args, self.boundary_conditions)

    def get_qoi_kernel(self):
        self.kernel = self.op.set_qoi_kernel(self.solver_obj)

    def plot_solution(self, adjoint=False):
        if adjoint:
            z, zeta = self.adjoint_solution.split()
            z.rename("Adjoint fluid velocity")
            zeta.rename("Adjoint elevation")
            self.adjoint_solution_file.write(z, zeta)
        else:
            u, eta = self.solution.split()
            u.rename("Fluid velocity")
            eta.rename("Elevation")
            self.solution_file.write(u, eta)
