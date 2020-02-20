from thetis import *

from adapt_utils.tracer.options import *
from adapt_utils.tracer.solver2d import *
from adapt_utils.adapt.metric import *
from adapt_utils.misc import index_string


__all__ = ["SteadyTracerProblem2d_Thetis", "UnsteadyTracerProblem2d_Thetis"]


# TODO: Revise this
class SteadyTracerProblem2d_Thetis(SteadyTracerProblem2d):
    r"""
    General discontinuous Galerkin solver object for stationary tracer advection problems of the form

..  math::
    \textbf{u} \cdot \nabla(\phi) - \nabla \cdot (\nu \cdot \nabla(\phi)) = f,

    for (prescribed) velocity :math:`\textbf{u}`, diffusivity :math:`\nu \geq 0`, source :math:`f`
    and (prognostic) concentration :math:`\phi`.
    """
    def __init__(self, op, mesh=None, 
        try:
            assert op.family in ("Discontinuous Lagrange", "DG", "dg")
        except AssertionError:
            raise ValueError("Finite element '{:s}' not supported in Thetis tracer model.".format(op.family))
        super(SteadyTracerProblem2d_Thetis, self).__init__(op, mesh, **kwargs)

    def solve(self):
        solver_obj = solver2d.FlowSolver2d(self.mesh, Constant(1.))
        options = solver_obj.options
        options.timestepper_type = 'SteadyState'
        options.timestep = 20.
        options.simulation_end_time = 0.9*options.timestep
        if self.op.plot_pvd:
            options.fields_to_export = ['tracer_2d']
        else:
            options.no_exports = True
        options.solve_tracer = True
        options.lax_friedrichs_tracer = self.stabilisation == 'lax_friedrichs'
        options.tracer_only = True
        options.horizontal_diffusivity = self.nu
        options.tracer_source_2d = self.source
        solver_obj.assign_initial_conditions(elev=Function(self.P1), uv=self.u)

        # Set SIPG parameter
        options.use_automatic_sipg_parameter = True
        solver_obj.create_equations()
        self.sipg_parameter = options.sipg_parameter

        # Assign BCs and solve
        solver_obj.bnd_functions['tracer'] = self.op.boundary_conditions
        solver_obj.iterate()
        self.solution.assign(solver_obj.fields.tracer_2d)
        self.lhs = self.solver_obj.timestepper.F

    def solve_discrete_adjoint(self):
        dFdu = derivative(self.lhs, self.solution, TrialFunction(self.V))
        dFdu_form = adjoint(dFdu)
        dJdu = derivative(self.quantity_of_interest_form(), self.solution, TestFunction(self.V))
        solve(dFdu_form == dJdu, self.adjoint_solution, solver_parameters=self.op.adjoint_params)
        self.plot()

    def solve_continuous_adjoint(self):
        raise NotImplementedError  # TODO

    def get_dwr_residual(self, sol, adjoint_sol, adjoint=False, weighted=True):
        if adjoint:
            F = self.kernel + div(self.u*adjoint_sol) + div(self.nu*grad(adjoint_sol))
            dwr = inner(F, sol)
        else:
            F = self.source - dot(self.u, grad(sol)) + div(self.nu*grad(sol))
            dwr = inner(F, adjoint_sol)

        if weighted:
            self.indicators['dwr_cell'] = assemble(self.p0test*dwr*dx)
        else:
            self.indicators['strong_residual'] = assemble(self.p0test*F*dx)

    def get_dwr_flux(self, sol, adjoint_sol, adjoint=False):
        try:
            assert self.divergence_free
        except AssertionError:
            raise NotImplementedError  # TODO
        uv = -self.u
        nu = self.nu
        n = self.n
        i = self.p0test
        h = self.h
        degree = self.finite_element.degree()
        family = self.finite_element.family()

        flux_terms = 0

        # Term resulting from integration by parts in advection term
        loc = i*dot(uv, n)*sol*adjoint_sol
        flux_integrand = (loc('+') + loc('-'))

        # Term resulting from integration by parts in diffusion term
        loc = -i*dot(nu*grad(sol), n)*adjoint_sol
        flux_integrand += (loc('+') + loc('-'))
        bdy_integrand = loc

        uv_av = avg(uv)
        un_av = dot(uv_av, n('-'))
        s = 0.5*(sign(un_av) + 1.0)
        phi_up = phi('-')*s + phi('+')*(1-s)

        # Interface term  # NOTE: there are some cancellations with IBP above
        loc = i*dot(uv, n)*adjoint_sol
        flux_integrand += -phi_up*(loc('+') + loc('-'))

        # TODO: Lax-Friedrichs

        # SIPG term
        alpha = avg(self.sipg_parameter/h)
        loc = i*n*adjoint_sol
        flux_integrand += -alpha*inner(avg(nu)*jump(sol, n), loc('+') + loc('-'))
        flux_integrand += inner(jump(nu*grad(sol)), loc('+') + loc('-'))
        loc = i*nu*grad(adjoint_sol)
        flux_integrand += 0.5*inner(loc('+') + loc('-'), jump(sol, n))

        flux_terms += flux_integrand*dS

        # Boundary conditions
        bcs = self.op.boundary_conditions
        for j in bcs:
            bdy_j_integrand = bdy_integrand
            if 'value' in bcs[j]:
                sol_ext = bcs[j]['value']
                uv_av = 0.5*(uv + sol_ext)
                un_av = dot(n, uv_av)
                s = 0.5*(sign(un_av) + 1.0)
                sol_up = (sol - sol_ext)*(1 - s)
                bdy_j_integrand += -i*sol_up*dot(uv_av, n)*adjoint_sol
                bdy_j_integrand += i*dot(uv, n)*phi*adjoint_sol
            flux_terms += bdy_j_integrand*ds(j)

        # Solve auxiliary finite element problem to get traces on particular element
        mass_term = i*self.p0trial*dx
        res = Function(self.P0)
        solve(mass_term == flux_terms, res)
        self.estimators['dwr_flux'] = abs(assemble(res*dx))
        self.indicators['dwr_flux'] = res


# TODO: revise this
class UnsteadyTracerProblem2d_Thetis(UnsteadyTracerProblem2d):
    def __init__(self, op, mesh=None, **kwargs):
        super(UnsteadyTracerProblem2d_Thetis, self).__init__(op, mesh, **kwargs)

        self.set_fields()

        # Classification
        self.nonlinear = False

    def get_qoi_kernel(self):
        self.kernel = self.op.set_qoi_kernel(self.P0)

    def set_fields(self):
        self.nu = self.op.set_diffusivity(self.V)
        self.u = self.op.set_velocity(self.P1_vec)
        self.source = self.op.set_source(self.V)
        self.kernel = self.op.set_qoi_kernel(self.P0)
        self.gradient_field = self.nu  # arbitrary field to take gradient for discrete adjoint

    def solve_step(self, adjoint=False, time=None):
        self.set_fields()
        if adjoint and not self.discrete_adjoint:  # TODO: discrete adjoint solver should be easy
            try:
                assert self.divergence_free
            except AssertionError:
                raise ValueError("Automated continuous adjoint not valid as velocity field is not divergence free")
        op = self.op

        # Create solver and pass parameters, etc.
        solver_obj = solver2d.FlowSolver2d(self.mesh, Constant(1.))
        options = solver_obj.options
        options.timestepper_type = op.timestepper
        options.timestep = op.dt
        options.simulation_export_time = op.dt*op.dt_per_export
        options.simulation_end_time = self.step_end-0.5*op.dt
        options.output_directory = 'outputs/adjoint' if adjoint else self.di
        options.fields_to_export_hdf5 = []
        if op.plot_pvd:
            options.fields_to_export = ['tracer_2d']
        elif not adjoint:
            options.no_exports = True
        options.solve_tracer = True
        options.lax_friedrichs_tracer = self.stabilisation == 'lax_friedrichs'
        options.tracer_only = True
        options.horizontal_diffusivity = self.nu
        if hasattr(self, 'source'):
            options.tracer_source_2d = self.source
        velocity = -self.u if adjoint else self.u
        init = self.adjoint_solution if adjoint else self.solution
        solver_obj.assign_initial_conditions(elev=Function(self.V), uv=velocity, tracer=init)

        # set up callbacks
        # cb = callback.TracerMassConservation2DCallback('tracer_2d', solver_obj)
        # if self.remesh_step == 0:
        #     self.init_norm = cb.initial_value
        # else:
        #     cb.initial_value = self.init_norm
        # solver_obj.add_callback(cb, 'export')

        # Ensure correct iteration count
        solver_obj.i_export = self.remesh_step
        solver_obj.next_export_t = self.remesh_step*op.dt*op.dt_per_remesh
        solver_obj.iteration = self.remesh_step*op.dt_per_remesh
        solver_obj.simulation_time = time or self.remesh_step*op.dt*op.dt_per_remesh
        for e in solver_obj.exporters.values():
            e.set_next_export_ix(solver_obj.i_export)

        # Set SIPG parameter
        options.use_automatic_sipg_parameter = True
        solver_obj.create_equations()
        self.sipg_parameter = options.sipg_parameter

        def store_old_value():
            """
            Script for storing previous iteration to HDF5, as well as current.
            """
            i = index_string(self.num_exports - int(solver_obj.iteration/self.op.dt_per_export))
            print(i)
            filename = 'Adjoint2d_{:5s}'.format(i)
            filepath = 'outputs/adjoint/hdf5'
            with DumbCheckpoint(os.path.join(filepath, filename), mode=FILE_CREATE) as dc:
                dc.store(solver_obj.timestepper.timesteppers.tracer.solution)
                dc.store(solver_obj.timestepper.timesteppers.tracer.solution_old)
                dc.close()

        # Solve
        solver_obj.bnd_functions['tracer'] = self.op.adjoint_boundary_conditions if adjoint else self.op.boundary_conditions
        if adjoint:
            solver_obj.iterate(export_func=store_old_value)
            self.adjoint_solution.assign(solver_obj.fields.tracer_2d)
        else:
            solver_obj.iterate()
            self.solution.assign(solver_obj.fields.tracer_2d)
            self.solution_old.assign(solver_obj.timestepper.timesteppers.tracer.solution_old)
            self.ts = solver_obj.timestepper.timesteppers.tracer

    def get_timestepper(self, adjoint=False):
        self.set_fields()
        op = self.op

        # Create solver and pass parameters, etc.
        solver_obj = solver2d.FlowSolver2d(self.mesh, Constant(1.))
        options = solver_obj.options
        options.timestepper_type = op.timestepper
        options.timestep = op.dt
        options.simulation_end_time = 0.9*op.dt
        options.fields_to_export = []
        options.solve_tracer = True
        options.lax_friedrichs_tracer = self.stabilisation == 'lax_friedrichs'
        options.tracer_only = True
        options.horizontal_diffusivity = self.nu
        if hasattr(self, 'source'):
            options.tracer_source_2d = self.source
        velocity = -self.u if adjoint else self.u
        solver_obj.assign_initial_conditions(elev=Function(self.V), uv=velocity, tracer=self.solution)

        # Ensure correct iteration count
        solver_obj.i_export = self.remesh_step
        solver_obj.next_export_t = self.remesh_step*op.dt*op.dt_per_remesh
        solver_obj.iteration = self.remesh_step*op.dt_per_remesh
        solver_obj.simulation_time = self.remesh_step*op.dt*op.dt_per_remesh
        for e in solver_obj.exporters.values():
            e.set_next_export_ix(solver_obj.i_export)

        # Set SIPG parameter
        options.use_automatic_sipg_parameter = True
        solver_obj.create_equations()
        self.sipg_parameter = options.sipg_parameter

        # Solve
        solver_obj.bnd_functions['tracer'] = op.boundary_conditions
        solver_obj.create_timestepper()
        if not adjoint:
            self.ts = solver_obj.timestepper.timesteppers.tracer

    def get_hessian(self, adjoint=False):
        f = self.adjoint_solution if adjoint else self.solution
        return steady_metric(f, mesh=self.mesh, noscale=True, op=self.op)

    def get_hessian_metric(self, adjoint=False):
        field_for_adapt = self.adjoint_solution if adjoint else self.solution
        if norm(field_for_adapt) > 1e-8:
            self.M = steady_metric(field_for_adapt, op=self.op)
        else:
            self.M = None

    def get_ts_components(self, adjoint=False):
        self.get_adjoint_state()
        phi_new = self.adjoint_solution_old if adjoint else self.ts.solution
        phi_old = self.adjoint_solution if adjoint else self.ts.solution_old
        adj_new = self.ts.solution if adjoint else self.adjoint_solution_old
        adj_old = self.ts.solution_old if adjoint else self.adjoint_solution

        # Account for timestepping
        if self.op.timestepper == 'CrankNicolson':
            phi = 0.5*(phi_new + phi_old)
            adj = 0.5*(adj_new + adj_old)
        elif self.op.timestepper == 'ForwardEuler':
            phi = phi_old
            adj = adj_new
        elif self.op.timestepper == 'BackwardEuler':
            phi = phi_new
            adj = adj_old
        else:
            raise NotImplementedError
        return phi, phi_new, phi_old, adj, adj_new, adj_old

    def get_dwr_residual(self, adjoint=False, weighted=True):
        i = self.p0test
        phi, phi_new, phi_old, adj, adj_new, adj_old = self.get_ts_components(adjoint)

        # TODO: time-dependent u case
        # TODO: non divergence-free u case
        uv = self.u
        f = self.kernel if adjoint else self.source
        F = f - (phi_new-phi_old)/self.op.dt - dot(uv, grad(phi)) + div(self.nu*grad(phi))

        if weighted:
            dwr = inner(F, adj)
            self.indicators['dwr_cell'] = assemble(i*dwr*dx)
        else:
            self.indicators['strong_residual'] = assemble(F*i*dx)

    def get_dwr_flux(self, adjoint=False):
        phi, phi_new, phi_old, adj, adj_new, adj_old = self.get_ts_components(adjoint)
        # TODO: time-dependent u case
        # TODO: non divergence-free u case
        uv = self.u
        nu = self.nu
        n = self.n
        i = self.p0test
        h = self.h
        degree = self.finite_element.degree()
        family = self.finite_element.family()

        flux_terms = 0

        # Term resulting from integration by parts in advection term
        loc = i*dot(uv, n)*phi*adj
        flux_integrand = (loc('+') + loc('-'))

        # Term resulting from integration by parts in diffusion term
        loc = -i*dot(nu*grad(phi), n)*adj
        flux_integrand += (loc('+') + loc('-'))
        bdy_integrand = loc

        uv_av = avg(uv)
        un_av = dot(uv_av, n('-'))
        s = 0.5*(sign(un_av) + 1.0)
        phi_up = phi('-')*s + phi('+')*(1-s)

        # Interface term  # NOTE: there are some cancellations with IBP above
        loc = i*dot(uv, n)*adj
        flux_integrand += -phi_up*(loc('+') + loc('-'))

        # TODO: Lax-Friedrichs

        # SIPG term
        alpha = avg(self.sipg_parameter/h)
        loc = i*n*adj
        flux_integrand += -alpha*inner(avg(nu)*jump(phi, n), loc('+') + loc('-'))
        flux_integrand += inner(jump(nu*grad(phi)), loc('+') + loc('-'))
        loc = i*nu*grad(adj)
        flux_integrand += 0.5*inner(loc('+') + loc('-'), jump(phi, n))

        flux_terms += flux_integrand*dS

        # Boundary conditions
        bcs = self.op.boundary_conditions
        for j in bcs:
            bdy_j_integrand = bdy_integrand
            if 'value' in bcs[j]:
                phi_ext = bcs[j]['value']
                uv_av = 0.5*(uv + phi_ext)
                un_av = dot(n, uv_av)
                s = 0.5*(sign(un_av) + 1.0)
                phi_up = (phi - phi_ext)*(1 - s)
                bdy_j_integrand += -i*phi_up*dot(uv_av, n)*adj
                bdy_j_integrand += i*dot(uv, n)*phi*adj
            flux_terms += bdy_j_integrand*ds(j)

        # Solve auxiliary finite element problem to get traces on particular element
        mass_term = i*self.p0trial*dx
        res = Function(self.P0)
        solve(mass_term == flux_terms, res)
        self.estimators['dwr_flux'] = abs(assemble(res*dx))
        self.indicators['dwr_flux'] = res
