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
    def __init__(self, op, mesh=None, **kwargs):
        try:
            assert op.family in ("Discontinuous Lagrange", "DG", "dg")
        except AssertionError:
            raise ValueError("Finite element '{:s}' not supported in Thetis tracer model.".format(op.family))
        super(SteadyTracerProblem2d_Thetis, self).__init__(op, mesh, **kwargs)

    def solve_forward(self):
        solver_obj = solver2d.FlowSolver2d(self.mesh, Constant(1.0))
        options = solver_obj.options
        options.timestepper_type = 'SteadyState'
        options.timestep = 20.
        options.simulation_end_time = 0.9*options.timestep
        if self.op.plot_pvd:
            options.fields_to_export = ['tracer_2d']
        else:
            options.no_exports = True
        options.solve_tracer = True
        options.use_lax_friedrichs_tracer = self.stabilisation == 'lax_friedrichs'
        options.tracer_only = True
        options.horizontal_diffusivity = self.fields['diffusivity']
        options.tracer_source_2d = self.fields['source']
        solver_obj.assign_initial_conditions(elev=Function(self.P1), uv=self.fields['velocity'])

        # Set SIPG parameter
        options.use_automatic_sipg_parameter = True
        solver_obj.create_equations()
        self.sipg_parameter = options.sipg_parameter

        # Assign BCs and solve
        solver_obj.bnd_functions['tracer'] = self.op.boundary_conditions
        solver_obj.iterate()
        self.solution.assign(solver_obj.fields.tracer_2d)
        self.lhs = self.solver_obj.timestepper.F

    def solve_continuous_adjoint(self):
        raise NotImplementedError  # TODO

    def get_strong_residual(self, adjoint=False, norm_type=None):
        u = self.fields['velocity']
        nu = self.fields['diffusivity']
        f = self.kernel if adjoint else self.fields['source']

        sol = self.adjoint_solution if adjoint else self.solution
        name = 'cell_residual'
        name = '_'.join([name, 'adjoint' if adjoint else 'forward'])

        # TODO: non divergence-free u case
        R = f - dot(u, grad(sol)) + div(nu*grad(sol))

        if norm_type is None:
            self.indicators[name] = assemble(self.p0test*R*dx)
        elif norm_type == 'L1':
            self.indicators[name] = assemble(self.p0test*abs(R)*dx)
        elif norm_type == 'L2':
            self.indicators[name] = assemble(self.p0test*R*R*dx)
        else:
            raise ValueError("Norm should be chosen from {None, 'L1' or 'L2'}.")
        self.estimate_error(name)
        return name

    def get_dwr_residual(self, adjoint=False):
        u = self.fields['velocity']
        nu = self.fields['diffusivity']
        f = self.kernel if adjoint else self.fields['source']

        sol = self.adjoint_solution if adjoint else self.solution
        adj = self.solution if adjoint self.adjoint_solution

        # TODO: non divergence-free u case
        F = f - dot(u, grad(sol)) + div(nu*grad(sol))
        res_name = self.get_strong_residual(adjoint=adjoint, norm_type=None)
        name = 'dwr_cell'
        name = '_'.join([name, 'adjoint' if adjoint else 'forward'])

        self.indicators[name] = assemble(self.p0test*self.indicators[res_name]*adj*dx)
        self.estimate_error(name)
        return name

    def get_dwr_flux(self, adjoint=False):
        i = self.p0test
        n = self.n
        h = self.h
        u = self.fields['velocity']
        nu = self.fields['diffusivity']
        degree = self.finite_element.degree()
        family = self.finite_element.family()

        sol = self.adjoint_solution if adjoint else self.solution
        adj = self.solution if adjoint self.adjoint_solution

        # TODO: non divergence-free u case

        flux_terms = 0

        # Term resulting from integration by parts in advection term
        loc = i*dot(u, n)*sol*adj
        flux_integrand = (loc('+') + loc('-'))

        # Term resulting from integration by parts in diffusion term
        loc = -i*dot(nu*grad(sol), n)*adj
        flux_integrand += (loc('+') + loc('-'))
        bdy_integrand = loc

        u_av = avg(u)
        un_av = dot(u_av, n('-'))
        s = 0.5*(sign(un_av) + 1.0)
        sol_up = sol('-')*s + sol('+')*(1-s)

        # Interface term  # NOTE: there are some cancellations with IBP above
        loc = i*dot(u, n)*adj
        flux_integrand += -sol_up*(loc('+') + loc('-'))

        # TODO: Lax-Friedrichs

        # SIPG term
        alpha = avg(self.sipg_parameter/h)
        loc = i*n*adj
        flux_integrand += -alpha*inner(avg(nu)*jump(sol, n), loc('+') + loc('-'))
        flux_integrand += inner(jump(nu*grad(sol)), loc('+') + loc('-'))
        loc = i*nu*grad(adj)
        flux_integrand += 0.5*inner(loc('+') + loc('-'), jump(sol, n))

        flux_terms += flux_integrand*dS

        # Boundary conditions
        bcs = self.op.boundary_conditions
        raise NotImplementedError  # TODO: BCs

        # Solve auxiliary finite element problem to get traces on particular element
        name = 'dwr_flux'
        name = '_'.join([name, 'adjoint' if adjoint else 'forward'])
        mass_term = i*self.p0trial*dx
        self.indicators[name] = Function(self.P0)
        solve(mass_term == flux_terms, self.indicators[name])
        self.estimate_error(name)
        return name


# TODO: revise this
class UnsteadyTracerProblem2d_Thetis(UnsteadyTracerProblem2d):
    def __init__(self, op, mesh=None, **kwargs):
        super(UnsteadyTracerProblem2d_Thetis, self).__init__(op, mesh, **kwargs)

        # Classification
        self.nonlinear = False

    def set_fields(self):
        op = self.op
        self.fields = {}
        self.fields['diffusivity'] = op.set_diffusivity(self.P1)
        self.fields['velocity'] = op.set_velocity(self.P1_vec)
        self.fields['source'] = op.set_source(self.P1)
        self.divergence_free = np.allclose(assemble(div(self.fields['velocity'])*dx), 0.0)

        # Arbitrary field to take gradient for discrete adjoint
        self.gradient_field = self.fields['diffusivity']

    def solve_step(self, adjoint=False):
        self.set_fields()
        if adjoint:
            try:
                assert not self.discrete_adjoint
            except AssertionError:
                raise NotImplementedError  # TODO
            try:
                assert self.divergence_free
            except AssertionError:
                raise NotImplementedError  # TODO
        op = self.op

        # Create solver and pass parameters, etc.
        solver_obj = solver2d.FlowSolver2d(self.mesh, Constant(1.0))
        options = solver_obj.options
        options.timestepper_type = op.timestepper
        options.timestep = op.dt
        options.simulation_export_time = op.dt*op.dt_per_export
        options.simulation_end_time = self.step_end - 0.5*op.dt
        if adjoint:
            self.di = create_directory(os.path.join(self.di, 'adjoint'))
        options.output_directory = self.di
        options.fields_to_export_hdf5 = []
        if op.plot_pvd:
            options.fields_to_export = ['tracer_2d']
        else:
            options.no_exports = True
        options.solve_tracer = True
        options.use_lax_friedrichs_tracer = self.stabilisation == 'lax_friedrichs'
        options.tracer_only = True
        options.horizontal_diffusivity = self.fields['diffusivity']
        options.tracer_source_2d = self.fields['source']
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

    def get_qoi_kernel(self):
        self.kernel = self.op.set_qoi_kernel(self.P0)
        return self.kernel
