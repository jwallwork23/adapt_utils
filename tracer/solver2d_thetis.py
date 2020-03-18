from thetis import *

from adapt_utils.tracer.options import *
from adapt_utils.tracer.solver2d import *
from adapt_utils.adapt.metric import *
from adapt_utils.misc import index_string


__all__ = ["SteadyTracerProblem2d_Thetis", "UnsteadyTracerProblem2d_Thetis"]


# FIXME: SteadyState timestepper doesn't work for tracers
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
        fe = FiniteElement("Discontinuous Lagrange", triangle, op.degree)
        super(SteadyTracerProblem2d_Thetis, self).__init__(op, mesh, **kwargs)

    def setup_solver_forward(self):
        op = self.op
        self.solver_obj = solver2d.FlowSolver2d(self.mesh, Constant(1.0))
        options = self.solver_obj.options
        options.timestepper_type = 'SteadyState'
        options.timestep = 20.
        options.simulation_end_time = 0.9*options.timestep
        if op.plot_pvd:
            options.fields_to_export = ['tracer_2d']
        else:
            options.no_exports = True
        options.solve_tracer = True
        options.use_lax_friedrichs_tracer = self.stabilisation == 'lax_friedrichs'
        options.tracer_only = True
        options.horizontal_diffusivity = self.fields['diffusivity']
        options.tracer_source_2d = self.fields['source']
        self.solver_obj.assign_initial_conditions(elev=Function(self.P1), uv=self.fields['velocity'])

        # Set SIPG parameter
        options.use_automatic_sipg_parameter = True
        self.solver_obj.create_equations()
        self.sipg_parameter = options.sipg_parameter

        # Assign BCs
        self.solver_obj.bnd_functions['tracer'] = op.boundary_conditions

    def solve_forward(self):
        self.setup_solver_forward()
        self.solver_obj.iterate()
        self.solution.assign(self.solver_obj.fields.tracer_2d)
        self.lhs = self.solver_obj.timestepper.F

    def get_bnd_functions(self, *args):
        tt = tracer_eq_2d.TracerTerm(self.V)
        return tt.get_bnd_functions(*args, self.boundary_conditions)

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
        adj = self.solution if adjoint else self.adjoint_solution

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
        adj = self.solution if adjoint else self.adjoint_solution

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
        for j in bcs:
            funcs = bcs.get(j)

            if funcs is not None:
                sol_ext, u_ext, eta_ext = tpe.get_bnd_functions(sol, u, eta)
                # TODO

            raise NotImplementedError

        # Solve auxiliary finite element problem to get traces on particular element
        name = 'dwr_flux'
        name = '_'.join([name, 'adjoint' if adjoint else 'forward'])
        mass_term = i*self.p0trial*dx
        self.indicators[name] = Function(self.P0)
        solve(mass_term == flux_terms, self.indicators[name])
        self.estimate_error(name)
        return name


class UnsteadyTracerProblem2d_Thetis(UnsteadyTracerProblem2d):
    # TODO: doc

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
        if not hasattr(self, 'remesh_step'):
            self.remesh_step = 0
        op = self.op

        # Create solver and pass parameters, etc.
        self.solver_obj = solver2d.FlowSolver2d(self.mesh, Constant(1.0))
        options = self.solver_obj.options
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
        options.use_automatic_sipg_parameter = op.sipg_parameter is None
        options.use_lagrangian_formulation = op.approach == 'ale'

        # Assign initial conditions
        velocity = -self.fields['velocity'] if adjoint else self.fields['velocity']
        init = self.adjoint_solution if adjoint else self.solution
        self.solver_obj.assign_initial_conditions(uv=velocity, tracer=init)

        # Set up callbacks
        # cb = callback.TracerMassConservation2DCallback('tracer_2d', self.solver_obj)
        # if self.remesh_step == 0:
        #     self.init_norm = cb.initial_value
        # else:
        #     cb.initial_value = self.init_norm
        # self.solver_obj.add_callback(cb, 'export')

        # Ensure correct iteration count
        self.solver_obj.i_export = self.remesh_step
        self.solver_obj.next_export_t = self.remesh_step*op.dt*op.dt_per_remesh
        self.solver_obj.iteration = self.remesh_step*op.dt_per_remesh
        self.solver_obj.simulation_time = self.remesh_step*op.dt*op.dt_per_remesh
        for e in self.solver_obj.exporters.values():
            e.set_next_export_ix(self.solver_obj.i_export)

        self.sipg_parameter = options.sipg_parameter

        def store_old_value():
            """Script for storing previous iteration to HDF5, as well as current."""
            i = index_string(self.num_exports - int(self.solver_obj.iteration/op.dt_per_export))
            print_output(i)
            filename = 'Adjoint2d_{:5s}'.format(i)
            filepath = 'outputs/adjoint/hdf5'
            with DumbCheckpoint(os.path.join(filepath, filename), mode=FILE_CREATE) as dc:
                dc.store(self.solver_obj.timestepper.timesteppers.tracer.solution)
                dc.store(self.solver_obj.timestepper.timesteppers.tracer.solution_old)
                dc.close()

        # Update forcings / exports
        update_forcings = op.get_update_forcings(self.solver_obj)
        export_func = op.get_export_func(self.solver_obj)
        if adjoint:
            def export_func_wrapper():
                store_old_value()
                export_func()
        else:
            export_func_wrapper = export_func

        # Solve
        bcs = op.adjoint_boundary_conditions if adjoint else op.boundary_conditions
        self.solver_obj.bnd_functions['tracer'] = bcs
        self.solver_obj.iterate(export_func=export_func_wrapper, update_forcings=update_forcings)
        if adjoint:
            self.adjoint_solution.assign(self.solver_obj.fields.tracer_2d)
        else:
            self.solution.assign(self.solver_obj.fields.tracer_2d)
            self.solution_old.assign(self.solver_obj.timestepper.timesteppers.tracer.solution_old)
            # self.lhs = self.solver_obj.timestepper.F
