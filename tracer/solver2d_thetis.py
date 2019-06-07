from thetis_adjoint import *
import pyadjoint

from time import clock
import numpy as np

from adapt_utils.tracer.options import *
from adapt_utils.adapt.metric import *
from adapt_utils.solver import SteadyProblem, UnsteadyProblem


__all__ = ["SteadyTracerProblem_DG", "UnsteadyTracerProblem_DG"]


class SteadyTracerProblem_DG(SteadyProblem):
    r"""
    General discontinuous Galerkin solver object for stationary tracer advection problems of the form

..  math::
    \textbf{u} \cdot \nabla(\phi) - \nabla \cdot (\nu \cdot \nabla(\phi)) = f,

    for (prescribed) velocity :math:`\textbf{u}`, diffusivity :math:`\nu \geq 0`, source :math:`f`
    and (prognostic) concentration :math:`\phi`.
    """
    def __init__(self,
                 op=PowerOptions(),
                 stab=None,
                 mesh=SquareMesh(40, 40, 4, 4),
                 discrete_adjoint=True,
                 high_order=False,
                 prev_solution=None):
        if op.family == 'dg':
             finite_element = FiniteElement("Discontinuous Lagrange", triangle, 1)
        elif op.family == 'cg':
             finite_element = FiniteElement("Lagrange", triangle, 1)
        else:
             raise NotImplementedError
        if mesh is None:
            mesh = op.default_mesh
        super(SteadyTracerProblem_DG, self).__init__(mesh,
                                                     op,
                                                     finite_element,
                                                     stab,
                                                     discrete_adjoint,
                                                     None)
        assert(finite_element.family() == "Discontinuous Lagrange")

        # Extract parameters from Options class
        self.nu = op.set_diffusivity(self.P1)
        self.u = op.set_velocity(self.P1_vec)
        self.source = op.set_source(self.P1)
        self.kernel = op.set_objective_kernel(self.P1)
        self.gradient_field = self.nu  # arbitrary field to take gradient for discrete adjoint

        # Rename solution fields
        self.solution.rename('Tracer concentration')
        self.adjoint_solution.rename('Adjoint tracer concentration')

        # Classification
        self.nonlinear = False

    def solve(self):
        bcs = self.op.boundary_conditions
        BCs = {'shallow water': {}, 'tracer': {}}
        for i in bcs.keys():
            if bcs[i] == 'dirichlet_zero':
                bcs[i] = {'value': Constant(0.)}
                BCs['tracer'][i] = {'value': Constant(0.)}
            elif bcs[i] == 'neumann_zero':
                continue
                # TODO: Neumann conditions not currently implemented for tracers
        b = Function(self.P1).assign(1.)
        eta = Function(self.P1)

        solver_obj = solver2d.FlowSolver2d(self.mesh, b)
        options = solver_obj.options
        options.timestepper_type = 'SteadyState'
        options.timestep = 20.
        options.simulation_end_time = 0.9*options.timestep
        options.fields_to_export = ['tracer_2d']
        options.compute_residuals_tracer = True
        options.solve_tracer = True
        options.tracer_only = True
        options.horizontal_diffusivity = self.nu
        options.tracer_source_2d = self.source
        solver_obj.assign_initial_conditions(elev=eta, uv=self.u)
        solver_obj.bnd_functions = BCs
        solver_obj.iterate()
        self.solution = solver_obj.fields.tracer_2d

    def solve_continuous_adjoint(self):
        raise NotImplementedError

    def get_hessian_metric(self, adjoint=False):
        self.M = steady_metric(self.adjoint_solution if adjoint else self.solution, op=self.op)

    def explicit_estimation(self):
        phi = self.solution
        i = TestFunction(self.P0)
        bcs = self.op.boundary_conditions

        # Compute residuals
        self.cell_res = dot(self.u, grad(phi)) - div(self.nu*grad(phi))
        self.edge_res = phi*dot(self.u, self.n) - self.nu*dot(self.n, nabla_grad(phi))
        R = self.cell_res
        r = self.edge_res

        # Assemble cell residual
        R_norm = assemble(i*R*R*dx)

        # Solve auxiliary problem to assemble edge residual
        r_norm = TrialFunction(self.P0)
        mass_term = i*r_norm*dx
        flux_terms = ((i*r*r)('+') + (i*r*r)('-'))*dS
        for j in bcs.keys():
            if bcs[j] == 'neumann_zero':
                flux_terms += i*r*r*ds(j)
            if bcs[j] == 'dirichlet_zero':
                flux_terms += i*phi*phi*ds(j)
        raise NotImplementedError  # TODO: flux terms
        r_norm = Function(self.P0)
        solve(mass_term == flux_terms, r_norm)

        # Form error estimator
        self.indicator = project(sqrt(self.h*self.h*R_norm + 0.5*self.h*r_norm), self.P0)
        self.indicator.rename('explicit')

    def explicit_estimation_adjoint(self):
        phi = self.solution
        lam = self.adjoint_solution
        u = self.u
        nu = self.nu
        n = self.n
        i = TestFunction(self.P0)
        bcs = self.op.boundary_conditions

        # Cell residual
        R = -div(u*lam) - div(nu*grad(lam))
        R_norm = assemble(i*R*R*dx)

        # Edge residual
        r = TrialFunction(self.P0)
        mass_term = i*r*dx
        flux = - lam*phi*dot(u, n) - nu*phi*dot(n, nabla_grad(lam))
        flux_terms = ((i*flux*flux)('+') + (i*flux*flux)('-')) * dS
        for j in bcs.keys():
            if bcs[j] != 'dirichlet_zero':
                flux_terms += i*flux*flux*ds(j)  # Robin BC in adjoint
            if bcs[j] != 'neumann_zero':
                flux_terms += i*lam*lam*ds(j)    # Dirichlet BC in adjoint
        raise NotImplementedError  # TODO: flux terms
        r_norm = Function(self.P0)
        solve(mass_term == flux_terms, r_norm)

        # Form error estimator
        self.indicator = project(sqrt(self.h*self.h*R_norm + 0.5*self.h*r_norm), self.P0)
        self.indicator.rename('explicit_adjoint')
 
    def dwr_estimation(self):
        i = TestFunction(self.P0)
        phi = self.solution
        u = self.u
        nu = self.nu
        n = self.n
        f = self.source
        bcs = self.op.boundary_conditions

        if self.high_order:
            lam = self.solve_high_order(adjoint=True)
        else:
            lam = self.adjoint_solution

        # Cell residual
        R = (f - dot(u, grad(phi)) + div(nu*grad(phi)))*lam

        # Edge residual
        r = TrialFunction(self.P0)
        mass_term = i*r*dx
        flux = nu*lam*dot(n, nabla_grad(phi))
        flux_terms = ((i*flux)('+') + (i*flux)('-'))*dS
        for j in bcs.keys():
            if bcs[j] == 'neumann_zero':
                flux_terms += i*flux*ds(j)
            if bcs[j] == 'dirichlet_zero':
                flux_terms += -i*phi*ds(j)
        raise NotImplementedError  # TODO: flux terms
        r = Function(self.P0)
        solve(mass_term == flux_terms, r)

        # Sum
        self.cell_res = R
        self.edge_res = r
        self.indicator = project(R + r, self.P0)
        self.indicator.rename('dwr')
        
    def dwr_estimation_adjoint(self):
        i = TestFunction(self.P0)
        lam = self.adjoint_solution
        u = self.u
        nu = self.nu
        n = self.n
        bcs = self.op.boundary_conditions
        
        if self.high_order:
            phi = self.solve_high_order(adjoint=False)
        else:
            phi = self.solution
            
        # Adjoint source term
        dJdphi = self.op.box(self.P0)
            
        # Cell residual
        R = (dJdphi + div(u*lam) + div(nu*grad(lam)))*phi
        
        # Edge residual
        r = TrialFunction(self.P0)
        mass_term = i*r*dx
        flux = - lam*phi*dot(u, n) - nu*phi*dot(n, nabla_grad(lam))
        flux_terms = ((i*flux)('+') + (i*flux)('-')) * dS
        for j in bcs.keys():
            if bcs[j] != 'dirichlet_zero':
                flux_terms += i*flux*ds(j)  # Robin BC in adjoint
            if bcs[j] != 'neumann_zero':
                flux_terms += -i*lam*ds(j)  # Dirichlet BC in adjoint
        raise NotImplementedError  # TODO: flux terms
        r = Function(self.P0)
        solve(mass_term == flux_terms, r)

        self.cell_res_adjoint = R
        self.edge_res_adjoint = r
        self.indicator = project(R + r, self.P0)
        self.indicator.rename('dwr_adjoint')
        
    def get_anisotropic_metric(self, adjoint=False, relax=False, superpose=True):
        assert not (relax and superpose)

        # Solve adjoint problem
        if self.high_order:
            adj = self.solve_high_order(adjoint=not adjoint)
        else:
            adj = self.solution if adjoint else self.adjoint_solution
        sol = self.adjoint_solution if adjoint else self.solution
        adj_diff = Function(self.P1_vec).interpolate(abs(construct_gradient(adj)))
        adj.interpolate(abs(adj))

        # Get potential to take Hessian w.r.t.
        x, y = SpatialCoordinate(self.mesh)
        if adjoint:
            source = self.op.box(self.P0)
            # F1 = -sol*self.u[0] - self.nu*sol.dx(0) - source*x
            # F2 = -sol*self.u[1] - self.nu*sol.dx(1) - source*y
            F1 = -sol*self.u[0] - self.nu*sol.dx(0)
            F2 = -sol*self.u[1] - self.nu*sol.dx(1)
        else:
            source = self.source
            # F1 = sol*self.u[0] - self.nu*sol.dx(0) - source*x
            # F2 = sol*self.u[1] - self.nu*sol.dx(1) - source*y
            F1 = sol*self.u[0] - self.nu*sol.dx(0)
            F2 = sol*self.u[1] - self.nu*sol.dx(1)

        # NOTES:
        #  * The derivation for the second potential uses the fact that u is divergence free (in
        #    particular, it is constant).
        #  * It is NOT the case that div(F) = f when the commented-out versions of F1 and F2 are
        #    used. In fact:
        #                    div(F)(x) = 1     if x in A
        #                                infty if x in partial A
        #                                0     else
        #    Using these forms lead to high resolution of the boundaries of the source / receiver
        #    regions and low resolution elsewhere.

        # Construct Hessians
        H1 = construct_hessian(F1, mesh=self.mesh, op=self.op)
        H2 = construct_hessian(F2, mesh=self.mesh, op=self.op)
        Hf = construct_hessian(source, mesh=self.mesh, op=self.op)

        # form metric
        self.M = Function(self.P1_ten)
        for i in range(len(adj.dat.data)):
            self.M.dat.data[i][:,:] += H1.dat.data[i]*adj_diff.dat.data[i][0]
            self.M.dat.data[i][:,:] += H2.dat.data[i]*adj_diff.dat.data[i][1]
            if relax:
                self.M.dat.data[i][:,:] += Hf.dat.data[i]*adj.dat.data[i]
        self.M = steady_metric(None, H=self.M, op=self.op)

        if superpose:
            Mf = Function(self.P1_ten)
            Mf.interpolate(Hf*adj)
            self.M = metric_intersection(self.M, Mf)

        # TODO: boundary contributions
        # bdy_contributions = i*(F1*n[0] + F2*n[1])*ds
        # n = self.n
        # Fhat = i*dot(phi, n)
        # bdy_contributions -= Fhat*ds(2) + Fhat*ds(3) + Fhat*ds(4)

        # TODO: flux terms?


class UnsteadyTracerProblem_DG(UnsteadyProblem):
    def __init__(self,
                 op,
                 mesh=None,
                 discrete_adjoint=True,
                 finite_element=FiniteElement("Discontinuous Lagrange", triangle, 1)):
        super(UnsteadyTracerProblem_DG, self).__init__(mesh, op, finite_element, discrete_adjoint)
        assert(finite_element.family() == "Discontinuous Lagrange")

        self.set_fields()
        self.solution = self.op.set_initial_condition(self.V)

        # Classification
        self.nonlinear = False

    def set_fields(self):
        self.nu = self.op.set_diffusivity(self.V)
        self.u = self.op.set_velocity(self.P1_vec)
        if hasattr(self.op, 'source'):
            self.source = self.op.set_source(self.V)
        self.kernel = self.op.set_objective_kernel(self.V)
        self.gradient_field = self.nu  # arbitrary field to take gradient for discrete adjoint

        # Rename solution fields
        self.solution.rename('Tracer concentration')
        self.adjoint_solution.rename('Adjoint tracer concentration')

    def solve_step(self):
        self.set_fields()
        one = Function(self.V).assign(1.)
        zero = Function(self.V)
        op = self.op

        # create solver and pass parameters, etc.
        solver_obj = solver2d.FlowSolver2d(self.mesh, one)
        options = solver_obj.options
        options.timestepper_type = op.timestepper
        options.timestep = op.dt
        options.simulation_export_time = op.dt*op.dt_per_export
        options.simulation_end_time = self.step_end-0.5*op.dt
        options.output_directory = self.di
        options.fields_to_export = ['tracer_2d']
        options.compute_residuals_tracer = True
        options.solve_tracer = True
        options.tracer_only = True
        options.horizontal_diffusivity = self.nu
        if hasattr(self, 'source'):
            options.tracer_source_2d = self.source
        solver_obj.assign_initial_conditions(elev=zero, uv=self.u, tracer=self.solution)

        # set up callbacks
        #cb = callback.TracerMassConservation2DCallback('tracer_2d', solver_obj)
        #if self.remesh_step == 0:
        #    self.init_norm = cb.initial_value
        #else:
        #    cb.initial_value = self.init_norm
        #solver_obj.add_callback(cb, 'export')

        # ensure correct iteration count
        solver_obj.i_export = self.remesh_step
        solver_obj.next_export_t = self.remesh_step*op.dt*op.dt_per_remesh
        solver_obj.iteration = self.remesh_step*op.dt_per_remesh
        solver_obj.simulation_time = self.remesh_step*op.dt*op.dt_per_remesh
        for e in solver_obj.exporters.values():
            e.set_next_export_ix(solver_obj.i_export)

        # solve
        solver_obj.bnd_functions['tracer'] = op.boundary_conditions
        solver_obj.iterate()
        self.solution = solver_obj.fields.tracer_2d
        self.ts = solver_obj.timestepper.timesteppers.tracer

    def get_timestepper(self):
        self.set_fields()
        one = Function(self.V).assign(1.)
        zero = Function(self.V)
        op = self.op

        # create solver and pass parameters, etc.
        solver_obj = solver2d.FlowSolver2d(self.mesh, one)
        options = solver_obj.options
        options.timestepper_type = op.timestepper
        options.timestep = op.dt
        options.simulation_end_time = 0.9*op.dt
        options.fields_to_export = []
        options.compute_residuals_tracer = True
        options.solve_tracer = True
        options.tracer_only = True
        options.horizontal_diffusivity = self.nu
        if hasattr(self, 'source'):
            options.tracer_source_2d = self.source
        solver_obj.assign_initial_conditions(elev=zero, uv=self.u, tracer=self.solution)

        # ensure correct iteration count
        solver_obj.i_export = self.remesh_step
        solver_obj.next_export_t = self.remesh_step*op.dt*op.dt_per_remesh
        solver_obj.iteration = self.remesh_step*op.dt_per_remesh
        solver_obj.simulation_time = self.remesh_step*op.dt*op.dt_per_remesh
        for e in solver_obj.exporters.values():
            e.set_next_export_ix(solver_obj.i_export)

        # solve
        solver_obj.bnd_functions['tracer'] = op.boundary_conditions
        solver_obj.create_timestepper()
        self.ts = solver_obj.timestepper.timesteppers.tracer

    def get_hessian_metric(self, adjoint=False):
        field_for_adapt = self.adjoint_solution if adjoint else self.solution
        if norm(field_for_adapt) > 1e-8:
            self.M = steady_metric(field_for_adapt, op=self.op)
        else:
            self.M = None

    def dwr_estimation(self):
        if self.remesh_step == 0:
            self.get_timestepper()
        cell_res = self.ts.cell_residual(adjoint=self.interpolated_adjoint_solution)
        edge_res = self.ts.edge_residual(adjoint=self.interpolated_adjoint_solution)
        self.indicator = Function(self.P0, name='dwr')
        self.indicator.interpolate(cell_res + edge_res)
