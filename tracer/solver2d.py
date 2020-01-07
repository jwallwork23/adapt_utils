from firedrake import *

import numpy as np

from adapt_utils.tracer.stabilisation import supg_coefficient, anisotropic_stabilisation
from adapt_utils.adapt.adaptation import *
from adapt_utils.adapt.metric import *
from adapt_utils.adapt.kernels import matscale_sum_kernel
from adapt_utils.adapt.recovery import *
from adapt_utils.adapt.p0_metric import *
from adapt_utils.solver import SteadyProblem, UnsteadyProblem


__all__ = ["SteadyTracerProblem2d", "UnsteadyTracerProblem2d"]


class SteadyTracerProblem2d(SteadyProblem):
    r"""
    General solver object for 2D stationary tracer advection problems of the form

..  math::
    \textbf{u} \cdot \nabla(\phi) - \nabla \cdot (\nu \cdot \nabla(\phi)) = f,

    for (prescribed) velocity :math:`\textbf{u}`, diffusivity :math:`\nu \geq 0`, source :math:`f`
    and (prognostic) concentration :math:`\phi`.

    Defaults to P1 continuous Galerkin with SUPG stabilisation.

    Implemented boundary condition types:
        * Neumann;
        * Dirichlet;
        * outflow.
    """
    def __init__(self,
                 op,
                 mesh=None,
                 finite_element=FiniteElement("Lagrange", triangle, 1),
                 discrete_adjoint=False,
                 prev_solution=None,
                 levels=1):
        super(SteadyTracerProblem2d, self).__init__(op, mesh, finite_element, discrete_adjoint, prev_solution, levels)

        # Extract parameters from Options class
        self.nu = op.set_diffusivity(self.P1)
        self.u = op.set_velocity(self.P1_vec)
        self.divergence_free = np.allclose(norm(div(self.u)), 0.0)
        self.source = op.set_source(self.P1)
        self.kernel = op.set_qoi_kernel(self.P0)
        self.gradient_field = self.nu  # arbitrary field to take gradient for discrete adjoint

        # Stabilisation
        if self.stab is None:
            self.stab = 'SUPG'
        assert self.stab in ('no', 'SU', 'SUPG', 'lax_friedrichs')
        if self.stab in ('SU', 'SUPG'):  # FIXME
            #self.stabilisation = supg_coefficient(self.u, self.nu, mesh=self.mesh, anisotropic=True)
            self.stabilisation = supg_coefficient(self.u, self.nu, mesh=self.mesh, anisotropic=False)
            #self.stabilisation = anisotropic_stabilisation(self.u, mesh=self.mesh)

        # Rename solution fields
        self.solution.rename('Tracer concentration')
        self.adjoint_solution.rename('Adjoint tracer concentration')

        # Classification
        self.nonlinear = False

    def solve_forward(self):
        phi = self.trial
        psi = self.test

        # Finite element problem
        a = psi*dot(self.u, grad(phi))*dx + self.nu*inner(grad(phi), grad(psi))*dx
        L = self.source*psi*dx

        # Stabilisation
        if self.stab in ("SU", "SUPG"):
            coeff = self.stabilisation*dot(self.u, grad(psi))
            a += coeff*dot(self.u, grad(phi))*dx
            if self.stab == "SUPG":
                a += coeff*-div(self.nu*grad(phi))*dx
                L += coeff*self.source*dx
                psi = psi + coeff
        elif not self.stab is None:
            raise ValueError("Unrecognised stabilisation method.")

        # Boundary conditions
        bcs = self.op.boundary_conditions
        dbcs = []
        for i in bcs.keys():
            if bcs[i] == {}:
                a += -self.nu*psi*dot(self.n, nabla_grad(phi))*ds(i)
            if 'diff_flux' in bcs[i]:
                a += -self.nu*psi*dot(self.n, nabla_grad(phi))*ds(i)
                L += -psi*bcs[i]['diff_flux']*ds(i)
            if 'value' in bcs[i]:
                dbcs.append(DirichletBC(self.V, bcs[i]['value'], i))

        # For condition number studies  # TODO: account for RHS
        self.lhs = a

        # Solve
        solve(a == L, self.solution, bcs=dbcs, solver_parameters=self.op.params)
        self.solution_file.write(self.solution)

    def solve_continuous_adjoint(self):
        lam = self.trial
        psi = self.test

        # Adjoint finite element problem
        a = lam*dot(self.u, grad(psi))*dx + self.nu*inner(grad(lam), grad(psi))*dx
        L = self.kernel*psi*dx

        # Stabilisation
        if self.stab in ("SU", "SUPG"):
            coeff = -self.stabilisation*div(self.u*psi)
            a += -coeff*div(self.u*lam)*dx
            if self.stab == 'SUPG':  # NOTE: this is not equivalent to discrete adjoint
                a += coeff*-div(self.nu*grad(lam))*dx
                L += coeff*self.kernel*dx
                psi = psi + coeff
        elif not self.stab is None:
            raise ValueError("Unrecognised stabilisation method.")

        # Boundary conditions
        bcs = self.op.boundary_conditions
        dbcs = []
        for i in bcs.keys():
            if not 'diff_flux' in bcs[i]:
                dbcs.append(DirichletBC(self.V, 0, i))  # Dirichlet BC in adjoint
            if not 'value' in bcs[i]:
                a += -lam*psi*(dot(self.u, self.n))*ds(i)
                a += -self.nu*psi*dot(self.n, nabla_grad(lam))*ds(i)  # Robin BC in adjoint

        # Solve
        solve(a == L, self.adjoint_solution, bcs=dbcs, solver_parameters=self.op.params)
        self.adjoint_solution_file.write(self.adjoint_solution)

    def get_hessian(self, adjoint=False):
        f = self.adjoint_solution if adjoint else self.solution
        return steady_metric(f, mesh=self.mesh, noscale=True, op=self.op)

    def get_hessian_metric(self, adjoint=False):
        self.M = steady_metric(self.adjoint_solution if adjoint else self.solution, op=self.op)

    def get_strong_residual_forward(self):
        R = self.source - dot(self.u, grad(self.solution)) + div(self.nu*grad(self.solution))
        self.indicators['cell_res_forward'] = assemble(self.p0test*abs(R)*dx)
        self.indicator = interpolate(self.indicators['cell_res_forward'], self.P1)
        self.indicator.rename('forward strong residual')

    def get_strong_residual_adjoint(self):
        R = self.kernel + div(self.u*self.adjoint_solution) + div(self.nu*grad(self.adjoint_solution))
        self.indicators['cell_res_adjoint'] = assemble(self.p0test*abs(R)*dx)
        self.indicator = interpolate(self.indicators['cell_res_adjoint'], self.P1)
        self.indicator.rename('adjoint strong residual')

    def get_dwr_residual_forward(self, sol, adjoint_sol):  # FIXME: Inputs are unused
        tpe = self.tp_enriched
        phi = project(self.solution, tpe.V)  # FIXME: prolong
        if not hasattr(self, 'adjoint_error'):
            self.solve_high_order(adjoint=True)
        strong_residual = tpe.source - dot(tpe.u, grad(phi)) + div(tpe.nu*grad(phi))
        dwr = strong_residual*self.adjoint_error
        if self.stab == 'SUPG':  # Account for stabilisation error
            dwr += strong_residual*tpe.stabilisation*dot(tpe.u, grad(self.adjoint_error))
        self.indicators['dwr_cell'] = project(assemble(tpe.p0test*dwr*dx), self.P0)  # FIXME: inject?
        self.indicators['dwr_cell'].interpolate(abs(self.indicators['dwr_cell']))
        self.estimators['dwr_cell'] = self.indicators['dwr_cell'].vector().gather().sum()

    def get_dwr_flux_forward(self, sol, adjoint_sol):  # FIXME: Inputs are unused
        tpe = self.tp_enriched
        i = tpe.p0test
        phi = project(self.solution, tpe.V)  # FIXME: prolong
        if not hasattr(self, 'adjoint_error'):
            self.solve_high_order(adjoint=True)

        # Flux terms (arising from integration by parts)
        mass_term = i*tpe.p0trial*dx
        flux = -tpe.nu*dot(tpe.n, nabla_grad(phi))
        dwr = flux*self.adjoint_error
        if self.stab == 'SUPG':  # Account for stabilisation error
            coeff = tpe.stabilisation*dot(tpe.u, grad(self.adjoint_error))
            dwr += coeff*flux
        flux_terms = ((i*dwr)('+') + (i*dwr)('-'))*dS

        # Account for boundary conditions
        # NOTES:
        #   * For CG methods, Dirichlet error is zero, by construction.
        #   * Negative sign in `flux`.
        bcs = tpe.op.boundary_conditions
        for j in bcs.keys():
            if 'diff_flux' in bcs[j]:
                flux_terms += i*(dwr + bcs[j]['diff_flux']*self.adjoint_error)*ds(j)
                if self.stab == "SUPG":
                    flux_terms += i*bcs[j]['diff_flux']*coeff*ds(j)

        # Solve auxiliary FEM problem
        edge_res = Function(tpe.P0)
        solve(mass_term == flux_terms, edge_res)
        self.indicators['dwr_flux'] = project(edge_res, self.P0)  # FIXME: inject?
        self.indicators['dwr_flux'].interpolate(abs(self.indicators['dwr_flux']))
        self.estimators['dwr_flux'] = self.indicators['dwr_flux'].vector().gather().sum()

    def get_dwr_residual_adjoint(self, sol, adjoint_sol):  # FIXME: Inputs are unused
        tpe = self.tp_enriched
        lam = project(self.adjoint_solution, tpe.V)  # FIXME: prolong
        if not hasattr(self, 'error'):
            self.solve_high_order(adjoint=False)
        strong_residual = tpe.op.box(tpe.P0) + div(tpe.u*lam) + div(tpe.nu*grad(lam))
        dwr = strong_residual*self.error
        if self.stab == 'SUPG':  # Account for stabilisation error
            dwr += strong_residual*tpe.stabilisation*dot(tpe.u, grad(self.error))
        self.indicators['dwr_cell_adjoint'] = project(assemble(tpe.p0test*dwr*dx), self.P0)  # FIXME: inject?
        self.indicators['dwr_cell_adjoint'].interpolate(abs(self.indicators['dwr_cell_adjoint']))
        self.estimators['dwr_cell_adjoint'] = self.indicators['dwr_cell_adjoint'].vector().gather().sum()
        
    def get_dwr_flux_adjoint(self, sol, adjoint_sol):  # FIXME: Inputs are unused
        tpe = self.tp_enriched
        i = tpe.p0test
        lam = project(self.adjoint_solution, tpe.V)  # FIXME: prolong
        if not hasattr(self, 'error'):
            self.solve_high_order(adjoint=False)

        # Edge residual
        mass_term = i*tpe.p0trial*dx
        flux = -(lam*dot(tpe.u, tpe.n) + tpe.nu*dot(tpe.n, nabla_grad(lam)))
        dwr = flux*self.error
        if self.stab == 'SUPG':  # Account for stabilisation error
            dwr += flux*tpe.stabilisation*dot(tpe.u, grad(self.error))
        flux_terms = ((i*dwr)('+') + (i*dwr)('-'))*dS

        # Account for boundary conditions
        bcs = tpe.op.boundary_conditions
        for j in bcs.keys():
            if not 'value' in bcs[j]:
                flux_terms += i*dwr*ds(j)  # Robin BC in adjoint

        # Solve auxiliary FEM problem
        edge_res_adjoint = Function(tpe.P0)
        solve(mass_term == flux_terms, edge_res_adjoint)
        self.indicators['dwr_flux_adjoint'] = project(edge_res_adjoint, self.P0)  # FIXME: inject?
        self.indicators['dwr_flux_adjoint'].interpolate(abs(self.indicators['dwr_flux_adjoint']))
        self.estimators['dwr_flux_adjoint'] = self.indicators['dwr_flux_adjoint'].vector().gather().sum()
        
    def get_loseille_metric(self, adjoint=False, relax=True):
        adj = self.solution if adjoint else self.adjoint_solution
        sol = self.adjoint_solution if adjoint else self.solution
        adj_diff = Function(self.P1_vec).interpolate(abs(construct_gradient(adj)))
        adj = Function(self.P1).interpolate(abs(adj))

        # Get potential to take Hessian w.r.t.
        # x, y = SpatialCoordinate(self.mesh)
        if adjoint:
            source = self.kernel
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
        H1 = steady_metric(F1, mesh=self.mesh, noscale=True, op=self.op)
        H2 = steady_metric(F2, mesh=self.mesh, noscale=True, op=self.op)
        Hf = steady_metric(source, mesh=self.mesh, noscale=True, op=self.op)

        # Hessian for source term
        Mf = Function(self.P1_ten).assign(np.finfo(0.0).min)
        kernel = op2.Kernel(matscale_kernel(2),
                            "matscale",
                            cpp=True,
                            include_dirs=include_dir)
        op2.par_loop(kernel,
                     self.P1_ten.node_set,
                     Mf.dat(op2.RW),
                     Hf.dat(op2.READ),
                     adj.dat(op2.READ))

        # Form metric
        self.M = Function(self.P1_ten).assign(np.finfo(0.0).min)
        kernel = op2.Kernel(matscale_sum_kernel(2),
                            "matscale_sum",
                            cpp=True,
                            include_dirs=include_dir)
        op2.par_loop(kernel,
                     self.P1_ten.node_set,
                     self.M.dat(op2.RW),
                     H1.dat(op2.READ),
                     H2.dat(op2.READ),
                     adj_diff.dat(op2.READ))
        self.M = metric_relaxation(self.M, Mf) if relax else metric_intersection(self.M, Mf)
        self.M = steady_metric(None, H=self.M, op=self.op)

        # TODO: boundary contributions
        # bdy_contributions = i*(F1*n[0] + F2*n[1])*ds
        # n = self.n
        # Fhat = i*dot(phi, n)
        # bdy_contributions -= Fhat*ds(2) + Fhat*ds(3) + Fhat*ds(4)


class UnsteadyTracerProblem2d(UnsteadyProblem):
    def __init__(self):
        raise NotImplementedError  # TODO: Copy over most of the above
