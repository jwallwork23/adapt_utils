from firedrake import *

import numpy as np

from adapt_utils.tracer.options import *
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
        if self.stab in ('SU', 'SUPG'):
            #self.stabilisation = supg_coefficient(self.u, self.nu, mesh=self.mesh, anisotropic=True)
            self.stabilisation = supg_coefficient(self.u, self.nu, mesh=self.mesh, anisotropic=False)
            #self.stabilisation = anisotropic_stabilisation(self.u, mesh=self.mesh)

        # Rename solution fields
        self.solution.rename('Tracer concentration')
        self.adjoint_solution.rename('Adjoint tracer concentration')

        # Classification
        self.nonlinear = False

    def solve(self):
        u = self.u
        nu = self.nu
        n = self.n
        f = self.source
        bcs = self.op.boundary_conditions
        dbcs = []
        phi = self.trial
        psi = self.test

        # Finite element problem
        a = psi*dot(u, grad(phi))*dx
        a += nu*inner(grad(phi), grad(psi))*dx
        L = f*psi*dx
        for i in bcs.keys():
            if bcs[i] == {}:
                a += -nu*psi*dot(n, nabla_grad(phi))*ds(i)
            if 'diff_flux' in bcs[i]:
                a += -nu*psi*dot(n, nabla_grad(phi))*ds(i)
                L += -psi*bcs[i]['diff_flux']*ds(i)
            if 'value' in bcs[i]:
                dbcs.append(DirichletBC(self.V, bcs[i]['value'], i))

        # Stabilisation
        if self.stab == "SU":
            a += self.stabilisation*dot(u, grad(psi))*dot(u, grad(phi))*dx
        elif self.stab == "SUPG":
            coeff = self.stabilisation*dot(u, grad(psi))
            a += coeff*dot(u, grad(phi))*dx
            a += coeff*-div(nu*grad(phi))*dx
            L += coeff*f*dx

        # For condition number studies  # TODO: what about RHS?
        self.lhs = a

        # Solve
        solve(a == L, self.solution, bcs=dbcs, solver_parameters=self.op.params)
        self.solution_file.write(self.solution)

    def solve_continuous_adjoint(self):
        u = self.u
        nu = self.nu
        n = self.n
        bcs = self.op.boundary_conditions
        dbcs = []
        lam = self.trial
        psi = self.test

        # Adjoint finite element problem  # TODO: Check below for non-zero BCs
        a = lam*dot(u, grad(psi))*dx
        a += nu*inner(grad(lam), grad(psi))*dx
        L = self.kernel*psi*dx
        for i in bcs.keys():
            if not 'diff_flux' in bcs[i]:
                dbcs.append(DirichletBC(self.V, 0, i))  # Dirichlet BC in adjoint
            if 'value' in bcs[i]:
                a += -lam*psi*(dot(u, n))*ds(i)
                a += -nu*psi*dot(n, nabla_grad(lam))*ds(i)  # Robin BC in adjoint

        # Stabilisation
        if self.stab == 'SU':
            a += self.stabilisation*div(u*psi)*div(u*lam)*dx
        elif self.stab == "SUPG":  # NOTE: this is not equivalent to discrete adjoint
            coeff = -self.stabilisation*div(u*psi)
            a += coeff*-div(u*lam)*dx
            a += coeff*-div(nu*grad(lam))*dx
            L += coeff*self.kernel*dx

        # Solve
        solve(a == L, self.adjoint_solution, bcs=dbcs, solver_parameters=self.op.params)
        self.adjoint_solution_file.write(self.adjoint_solution)

    def solve_high_order(self, adjoint=True):
        """
        Solve the problem using linear and quadratic approximations on a refined mesh, take the
        difference and project back into the original space.
        """
        sol = self.adjoint_solution if adjoint else self.solution

        # Solve adjoint problem on fine mesh using quadratic elements
        if adjoint:
            self.tp_enriched.solve_adjoint()
            sol_p2 = self.tp_enriched.adjoint_solution
        else:
            self.tp_enriched.solve()
            sol_p2 = self.tp_enriched.solution

        # Project into P1 to get linear approximation, too
        sol_p1 = Function(self.tp_enriched.P1)
        # prolong(sol, sol_p1)  # FIXME: Maybe the hierarchy isn't recognised?
        sol_p1.project(sol)

        # Evaluate difference in enriched space
        if adjoint:
            self.adjoint_error = interpolate(sol_p2 - sol_p1, self.tp_enriched.P2)
        else:
            self.error = interpolate(sol_p2 - sol_p1, self.tp_enriched.P2)

    def get_hessian(self, adjoint=False):
        f = self.adjoint_solution if adjoint else self.solution
        return steady_metric(f, mesh=self.mesh, noscale=True, op=self.op)

    def get_hessian_metric(self, adjoint=False):
        self.M = steady_metric(self.adjoint_solution if adjoint else self.solution, op=self.op)

    def get_strong_residual(self):
        phi = self.solution

        # Compute residuals
        R = self.source - dot(self.u, grad(phi)) + div(self.nu*grad(phi))
        self.indicators['cell_res_forward'] = assemble(self.p0test*abs(R)*dx)

        # For use in anisotropic methods
        self.indicator = interpolate(self.indicators['cell_res_forward'], self.P1)
        self.indicator.rename('forward strong residual')

    def get_strong_residual_adjoint(self):
        lam = self.adjoint_solution

        # Cell residual
        R = self.kernel + div(self.u*lam) + div(self.nu*grad(lam))
        self.indicators['cell_res_adjoint'] = assemble(self.p0test*abs(R)*dx)

        # For use in anisotropic approaches
        self.indicator = interpolate(self.indicators['cell_res_adjoint'], self.P1)
        self.indicator.rename('adjoint strong residual')

    def dwr_indication(self):
        tpe = self.tp_enriched
        i = tpe.p0test
        phi = project(self.solution, tpe.V)  # FIXME: prolong
        u = tpe.u
        nu = tpe.nu
        n = tpe.n
        f = tpe.source
        bcs = tpe.op.boundary_conditions
        if not hasattr(self, 'adjoint_error'):
            self.solve_high_order(adjoint=True)
        lam = self.adjoint_error

        # Cell residual
        R = (f - dot(u, grad(phi)) + div(nu*grad(phi)))*lam
        if self.stab == 'SUPG':  # Account for stabilisation error
            R += (f - dot(u, grad(phi)) + div(nu*grad(phi)))*tpe.stabilisation*dot(u, grad(lam))
        self.cell_res = project(assemble(i*R*dx), self.P0)  # FIXME: inject

        # Flux terms (arising from integration by parts)
        mass_term = i*tpe.p0trial*dx
        flux = -nu*lam*dot(n, nabla_grad(phi))
        flux_terms = ((i*flux)('+') + (i*flux)('-'))*dS
        for j in bcs.keys():
            if 'diff_flux' in bcs[j]:  # TODO: Non-zero case
                flux_terms += i*flux*ds(j)
            # NOTE: For CG methods, Dirichlet error is zero, by construction
        edge_res = Function(tpe.P0)
        solve(mass_term == flux_terms, edge_res)
        self.edge_res = project(edge_res, self.P0)  # FIXME: inject

        # Sum
        self.p0indicator = Function(self.cell_res)
        self.p0indicator += self.edge_res
        self.indicator = project(self.p0indicator, self.P1)
        self.p0indicator.interpolate(abs(self.p0indicator))
        self.indicator.interpolate(abs(self.indicator))
        self.p0indicator.rename('dwr')
        self.indicator.rename('dwr')

    def dwr_indication_adjoint(self):
        tpe = self.tp_enriched
        i = tpe.p0test
        lam = project(self.adjoint_solution, tpe.V)  # FIXME: prolong
        u = tpe.u
        nu = tpe.nu
        n = tpe.n
        bcs = tpe.op.boundary_conditions
        if not hasattr(self, 'error'):
            self.solve_high_order(adjoint=False)
        phi = self.error

        # Cell residual
        dJdphi = tpe.op.box(tpe.P0)  # Adjoint source term
        R = (dJdphi + div(u*lam) + div(nu*grad(lam)))*phi
        if self.stab == 'SUPG':  # Account for stabilisation error
            R += (dJdphi + div(u*lam) + div(nu*grad(lam)))*tpe.stabilisation*dot(u, grad(phi))
        self.cell_res_adjoint = project(assemble(i*R*dx), self.P0)  # FIXME: inject

        # Edge residual
        mass_term = i*tpe.p0trial*dx
        flux = -(lam*dot(u, n) + nu*dot(n, nabla_grad(lam)))*phi
        flux_terms = ((i*flux)('+') + (i*flux)('-'))*dS
        for j in bcs.keys():
            if 'diff_flux' in bcs[j]:  # TODO: Non-zero case
                flux_terms += i*flux*ds(j)  # Robin BC in adjoint
        edge_res_adjoint = Function(tpe.P0)
        solve(mass_term == flux_terms, edge_res_adjoint)
        self.edge_res_adjoint = project(edge_res_adjoint, self.P0)  # FIXME: inject

        # Sum
        self.p0indicator = Function(self.P0)
        self.p0indicator += self.cell_res_adjoint + self.edge_res_adjoint
        self.indicator = project(self.p0indicator, self.P1)
        self.p0indicator.interpolate(abs(self.p0indicator))
        self.indicator.interpolate(abs(self.indicator))
        self.p0indicator.rename('dwr_adjoint')
        self.indicator.rename('dwr_adjoint')
        
    def get_loseille_metric(self, adjoint=False, relax=True, superpose=False):
        assert (relax or superpose) and not (relax and superpose)

        # Solve adjoint problem
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
        if relax:
            self.M = metric_relaxation(self.M, Mf)
        elif superpose:
            self.M = metric_intersection(self.M, Mf)
        self.M = steady_metric(None, H=self.M, op=self.op)

        # TODO: boundary contributions
        # bdy_contributions = i*(F1*n[0] + F2*n[1])*ds
        # n = self.n
        # Fhat = i*dot(phi, n)
        # bdy_contributions -= Fhat*ds(2) + Fhat*ds(3) + Fhat*ds(4)


class UnsteadyTracerProblem2d(UnsteadyProblem):
    def __init__(self):
        raise NotImplementedError  # TODO: Copy over most of the above
