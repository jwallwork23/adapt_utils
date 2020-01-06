from firedrake import *

import numpy as np

from adapt_utils.tracer.options import *
from adapt_utils.tracer.stabilisation import supg_coefficient, anisotropic_stabilisation
from adapt_utils.adapt.adaptation import *
from adapt_utils.adapt.metric import *
from adapt_utils.adapt.kernels import matscale_component_kernel
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
                 discrete_adjoint=False,
                 finite_element=FiniteElement("Lagrange", triangle, 1),
                 prev_solution=None):
        super(SteadyTracerProblem2d, self).__init__(mesh, op, finite_element, discrete_adjoint, None, 1)

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
        for i in bcs.keys():
            if bcs[i] == {}:
                a += -nu*psi*dot(n, nabla_grad(phi))*ds(i)
            if 'diff_flux' in bcs[i]:
                a += -psi*dot(n, bcs[i]['diff_flux'])*ds(i)
            if 'value' in bcs[i]:
                dbcs.append(DirichletBC(self.V, bcs[i]['value'], i))
        L = f*psi*dx

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
        for i in bcs.keys():
            if not 'diff_flux' in bcs[i]:
                dbcs.append(DirichletBC(self.V, 0, i))  # Dirichlet BC in adjoint
            if 'value' in bcs[i]:
                a += -lam*psi*(dot(u, n))*ds(i)
                a += -nu*psi*dot(n, nabla_grad(lam))*ds(i)  # Robin BC in adjoint
        L = self.kernel*psi*dx

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
        family = self.V.ufl_element().family()

        # Solve adjoint problem on fine mesh using quadratic elements
        tp_p2 = SteadyTracerProblem2d(self.op,
                                      mesh=self.am.hierarchy[1],
                                      finite_element=FiniteElement(family, triangle, 2))
        if adjoint:
            tp_p2.solve_adjoint()
        else:
            tp_p2.solve()
        sol_p2 = tp_p2.adjoint_solution if adjoint else tp_p2.solution

        # Project into P1 to get linear approximation, too
        sol = self.adjoint_solution if adjoint else self.solution
        sol_p1 = Function(tp_p2.P1)
        prolong(sol, sol_p1)

        # Evaluate difference in enriched space and inject onto coarse mesh
        sol = Function(sol_p2)
        sol -= sol_p1
        self.errorterm = Function(self.P2)
        try:
            inject(sol, self.errorterm)
        except:
            self.errorterm.project(sol)
        return self.errorterm  # FIXME: Should keep errorterm in h.o. space as long as possible

    def get_hessian(self, adjoint=False):
        f = self.adjoint_solution if adjoint else self.solution
        return steady_metric(f, mesh=self.mesh, noscale=True, op=self.op)

    def get_hessian_metric(self, adjoint=False):
        self.M = steady_metric(self.adjoint_solution if adjoint else self.solution, op=self.op)

    def explicit_indication(self):
        phi = self.solution
        i = self.p0test
        bcs = self.op.boundary_conditions

        # Compute residuals
        R = dot(self.u, grad(phi)) - div(self.nu*grad(phi))
        r = phi*dot(self.u, self.n) - self.nu*dot(self.n, nabla_grad(phi))

        # For use in anisotropic methods
        self.cell_res = assemble(i*R*dx)
        self.p1cell_res = interpolate(R, self.P1)

        # Solve auxiliary problem to assemble edge residual
        mass_term = i*self.p0trial*dx
        flux_terms = ((i*r)('+') + (i*r)('-'))*dS
        for j in bcs.keys():
            if 'diff_flux' in bcs[j]:
                assert np.allclose(bcs[j]['diff_flux'], 0.0)  # TODO: Nonzero case
                flux_terms += i*r*ds(j)
        self.edge_res = Function(self.P0)
        solve(mass_term == flux_terms, self.edge_res)

        # Form error estimator
        self.p0indicator = Function(self.P0)
        self.p0indicator += self.cell_res + self.edge_res
        self.p1indicator = project(self.p0indicator, self.P1)
        self.p0indicator.interpolate(abs(self.p0indicator))
        self.p1indicator.interpolate(abs(self.p1indicator))
        self.p0indicator.rename('explicit')
        self.p1indicator.rename('explicit')

    def explicit_indication_adjoint(self):
        lam = self.adjoint_solution
        u = self.u
        nu = self.nu
        n = self.n
        i = self.p0test
        bcs = self.op.boundary_conditions

        # Cell residual
        R = -div(u*lam) - div(nu*grad(lam))
        self.cell_res_adjoint = assemble(i*R*dx)

        # For use in anisotropic approaches
        self.p1cell_res_adjoint = interpolate(R, self.P1)

        # Edge residual
        mass_term = i*self.p0trial*dx
        r = -lam*phi*dot(u, n) - nu*dot(n, nabla_grad(lam))
        flux_terms = ((i*r)('+') + (i*r)('-'))*dS
        for j in bcs.keys():
            if not 'value' in bcs[j]:
                flux_terms += i*r*ds(j)  # Robin BC in adjoint
        self.edge_res_adjoint = Function(self.P0)
        solve(mass_term == flux_terms, self.edge_res_adjoint)

        # Form error estimator
        self.p0indicator = Function(self.P0)
        self.p0indicator += self.cell_res_adjoint + self.edge_res_adjoint
        self.p1indicator = project(self.p0indicator, self.P1)
        self.p0indicator.interpolate(abs(self.p0indicator))
        self.p1indicator.interpolate(abs(self.p1indicator))
        self.p0indicator.rename('explicit_adjoint')
        self.p1indicator.rename('explicit_adjoint')

    def dwr_indication(self):
        i = self.p0test
        phi = self.solution
        u = self.u
        nu = self.nu
        n = self.n
        f = self.source
        bcs = self.op.boundary_conditions
        if self.op.order_increase:
            lam = self.solve_high_order(adjoint=True) if not hasattr(self, 'errorterm') else self.errorterm
        else:
            lam = self.adjoint_solution

        # Cell residual
        R = (f - dot(u, grad(phi)) + div(nu*grad(phi)))*lam
        self.cell_res = assemble(i*R*dx)

        # Flux terms (arising from integration by parts)
        mass_term = i*self.p0trial*dx
        flux = -nu*lam*dot(n, nabla_grad(phi))
        flux_terms = ((i*flux)('+') + (i*flux)('-'))*dS
        for j in bcs.keys():
            if 'diff_flux' in bcs[j]:  # TODO: Non-zero case
                flux_terms += i*flux*ds(j)
            # NOTE: For CG methods, Dirichlet error is zero, by construction
        self.edge_res = Function(self.P0)
        solve(mass_term == flux_terms, self.edge_res)

        # Account for stabilisation error
        if self.op.order_increase and self.stab == 'SUPG':
            R += (f - dot(u, grad(phi)) + div(nu*grad(phi)))*self.stabilisation*dot(u, grad(self.adjoint_solution))

        # Sum
        self.p0indicator = Function(self.cell_res)
        self.p0indicator += self.edge_res
        self.p1indicator = project(self.p0indicator, self.P1)
        self.p0indicator.interpolate(abs(self.p0indicator))
        self.p1indicator.interpolate(abs(self.p1indicator))
        self.p0indicator.rename('dwr')
        self.p1indicator.rename('dwr')

    def dwr_indication_adjoint(self):
        i = self.p0test
        lam = self.adjoint_solution
        u = self.u
        nu = self.nu
        n = self.n
        bcs = self.op.boundary_conditions
        phi = self.solve_high_order(adjoint=False) if self.op.order_increase else self.solution

        # Cell residual
        dJdphi = self.op.box(self.P0)  # Adjoint source term
        R = (dJdphi + div(u*lam) + div(nu*grad(lam)))*phi

        # Edge residual
        mass_term = i*self.p0trial*dx
        flux = -(lam*dot(u, n) + nu*dot(n, nabla_grad(lam)))*phi
        flux_terms = ((i*flux)('+') + (i*flux)('-'))*dS
        for j in bcs.keys():
            if 'diff_flux' in bcs[j]:  # TODO: Non-zero case
                flux_terms += i*flux*ds(j)  # Robin BC in adjoint
        self.edge_res_adjoint = Function(self.P0)
        solve(mass_term == flux_terms, self.edge_res_adjoint)

        # Account for stabilisation error
        if self.op.order_increase and self.stab == 'SUPG':
            R += (dJdphi + div(u*lam) + div(nu*grad(lam)))*self.stabilisation*dot(u, grad(self.adjoint_solution))

        # Sum
        self.cell_res_adjoint = assemble(i*R*dx)
        self.p0indicator = Function(self.P0)
        self.p0indicator += self.cell_res_adjoint + self.edge_res_adjoint
        self.p1indicator = project(self.p0indicator, self.P1)
        self.p0indicator.interpolate(abs(self.p0indicator))
        self.p1indicator.interpolate(abs(self.p1indicator))
        self.p0indicator.rename('dwr_adjoint')
        self.p1indicator.rename('dwr_adjoint')
        
    def get_loseille_metric(self, adjoint=False, relax=True, superpose=False):
        assert (relax or superpose) and not (relax and superpose)

        # Solve adjoint problem
        if self.op.order_increase:
            adj = self.solve_high_order(adjoint=not adjoint)
        else:
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
    raise NotImplementedError  # TODO: Copy over most of the above
