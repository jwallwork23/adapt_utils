from firedrake import *
from firedrake_adjoint import *

import numpy as np

from adapt_utils.tracer.options import *
from adapt_utils.tracer.stabilisation import supg_coefficient, anisotropic_stabilisation
from adapt_utils.adapt.adaptation import *
from adapt_utils.adapt.metric import *
from adapt_utils.adapt.recovery import *
from adapt_utils.adapt.p0_metric import *
from adapt_utils.solver import SteadyProblem


__all__ = ["SteadyTracerProblem_CG"]


# TODO: Generalise to consider spaces other than CG
class SteadyTracerProblem_CG(SteadyProblem):
    r"""
    General continuous Galerkin solver object for stationary tracer advection problems of the form

..  math::
    \textbf{u} \cdot \nabla(\phi) - \nabla \cdot (\nu \cdot \nabla(\phi)) = f,

    for (prescribed) velocity :math:`\textbf{u}`, diffusivity :math:`\nu \geq 0`, source :math:`f`
    and (prognostic) concentration :math:`\phi`.

    Implemented boundary conditions:
        * Neumann zero;
        * Dirichlet zero;
        * outflow.
    """
    def __init__(self,
                 op,
                 mesh=None,
                 discrete_adjoint=False,
                 #discrete_adjoint=True,
                 finite_element=FiniteElement("Lagrange", triangle, 1),
                 prev_solution=None):
        super(SteadyTracerProblem_CG, self).__init__(mesh, op, finite_element, discrete_adjoint, None)

        # Extract parameters from Options class
        self.nu = op.set_diffusivity(self.P1)
        self.u = op.set_velocity(self.P1_vec)
        self.source = op.set_source(self.P1)
        self.kernel = op.set_objective_kernel(self.P0)
        self.gradient_field = self.nu  # arbitrary field to take gradient for discrete adjoint

        # Stabilisation
        if self.stab is None:
            self.stab = 'SUPG'
        assert self.stab in ('no', 'SU', 'SUPG')
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
            if bcs[i] == 'none':  # TODO: make consistent with Thetis
                a += -nu*psi*dot(n, nabla_grad(phi))*ds(i)
            if bcs[i] == 'dirichlet_zero':
                dbcs.append(i)
        L = f*psi*dx

        # Stabilisation
        if self.stab == "SU":
            a += self.stabilisation*dot(u, grad(psi))*dot(u, grad(phi))*dx
        elif self.stab == "SUPG":
            coeff = self.stabilisation*dot(u, grad(psi))
            a += coeff*dot(u, grad(phi))*dx
            a += coeff*-div(nu*grad(phi))*dx
            L += coeff*f*dx

        # Solve
        bc = DirichletBC(self.V, 0, dbcs)
        solve(a == L, self.solution, bcs=bc, solver_parameters=self.op.params)
        self.solution_file.write(self.solution)

    def solve_continuous_adjoint(self):
        u = self.u
        nu = self.nu
        n = self.n
        bcs = self.op.boundary_conditions
        dbcs = []
        lam = self.trial
        psi = self.test

        # Adjoint finite element problem
        a = lam*dot(u, grad(psi))*dx
        a += nu*inner(grad(lam), grad(psi))*dx
        for i in bcs.keys():
            if bcs[i] != 'neumann_zero':  # TODO: make consistent with Thetis
                dbcs.append(i)                              # Dirichlet BC in adjoint
            if bcs[i] == 'dirichlet_zero':
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
        bc = DirichletBC(self.V, 0, dbcs)
        solve(a == L, self.adjoint_solution, bcs=bc, solver_parameters=self.op.params)
        self.adjoint_solution_file.write(self.adjoint_solution)

    def solve_high_order(self, adjoint=True):
        """
        Solve the problem using linear and quadratic approximations on a refined mesh, take the
        difference and project back into the original space.
        """
        family = self.V.ufl_element().family()

        # Solve adjoint problem on fine mesh using quadratic elements
        tp_p2 = SteadyTracerProblem_CG(self.op,
                                       mesh=iso_P2(self.mesh),
                                       finite_element=FiniteElement(family, triangle, 2))
        sol_p1 = Function(tp_p2.P1)  # Project into P1 to get linear approximation, too
        if adjoint:
            tp_p2.solve_adjoint()
            sol_p1.project(tp_p2.adjoint_solution)
        else:
            tp_p2.solve()
            sol_p1.project(tp_p2.solution)

        # Evaluate difference on fine mesh and project onto coarse mesh
        sol_p2 = tp_p2.adjoint_solution if adjoint else tp_p2.solution
        sol = Function(tp_p2.V)
        sol.interpolate(sol_p2 - sol)
        with pyadjoint.stop_annotating():  # TODO: temp
            self.errorterm = Function(self.P2)
            self.errorterm.project(sol)
        return self.errorterm

    def get_hessian(self, adjoint=False):
        f = self.adjoint_solution if adjoint else self.solution
        return steady_metric(f, mesh=self.mesh, noscale=True, op=self.op)

    def get_hessian_metric(self, adjoint=False):
        self.M = steady_metric(self.adjoint_solution if adjoint else self.solution, op=self.op)

    def explicit_estimation(self, square=True):
        phi = self.solution
        i = self.p0test
        bcs = self.op.boundary_conditions

        # Compute residuals
        R = dot(self.u, grad(phi)) - div(self.nu*grad(phi))
        r = phi*dot(self.u, self.n) - self.nu*dot(self.n, nabla_grad(phi))

        # Assemble cell residual
        self.cell_res = assemble(i*R*R*dx) if square else assemble(i*R*dx)
        if not square:
            self.p1cell_res = interpolate(R, self.P1)  # FIXME: clarify difference with above

        # Solve auxiliary problem to assemble edge residual
        mass_term = i*self.p0trial*dx
        flux_terms = ((i*r*r)('+') + (i*r*r)('-'))*dS if square else ((i*r)('+') + (i*r)('-'))*dS
        for j in bcs.keys():
            if bcs[j] == 'neumann_zero':
                flux_terms += i*r*r*ds(j) if square else i*r*ds(j)
        self.edge_res = Function(self.P0)
        solve(mass_term == flux_terms, self.edge_res)

        # Form error estimator
        if square:
            self.p0indicator = project(sqrt(self.h*self.h*self.cell_res + 0.5*self.h*self.edge_res), self.P0)
            self.p1indicator = project(sqrt(self.h*self.h*self.cell_res_adjoint + 0.5*self.h*self.edge_res_adjoint), self.P1)
        else:
            self.p0indicator = project(self.cell_res + self.edge_res, self.P0)
            self.p1indicator = project(self.cell_res + self.edge_res, self.P1)
        self.p0indicator.interpolate(abs(self.p0indicator))
        self.p1indicator.interpolate(abs(self.p1indicator))
        self.p0indicator.rename('explicit')
        self.p1indicator.rename('explicit')

    def explicit_estimation_adjoint(self, square=True):
        lam = self.adjoint_solution
        u = self.u
        nu = self.nu
        n = self.n
        i = self.p0test
        bcs = self.op.boundary_conditions

        # Cell residual
        R = -div(u*lam) - div(nu*grad(lam))
        self.cell_res_adjoint = assemble(i*R*R*dx) if square else assemble(i*R*dx)
        if not square:
            self.p1cell_res_adjoint = interpolate(R, self.P1)

        # Edge residual
        mass_term = i*self.p0trial*dx
        r = - lam*dot(u, n) - nu*dot(n, nabla_grad(lam))
        flux_terms = ((i*r*r)('+') + (i*r*r)('-'))*dS if square else ((i*r)('+') + (i*r)('-'))*dS
        for j in bcs.keys():
            if bcs[j] == 'dirichlet_zero':
                flux_terms += i*r*r*ds(j) if square else i*r*ds(j)  # Robin BC in adjoint
        self.edge_res_adjoint = Function(self.P0)
        solve(mass_term == flux_terms, self.edge_res_adjoint)

        # Form error estimator
        if square:
            self.p0indicator = project(sqrt(self.h*self.h*self.cell_res_adjoint + 0.5*self.h*self.edge_res_adjoint), self.P0)
            self.p1indicator = project(sqrt(self.h*self.h*self.cell_res_adjoint + 0.5*self.h*self.edge_res_adjoint), self.P1)
        else:
            self.p0indicator = project(self.cell_res_adjoint + self.edge_res_adjoint, self.P0)
            self.p1indicator = project(self.cell_res_adjoint + self.edge_res_adjoint, self.P1)
        self.p0indicator.interpolate(abs(self.p0indicator))
        self.p1indicator.interpolate(abs(self.p1indicator))
        self.p0indicator.rename('explicit_adjoint')
        self.p1indicator.rename('explicit_adjoint')

    def dwr_indication_(self):  # TODO: check
        i = self.p0test
        u = self.u
        nu = self.nu
        n = self.n
        f = self.source
        bcs = self.op.boundary_conditions
        phi = self.solution
        if self.op.order_increase:
            lam = self.solve_high_order(adjoint=True) if not hasattr(self, 'errorterm') else self.errorterm
        else:
            self.adjoint_solution

        # Finite element problem
        a = i*lam*dot(u, grad(phi))*dx
        a += i*nu*inner(grad(phi), grad(lam))*dx
        for j in bcs.keys():
            if bcs[j] == 'none':  # TODO: make consistent with Thetis
                a += -i*nu*lam*dot(n, nabla_grad(phi))*ds(j)
        L = i*f*lam*dx

        # Stabilisation
        if self.stab == "SU":
            a += i*self.stabilisation*dot(u, grad(lam))*dot(u, grad(phi))*dx
        elif self.stab == "SUPG":
            coeff = self.stabilisation*dot(u, grad(lam))
            a += i*coeff*dot(u, grad(phi))*dx
            a += i*coeff*-div(nu*grad(phi))*dx
            L += i*coeff*f*dx

        # Get values on each element
        mass_term = i*self.p0trial*dx
        self.p0indicator = Function(self.P0)
        solve(mass_term == L-a, self.p0indicator)
        self.p0indicator.interpolate(abs(self.p0indicator))
 
    def dwr_estimation(self):
        u = self.u
        nu = self.nu
        n = self.n
        f = self.source
        bcs = self.op.boundary_conditions
        phi = self.solution
        lam = self.solve_high_order(adjoint=True) if not hasattr(self, 'errorterm') else self.errorterm

        # Finite element problem
        a = lam*dot(u, grad(phi))*dx
        a += nu*inner(grad(phi), grad(lam))*dx
        for j in bcs.keys():
            if bcs[j] == 'none':  # TODO: make consistent with Thetis
                a += -nu*lam*dot(n, nabla_grad(phi))*ds(j)
        L = f*lam*dx

        # Stabilisation
        if self.stab == "SU":
            a += self.stabilisation*dot(u, grad(lam))*dot(u, grad(phi))*dx
        elif self.stab == "SUPG":
            coeff = self.stabilisation*dot(u, grad(lam))
            a += coeff*dot(u, grad(phi))*dx
            a += coeff*-div(nu*grad(phi))*dx
            L += coeff*f*dx

        # Evaluate error estimator
        self.estimator = assemble(L-a)
        return self.estimator

    def dwr_estimation_adjoint(self):
        u = self.u
        nu = self.nu
        n = self.n
        bcs = self.op.boundary_conditions
        lam = self.adjoint_solution
        phi = self.solve_high_order(adjoint=False) if not hasattr(self, 'errorterm') else self.errorterm

        # Adjoint finite element problem
        a = lam*dot(u, grad(phi))*dx
        a += nu*inner(grad(lam), grad(phi))*dx
        for i in bcs.keys():
            if bcs[i] == 'dirichlet_zero':
                a += -lam*phi*(dot(u, n))*ds(i)
                a += -nu*phi*dot(n, nabla_grad(lam))*ds(i)  # Robin BC in adjoint
        L = self.kernel*phi*dx

        # Stabilisation
        if self.stab == 'SU':
            a += self.stabilisation*div(u*phi)*div(u*lam)*dx
        elif self.stab == "SUPG":  # NOTE: this is not equivalent to discrete adjoint
            coeff = -self.stabilisation*div(u*phi)
            a += coeff*-div(u*lam)*dx
            a += coeff*-div(nu*grad(lam))*dx
            L += coeff*self.kernel*dx

        # Evaluate error estimator
        self.estimator = assemble(L-a)
        return self.estimator

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

        # Residual
        R = (f - dot(u, grad(phi)) + div(nu*grad(phi)))*lam

        # Flux terms (arising from integration by parts)
        mass_term = i*self.p0trial*dx
        flux = -nu*lam*dot(n, nabla_grad(phi))
        flux_terms = ((i*flux)('+') + (i*flux)('-'))*dS
        for j in bcs.keys():
            if bcs[j] == 'neumann_zero':
                flux_terms += i*flux*ds(j)
            # NOTE: For CG methods, Dirichlet error is zero, by construction
        self.edge_res = Function(self.P0)
        solve(mass_term == flux_terms, self.edge_res)

        # Account for stabilisation error
        if self.op.order_increase and self.stab == 'SUPG':
            R -= (f - dot(u, grad(phi)) + div(nu*grad(phi)))*self.stabilisation*dot(u, grad(self.adjoint_solution))

        # Sum
        self.cell_res = assemble(i*R*dx)
        if self.op.dwr_approach == 'error_representation':
            self.p0indicator = project(self.cell_res + self.edge_res, self.P0)
            #self.p1indicator = project(R + self.edge_res, self.P1)
            self.p1indicator = project(self.cell_res + self.edge_res, self.P1)
        elif self.op.dwr_approach == 'ainsworth_oden':
            self.p0indicator = project(self.h*self.h*R + self.h*self.edge_res, P0)
            self.p1indicator = project(self.h*self.h*R + self.h*self.edge_res, P1)
        else:
            raise NotImplementedError
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

        # Adjoint source term
        dJdphi = self.op.box(self.P0)

        # Cell residual
        R = (dJdphi + div(u*lam) + div(nu*grad(lam)))*phi

        # Edge residual
        mass_term = i*self.p0trial*dx
        flux = -(lam*dot(u, n) + nu*dot(n, nabla_grad(lam)))*phi
        flux_terms = ((i*flux)('+') + (i*flux)('-')) * dS
        for j in bcs.keys():
            if bcs[j] == 'neumann_zero':  # TODO: check
                flux_terms += i*flux*ds(j)  # Robin BC in adjoint
        self.edge_res_adjoint = Function(self.P0)
        solve(mass_term == flux_terms, self.edge_res_adjoint)

        ## Account for stabilisation error
        #if self.op.order_increase and self.stab == 'SUPG':
        #    R -= -(dJdphi + div(u*lam) + div(nu*grad(lam)))*self.stabilisation*dot(u, grad(self.adjoint_solution))

        # Sum
        self.cell_res_adjoint = assemble(i*R*dx)
        if self.op.dwr_approach == 'error_representation':
            self.p0indicator = project(self.cell_res_adjoint + self.edge_res_adjoint, self.P0)
            #self.p1indicator = project(R + self.edge_res_adjoint, self.P1)
            self.p1indicator = project(self.cell_res_adjoint + self.edge_res_adjoint, self.P1)
        elif self.op.dwr_approach == 'ainsworth_oden':
            self.p0indicator = project(self.h*self.h*R + self.h*self.edge_res_adjoint, P0)
            self.p1indicator = project(self.h*self.h*R + self.h*self.edge_res_adjoint, P1)
        else:
            raise NotImplementedError
        self.p0indicator.interpolate(abs(self.p0indicator))
        self.p1indicator.interpolate(abs(self.p1indicator))
        self.p0indicator.rename('dwr_adjoint')
        self.p1indicator.rename('dwr_adjoint')
        
    def get_anisotropic_metric(self, adjoint=False, relax=False, superpose=True):
        assert not (relax and superpose)

        # Solve adjoint problem
        if self.op.order_increase:
            adj = self.solve_high_order(adjoint=not adjoint)
        else:
            adj = self.solution if adjoint else self.adjoint_solution
        sol = self.adjoint_solution if adjoint else self.solution
        adj_diff = Function(self.P1_vec).interpolate(abs(construct_gradient(adj)))
        adj = Function(self.P1).interpolate(abs(adj))

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
        H1 = steady_metric(F1, mesh=self.mesh, noscale=True, op=self.op)
        H2 = steady_metric(F2, mesh=self.mesh, noscale=True, op=self.op)
        Hf = steady_metric(source, mesh=self.mesh, noscale=True, op=self.op)

        # form metric  # TODO: use pyop2
        self.M = Function(self.P1_ten)
        for i in range(self.mesh.num_vertices()):
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

    def custom_adapt(self):  # TODO: also consider seperate forward / adjoint versions
        if self.approach == 'carpio':
            H = self.get_hessian(adjoint=False)
            H_adj = self.get_hessian(adjoint=True)
            M = metric_intersection(H, H_adj)
            self.dwr_indication()
            i = self.p1indicator.copy()
            self.dwr_indication_adjoint()
            self.p1indicator.interpolate(i + self.p1indicator)
            amd = AnisotropicMetricDriver(self.mesh, hessian=M, indicator=self.p1indicator, op=self.op)
            amd.get_anisotropic_metric()
            self.M = amd.p1metric
