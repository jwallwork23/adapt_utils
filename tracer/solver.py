from firedrake import *
from firedrake_adjoint import *

from time import clock
import numpy as np

from adapt_utils.tracer.options import PowerOptions
from adapt_utils.adapt.metric import *
from adapt_utils.solver import SteadyProblem


__all__ = ["SteadyTracerProblem"]


class SteadyTracerProblem(SteadyProblem):
    """
    General solver object for stationary tracer advection problems.
    """
    def __init__(self,
                 op=PowerOptions(),
                 stab=None,
                 mesh=SquareMesh(40, 40, 4, 4),
                 approach='fixed_mesh',
                 discrete_adjoint=False,
                 finite_element=FiniteElement("Lagrange", triangle, 1),
                 high_order=False,
                 prev_solution=None):
        super(SteadyTracerProblem, self).__init__(mesh,
                                                  finite_element,
                                                  approach,
                                                  stab,
                                                  discrete_adjoint,
                                                  op,
                                                  high_order,
                                                  None)
        assert(finite_element.family() == 'Lagrange')  # TODO: DG option

        # Extract parameters from Options class
        self.nu = op.set_diffusivity(self.P1)
        self.u = op.set_velocity(self.P1_vec)
        self.source = op.set_source(self.P0)
        self.kernel = op.set_objective_kernel(self.P0)
        self.gradient_field = self.nu  # arbitrary field to take gradient for discrete adjoint

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

        # Finite element problem
        phi = TrialFunction(self.V)
        psi = TestFunction(self.V)
        a = psi*dot(u, grad(phi))*dx
        a += nu*inner(grad(phi), grad(psi))*dx
        for i in bcs.keys():
            if bcs[i] != 'neumann_zero':
                a += - nu*psi*dot(n, nabla_grad(phi))*ds(i)
            if bcs[i] == 'dirichlet_zero':
                dbcs.append(i)
        L = f*psi*dx

        # Stabilisation
        if self.stab in ("SU", "SUPG"):
            tau = self.h / (2*sqrt(inner(u, u)))
            stab_coeff = tau * dot(u, grad(psi))
            R_a = dot(u, grad(phi))         # LHS component of strong residual
            if self.stab == 'SUPG':
                R_a += - div(nu*grad(phi))
                R_L = f                     # RHS component of strong residual
                L += stab_coeff*R_L*dx
            a += stab_coeff*R_a*dx

        # Solve
        bc = DirichletBC(self.V, 0, dbcs)
        solve(a == L, self.solution, bcs=bc, solver_parameters=self.op.params)
        #self.sol_file.write(self.solution)

    def solve_continuous_adjoint(self):
        u = self.u
        nu = self.nu
        n = self.n
        bcs = self.op.boundary_conditions
        dbcs = []

        # Adjoint source term
        lam = TrialFunction(self.V)
        psi = TestFunction(self.V)

        # Adjoint finite element problem
        a = lam*dot(u, grad(psi))*dx
        a += nu*inner(grad(lam), grad(psi))*dx
        for i in bcs.keys():
            if bcs[i] != 'neumann_zero':
                dbcs.append(i)                              # Dirichlet BC in adjoint
            if bcs[i] != 'dirichlet_zero':
                a += -lam*psi*(dot(u, n))*ds(i)
                a += -nu*psi*dot(n, nabla_grad(lam))*ds(i)  # Robin BC in adjoint
        L = self.kernel*psi*dx

        # Stabilisation
        if self.stab in ("SU", "SUPG"):
            tau = self.h / (2*sqrt(inner(u, u)))
            stab_coeff = tau * -div(u*psi)
            R_a = -div(u*lam)             # LHS component of strong residual
            if self.stab == "SUPG":
                R_a += -div(nu*grad(lam))
                R_L = self.kernel         # RHS component of strong residual
                L += stab_coeff*R_L*dx
            a += stab_coeff*R_a*dx

        bc = DirichletBC(self.V, 0, dbcs)
        solve(a == L, self.adjoint_solution, bcs=bc, solver_parameters=self.op.params)
        #self.sol_adjoint_file.write(self.adjoint_solution)

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
        r_norm = Function(self.P0)
        solve(mass_term == flux_terms, r_norm)

        # Form error estimator
        self.indicator = project(sqrt(self.h*self.h*R_norm + 0.5*self.h*r_norm), self.P0)
        self.indicator.rename('explicit_adjoint')
 
    def solve_high_order(self, adjoint=True):  # TODO: reimplement in base class

        # Consider an iso-P2 refined mesh
        fine_mesh = iso_P2(self.mesh)

        # Solve adjoint problem on fine mesh using linear elements
        tp_p1 = SteadyTracerProblem(stab=self.stab,
                              mesh=fine_mesh,
                              fe=FiniteElement('Lagrange', triangle, 1))
        if adjoint:
            tp_p1.setup_adjoint_equation()
            tp_p1.solve_adjoint()
        else:
            tp_p1.setup_equation()
            tp_p1.solve()

        # Solve adjoint problem on fine mesh using quadratic elements
        tp_p2 = SteadyTracerProblem(stab=self.stab,
                                    mesh=fine_mesh,
                                    fe=FiniteElement('Lagrange', triangle, 2))
        if adjoint:
            tp_p2.setup_adjoint_equation()
            tp_p2.solve_adjoint()
        else:
            tp_p2.setup_equation()
            tp_p2.solve()

        # Evaluate difference on fine mesh and project onto coarse mesh
        sol_p1 = tp_p1.sol_adjoint if adjoint else tp_p1.sol
        sol_p2 = tp_p2.sol_adjoint if adjoint else tp_p2.sol
        sol = Function(tp_p2.V).interpolate(sol_p1)
        sol.interpolate(sol_p2 - sol)
        coarse = Function(self.V)
        coarse.project(sol)
        return coarse

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
        dJdphi = interpolate(self.op.box(self.mesh), self.P0)
            
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
        r = Function(self.P0)
        solve(mass_term == flux_terms, r)

        self.cell_res_adjoint = R
        self.edge_res_adjoint = r
        self.indicator = project(R + r, self.P0)
        self.indicator.rename('dwr_adjoint')
        
    def get_anisotropic_metric(self, adjoint=False, relax=False, superpose=True):
        assert not (relax and superpose)
        try:
            assert self.op.restrict == 'anisotropy'
        except:
            self.op.restrict = 'anisotropy'
            raise Warning("Setting metric restriction method to 'anisotropy'")

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
            source = interpolate(self.op.box(self.mesh), self.P0)
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
                self.M.dat.data[i][:,:] += Hf.dat.dat[i]*adj.dat.data[i]
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
