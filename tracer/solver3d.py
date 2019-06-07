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


__all__ = ["SteadyTracerProblem3d"]


class SteadyTracerProblem3d(SteadyProblem):
    # TODO: doc

    def __init__(self,
                 op,
                 mesh=None,
                 discrete_adjoint=False,
                 finite_element=FiniteElement("Lagrange", tetrahedron, 1),
                 prev_solution=None):
        super(SteadyTracerProblem3d, self).__init__(mesh, op, finite_element, discrete_adjoint, None)

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
                                       finite_element=FiniteElement(family, tetrahedron, 2))
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

