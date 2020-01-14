from firedrake import *

import numpy as np

from adapt_utils.adapt.adaptation import *
from adapt_utils.adapt.metric import *
from adapt_utils.adapt.kernels import eigen_kernel, matscale
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
    def __init__(self, op, mesh=None, discrete_adjoint=False, prev_solution=None, levels=1):
        if op.family in ("Lagrange", "CG", "cg"):
            fe = FiniteElement("Lagrange", triangle, op.degree)
        elif op.family in ("Discontinuous Lagrange", "DG", "dg"):
            fe = FiniteElement("Discontinuous Lagrange", triangle, op.degree)
        else:
            raise NotImplementedError
        super(SteadyTracerProblem2d, self).__init__(op, mesh, fe, discrete_adjoint, prev_solution, levels)
        self.nonlinear = False

    def set_fields(self):
        op = self.op
        self.nu = op.set_diffusivity(self.P1)
        self.u = op.set_velocity(self.P1_vec)
        self.divergence_free = np.allclose(norm(div(self.u)), 0.0)
        self.source = op.set_source(self.P1)
        self.kernel = op.set_qoi_kernel(self.P0)
        self.gradient_field = self.nu  # arbitrary field to take gradient for discrete adjoint

        # Stabilisation
        if self.stabilisation is None:
            self.stabilisation = 'SUPG'
        if self.stabilisation in ('SU', 'SUPG'):
            self.supg_coefficient(mode='nguyen')
        elif self.stabilisation == 'lax_friedrichs':
            self.stabilisation_parameter = op.stabilisaton_parameter
        elif self.stabilisation != 'no':
            raise ValueError("Stabilisation method {:s} not recognised".format(self.stabilisation))

        # Rename solution fields
        self.solution.rename('Tracer concentration')
        self.adjoint_solution.rename('Adjoint tracer concentration')

    def supg_coefficient(self, mode='nguyen'):
        r"""
        Compute SUPG stabilisation coefficent for the advection diffusion problem. There are three
        modes in which this can be calculated, as determined by the kwarg `mode`:

        In 'diameter' mode, we use the cell diameter as our measure of element size.

        In 'nguyen' mode, we follow [Nguyen et al., 2009] in projecting the edge of maximal length
        into a vector space spanning the velocity field `u` and taking the length of this projected
        edge as the measure of element size.

        In 'cell_metric' mode, we use the cell metric :math:`M` to give the measure
    ..  math::
            h = u^T M u

        In each case, we compute the stabilisation coefficent as

    ..  math::
            \tau = \frac h{2\|\textbf{u}\|}
        """
        self.am.get_cell_size(self.u, mode=mode)
        Pe = 0.5*sqrt(inner(self.u, self.u))*self.am.cell_size/self.nu
        tau = 0.5*self.am.cell_size/sqrt(inner(self.u, self.u))
        self.stabilisation_parameter = tau*min_value(1, Pe/3)

    def setup_solver_forward(self):
        phi = self.trial
        psi = self.test

        # Finite element problem
        self.lhs = psi*dot(self.u, grad(phi))*dx + self.nu*inner(grad(phi), grad(psi))*dx
        self.rhs = self.source*psi*dx

        # Stabilisation
        if self.stabilisation in ('SU', 'SUPG'):
            coeff = self.stabilisation_parameter*dot(self.u, grad(psi))
            self.lhs += coeff*dot(self.u, grad(phi))*dx
            if self.stabilisation == 'SUPG':
                self.lhs += coeff*-div(self.nu*grad(phi))*dx
                self.rhs += coeff*self.source*dx
                psi = psi + coeff
        elif self.stabilisation != 'no':
            raise ValueError("Unrecognised stabilisation method.")

        # Boundary conditions
        bcs = self.boundary_conditions
        self.dbcs = []
        for i in bcs.keys():
            if bcs[i] == {}:
                self.lhs += -self.nu*psi*dot(self.n, nabla_grad(phi))*ds(i)
            if 'diff_flux' in bcs[i]:
                self.lhs += -self.nu*psi*dot(self.n, nabla_grad(phi))*ds(i)
                self.rhs += -psi*bcs[i]['diff_flux']*ds(i)
            if 'value' in bcs[i]:
                self.dbcs.append(DirichletBC(self.V, bcs[i]['value'], i))

    def setup_solver_adjoint(self):
        lam = self.trial
        psi = self.test

        # Adjoint finite element problem
        self.lhs_adjoint = lam*dot(self.u, grad(psi))*dx + self.nu*inner(grad(lam), grad(psi))*dx
        self.rhs_adjoint = self.kernel*psi*dx

        # Stabilisation
        if self.stabilisation in ('SU', 'SUPG'):
            coeff = -self.stabilisation_parameter*div(self.u*psi)
            self.lhs_adjoint += -coeff*div(self.u*lam)*dx
            if self.stabilisation == 'SUPG':  # NOTE: this is not equivalent to discrete adjoint
                self.lhs_adjoint += coeff*-div(self.nu*grad(lam))*dx
                self.rhs_adjoint += coeff*self.kernel*dx
                psi = psi + coeff
        elif self.stabilisation != 'no':
            raise ValueError("Unrecognised stabilisation method.")

        # Boundary conditions
        bcs = self.boundary_conditions
        self.dbcs_adjoint = []
        for i in bcs.keys():
            if not 'diff_flux' in bcs[i]:
                self.dbcs_adjoint.append(DirichletBC(self.V, 0, i))  # Dirichlet BC in adjoint
            if not 'value' in bcs[i]:
                self.lhs_adjoint += -lam*psi*(dot(self.u, self.n))*ds(i)
                self.lhs_adjoint += -self.nu*psi*dot(self.n, nabla_grad(lam))*ds(i)  # Robin BC in adjoint

    def get_strong_residual_forward(self):
        R = self.source - dot(self.u, grad(self.solution)) + div(self.nu*grad(self.solution))
        self.indicators['cell_residual_forward'] = assemble(self.p0test*abs(R)*dx)
        self.indicator = interpolate(self.indicators['cell_residual_forward'], self.P1)
        # self.indicator = interpolate(R, self.P1)
        # self.indicator = interpolate(abs(self.indicator), self.P1)
        self.indicator.rename('forward strong residual')
        self.estimate_error('cell_residual_forward')

    def get_strong_residual_adjoint(self):
        R = self.kernel + div(self.u*self.adjoint_solution) + div(self.nu*grad(self.adjoint_solution))
        self.indicators['cell_residual_adjoint'] = assemble(self.p0test*abs(R)*dx)
        self.indicator = interpolate(self.indicators['cell_residual_adjoint'], self.P1)
        # self.indicator = interpolate(R, self.P1)
        # self.indicator = interpolate(abs(self.indicator), self.P1)
        self.indicator.rename('adjoint strong residual')
        self.estimate_error('cell_residual_adjoint')

    def get_dwr_residual_forward(self):
        tpe = self.tp_enriched
        tpe.project_solution(self.solution)  # FIXME: prolong
        strong_residual = tpe.source - dot(tpe.u, grad(tpe.solution)) + div(tpe.nu*grad(tpe.solution))
        dwr = strong_residual*self.adjoint_error
        if self.stabilisation == 'SUPG':  # Account for stabilisation error
            dwr += strong_residual*tpe.stabilisation_parameter*dot(tpe.u, grad(self.adjoint_error))
        self.indicators['dwr_cell'] = project(assemble(tpe.p0test*abs(dwr)*dx), self.P0)
        self.estimate_error('dwr_cell')

    def get_dwr_flux_forward(self):
        tpe = self.tp_enriched
        i = tpe.p0test
        tpe.project_solution(self.solution)  # FIXME: prolong

        # Flux terms (arising from integration by parts)
        mass_term = i*tpe.p0trial*dx
        flux = -tpe.nu*dot(tpe.n, nabla_grad(tpe.solution))
        dwr = flux*self.adjoint_error
        if self.stabilisation == 'SUPG':  # Account for stabilisation error
            coeff = tpe.stabilisation_parameter*dot(tpe.u, grad(self.adjoint_error))
            dwr += coeff*flux
        flux_terms = ((i*dwr)('+') + (i*dwr)('-'))*dS

        # Account for boundary conditions
        # NOTES:
        #   * For CG methods, Dirichlet error is zero, by construction.
        #   * Negative sign in `flux`.
        bcs = tpe.boundary_conditions
        for j in bcs:
            if 'diff_flux' in bcs[j]:
                flux_terms += i*(dwr + bcs[j]['diff_flux']*self.adjoint_error)*ds(j)
                if self.stabilisation == "SUPG":
                    flux_terms += i*bcs[j]['diff_flux']*coeff*ds(j)

        # Solve auxiliary FEM problem
        edge_res = Function(tpe.P0)
        solve(mass_term == flux_terms, edge_res)
        self.indicators['dwr_flux'] = project(assemble(i*abs(edge_res)*dx), self.P0)
        self.estimate_error('dwr_flux')

    def get_dwr_residual_adjoint(self):
        tpe = self.tp_enriched
        tpe.project_solution(self.adjoint_solution, adjoint=True)  # FIXME: prolong
        strong_residual = tpe.op.box(tpe.P0) + div(tpe.u*tpe.adjoint_solution) + div(tpe.nu*grad(tpe.adjoint_solution))
        dwr = strong_residual*self.error
        if self.stabilisation == 'SUPG':  # Account for stabilisation error
            dwr += strong_residual*tpe.stabilisation_parameter*dot(tpe.u, grad(self.error))
        self.indicators['dwr_cell_adjoint'] = project(assemble(tpe.p0test*abs(dwr)*dx), self.P0)
        self.estimate_error('dwr_cell_adjoint')
        
    def get_dwr_flux_adjoint(self):
        tpe = self.tp_enriched
        i = tpe.p0test
        tpe.project_solution(self.adjoint_solution, adjoint=True)  # FIXME: prolong

        # Edge residual
        mass_term = i*tpe.p0trial*dx
        flux = -(tpe.adjoint_solution*dot(tpe.u, tpe.n) + tpe.nu*dot(tpe.n, nabla_grad(tpe.adjoint_solution)))
        dwr = flux*self.error
        if self.stabilisation == 'SUPG':  # Account for stabilisation error
            dwr += flux*tpe.stabilisation_parameter*dot(tpe.u, grad(self.error))
        flux_terms = ((i*dwr)('+') + (i*dwr)('-'))*dS

        # Account for boundary conditions
        bcs = tpe.boundary_conditions
        for j in bcs.keys():
            if not 'value' in bcs[j]:
                flux_terms += i*dwr*ds(j)  # Robin BC in adjoint

        # Solve auxiliary FEM problem
        edge_res_adjoint = Function(tpe.P0)
        solve(mass_term == flux_terms, edge_res_adjoint)
        self.indicators['dwr_flux_adjoint'] = project(assemble(i*abs(edge_res_adjoint)*dx), self.P0)
        self.estimate_error('dwr_flux_adjoint')

    def get_hessian_metric(self, adjoint=False, noscale=False):
        self.M = steady_metric(self.get_solution(adjoint), mesh=self.mesh, noscale=noscale, op=self.op)
        
    def get_loseille_metric(self, adjoint=False, relax=True):
        adj = self.get_solution(not adjoint)
        sol = self.get_solution(adjoint)
        adj_diff = interpolate(abs(construct_gradient(adj)), self.P1_vec)
        adj_diff.rename("Gradient of adjoint solution")
        adj_diff_x = interpolate(adj_diff[0], self.P1)
        adj_diff_x.rename("x-derivative of adjoint solution")
        adj_diff_y = interpolate(adj_diff[1], self.P1)
        adj_diff_y.rename("y-derivative of adjoint solution")
        adj = interpolate(abs(adj), self.P1)
        adj.rename("Adjoint solution in modulus")

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
        H1.rename("Hessian for x-component")
        H2 = steady_metric(F2, mesh=self.mesh, noscale=True, op=self.op)
        H2.rename("Hessian for y-component")
        Hf = steady_metric(source, mesh=self.mesh, noscale=True, op=self.op)
        Hf.rename("Hessian for source term")

        # Hessians for conservative parts
        M1 = Function(self.P1_ten).assign(0.0)
        M2 = Function(self.P1_ten).assign(0.0)
        kernel = eigen_kernel(matscale, 2)
        op2.par_loop(kernel,
                     self.P1_ten.node_set,
                     M1.dat(op2.RW),
                     H1.dat(op2.READ),
                     adj_diff_x.dat(op2.READ))
        M1 = steady_metric(None, H=M1, op=self.op)
        M1.rename("Metric for x-component of conservative terms")
        op2.par_loop(kernel,
                     self.P1_ten.node_set,
                     M2.dat(op2.RW),
                     H2.dat(op2.READ),
                     adj_diff_y.dat(op2.READ))
        M2 = steady_metric(None, H=M2, op=self.op)
        M2.rename("Metric for y-component of conservative terms")
        M = combine_metrics(M1, M2, average=relax)
        # self.M = combine_metrics(M1, M2, average=relax)

        # Account for source term
        Mf = Function(self.P1_ten).assign(0.0)
        op2.par_loop(kernel,
                     self.P1_ten.node_set,
                     Mf.dat(op2.RW),
                     Hf.dat(op2.READ),
                     adj.dat(op2.READ))
        Mf = steady_metric(None, H=Mf, op=self.op)
        Mf.rename("Metric for source term")

        # Combine contributions
        self.M = combine_metrics(M, Mf, average=relax)
        self.M.rename("Loseille metric")

        # Account for boundary contributions  # TODO: Use EquationBC
        # bdy_contributions = i*(F1*n[0] + F2*n[1])*ds
        # n = self.n
        # Fhat = i*dot(phi, n)
        # bdy_contributions -= Fhat*ds(2) + Fhat*ds(3) + Fhat*ds(4)


class UnsteadyTracerProblem2d(UnsteadyProblem):
    def __init__(self):
        raise NotImplementedError  # TODO: Copy over most of the above
