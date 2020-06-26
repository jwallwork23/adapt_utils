from firedrake import *

from adapt_utils.adapt.metric import *
from adapt_utils.adapt.kernels import eigen_kernel, matscale
from adapt_utils.adapt.recovery import *
from adapt_utils.adapt.r import *
from adapt_utils.steady.solver import SteadyProblem


__all__ = ["SteadyTracerProblem2d"]


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
    def __init__(self, op, mesh=None, **kwargs):
        if op.family in ("Lagrange", "CG", "cg"):
            fe = FiniteElement("Lagrange", triangle, op.degree)
        elif op.family in ("Discontinuous Lagrange", "DG", "dg"):
            fe = FiniteElement("Discontinuous Lagrange", triangle, op.degree, variant='equispaced')
        else:
            raise NotImplementedError
        super(SteadyTracerProblem2d, self).__init__(op, mesh, fe, **kwargs)
        self.nonlinear = False

    def set_fields(self, adapted=False):
        op = self.op
        self.fields = {}
        self.fields['diffusivity'] = op.set_diffusivity(self.P1)
        self.fields['velocity'] = op.set_velocity(self.P1_vec)
        self.fields['source'] = op.set_source(self.P1)

    def create_solutions(self):
        super(SteadyTracerProblem2d, self).create_solutions()
        self.solution.rename('Tracer concentration')
        self.adjoint_solution.rename('Adjoint tracer concentration')

    def set_stabilisation(self):
        self.stabilisation = self.stabilisation or 'SUPG'
        if self.stabilisation in ('SU', 'SUPG'):
            self.supg_coefficient(mode='diameter')
            # self.supg_coefficient(mode='nguyen')
        elif self.stabilisation == 'lax_friedrichs':
            self.stabilisation_parameter = op.stabilisaton_parameter
        elif self.stabilisation != 'no':
            raise ValueError("Stabilisation method {:s} not recognised".format(self.stabilisation))

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
        u = self.fields['velocity']
        self.am.get_cell_size(u, mode=mode)
        unorm = sqrt(inner(u, u))
        Pe = 0.5*unorm*self.am.cell_size/self.fields['diffusivity']
        tau = 0.5*self.am.cell_size/unorm
        self.stabilisation_parameter = tau*min_value(1, Pe/3)

    def setup_solver_forward(self):
        phi = self.trial
        psi = self.test
        nu = self.fields['diffusivity']
        u = self.fields['velocity']
        source = self.fields['source']

        if self.stabilisation in ('SU', 'SUPG'):
            coeff = self.stabilisation_parameter*dot(u, grad(psi))
            if self.stabilisation == 'SUPG':
                psi = psi + coeff
        elif self.stabilisation != 'no':
            raise ValueError("Unrecognised stabilisation method.")

        # Finite element problem
        self.lhs = psi*dot(u, grad(phi))*dx + nu*inner(grad(psi), grad(phi))*dx
        self.rhs = psi*source*dx

        # Stabilisation
        if self.stabilisation == 'SU':
            self.lhs += coeff*dot(u, grad(phi))*dx
        # if self.stabilisation in ('SU', 'SUPG'):
        #     coeff = self.stabilisation_parameter*dot(u, grad(psi))
        #     self.lhs += coeff*dot(u, grad(phi))*dx
        #     if self.stabilisation == 'SUPG':
        #         self.lhs += coeff*-div(nu*grad(phi))*dx
        #         self.rhs += coeff*source*dx
        # elif self.stabilisation != 'no':
        #     raise ValueError("Unrecognised stabilisation method.")

        # Boundary conditions
        bcs = self.boundary_conditions
        self.dbcs = []
        for i in bcs.keys():
            if bcs[i] == {}:
                self.lhs += -nu*psi*dot(self.n, nabla_grad(phi))*ds(i)
            elif 'value' in bcs[i]:
                self.dbcs.append(DirichletBC(self.V, bcs[i]['value'], i))
            elif 'diff_flux' in bcs[i]:
                self.rhs += psi*bcs[i]['diff_flux']*ds(i)
            else:
                raise ValueError("Unrecognised BC types in {:}.".format(bcs[i]))

    def setup_solver_adjoint(self):
        lam = self.trial
        psi = self.test
        nu = self.fields['diffusivity']
        u = self.fields['velocity']
        self.get_qoi_kernel()

        if self.stabilisation in ('SU', 'SUPG'):
            coeff = -self.stabilisation_parameter*div(u*psi)
            if self.stabilisation == 'SUPG':
                psi = psi + coeff
        elif self.stabilisation != 'no':
            raise ValueError("Unrecognised stabilisation method.")

        # Adjoint finite element problem
        self.lhs_adjoint = lam*dot(u, grad(psi))*dx + nu*inner(grad(lam), grad(psi))*dx
        self.rhs_adjoint = self.kernel*psi*dx

        # Stabilisation
        if self.stabilisation == 'SU':
            self.lhs_adjoint += -coeff*div(u*lam)*dx
        # if self.stabilisation in ('SU', 'SUPG'):
        #     coeff = -self.stabilisation_parameter*div(u*psi)
        #     self.lhs_adjoint += -coeff*div(u*lam)*dx
        #     if self.stabilisation == 'SUPG':  # NOTE: this is not equivalent to discrete adjoint
        #         self.lhs_adjoint += coeff*-div(nu*grad(lam))*dx
        #         self.rhs_adjoint += coeff*self.kernel*dx
        # elif self.stabilisation != 'no':
        #     raise ValueError("Unrecognised stabilisation method.")

        # Boundary conditions
        bcs = self.boundary_conditions
        self.dbcs_adjoint = []
        for i in bcs.keys():
            if bcs[i] != {}:
                try:
                    assert 'diff_flux' in bcs[i] or 'value' in bcs[i]
                except AssertionError:
                    raise ValueError("Unrecognised BC types in {:}.".format(bcs[i]))
            if 'diff_flux' not in bcs[i]:
                self.dbcs_adjoint.append(DirichletBC(self.V, 0, i))  # Dirichlet BC in adjoint
            elif 'value' in bcs[i]:
                self.lhs_adjoint += -lam*psi*(dot(u, self.n))*ds(i)
                self.lhs_adjoint += -nu*psi*dot(self.n, nabla_grad(lam))*ds(i)  # Robin BC in adjoint

    def get_qoi_kernel(self):
        self.kernel = self.op.set_qoi_kernel(self.P0)

    def get_strong_residual_forward(self, norm_type=None):
        u = self.fields['velocity']
        nu = self.fields['diffusivity']
        sol = self.solution
        R = self.fields['source'] - dot(u, grad(sol)) + div(nu*grad(sol))
        if norm_type is None:
            self.indicators['cell_residual_forward'] = assemble(self.p0test*R*dx)
        elif norm_type == 'L1':
            self.indicators['cell_residual_forward'] = assemble(self.p0test*abs(R)*dx)
        elif norm_type == 'L2':
            self.indicators['cell_residual_forward'] = assemble(self.p0test*R*R*dx)
        else:
            raise ValueError("Norm should be chosen from {None, 'L1' or 'L2'}.")
        self.indicator = interpolate(self.indicators['cell_residual_forward'], self.P1)  # NOTE
        # self.indicator = interpolate(R, self.P1)
        # self.indicator = interpolate(abs(self.indicator), self.P1)
        self.indicator.rename('forward strong residual')
        self.estimate_error('cell_residual_forward')

    def get_strong_residual_adjoint(self, norm_type=None):
        u = self.fields['velocity']
        nu = self.fields['diffusivity']
        sol = self.adjoint_solution
        R = self.kernel + div(u*sol) + div(nu*grad(sol))
        if norm_type is None:
            self.indicators['cell_residual_adjoint'] = assemble(self.p0test*R*dx)
        elif norm_type == 'L1':
            self.indicators['cell_residual_adjoint'] = assemble(self.p0test*abs(R)*dx)
        elif norm_type == 'L2':
            self.indicators['cell_residual_adjoint'] = assemble(self.p0test*R*R*dx)
        else:
            raise ValueError("Norm should be chosen from {None, 'L1' or 'L2'}.")
        self.indicator = interpolate(self.indicators['cell_residual_adjoint'], self.P1)  # NOTE
        # self.indicator = interpolate(R, self.P1)
        # self.indicator = interpolate(abs(self.indicator), self.P1)
        self.indicator.rename('adjoint strong residual')
        self.estimate_error('cell_residual_adjoint')

    def get_flux_forward(self, norm_type=None, residual_approach='classical'):
        i = self.p0test
        nu = self.fields['diffusivity']
        assert residual_approach in ('classical', 'difference_quotient')
        sol = self.solution if residual_approach == 'classical' else self.adjoint_solution

        # Flux terms (arising from integration by parts)
        mass_term = i*self.p0trial*dx
        flux = -nu*dot(self.n, nabla_grad(sol))
        if norm_type is None:
            flux_terms = ((i*flux)('+') + (i*flux)('-'))*dS
        elif norm_type == 'L1':
            flux_terms = ((i*abs(flux))('+') + (i*abs(flux))('-'))*dS
        elif norm_type == 'L2':
            flux_terms = ((i*flux*flux)('+') + (i*flux*flux)('-'))*dS
        else:
            raise ValueError("Norm should be chosen from {None, 'L1' or 'L2'}.")

        # Account for boundary conditions
        # NOTES:
        #   * For CG methods, Dirichlet error is zero, by construction.
        #   * Negative sign in `flux`.
        bcs = self.boundary_conditions
        for j in bcs:
            if 'diff_flux' in bcs[j]:
                bdy_flux = flux + bcs[j]['diff_flux']
                if norm_type is None:
                    flux_terms += i*bdy_flux*ds(j)
                elif norm_type == 'L1':
                    flux_terms += i*abs(bdy_flux)*ds(j)
                elif norm_type == 'L2':
                    flux_terms += i*bdy_flux*bdy_flux*ds(j)

        # Solve auxiliary FEM problem
        name = 'flux_forward'
        self.indicators[name] = Function(self.P0)
        solve(mass_term == flux_terms, self.indicators[name])
        self.estimate_error(name)
        return name

    def get_flux_adjoint(self, norm_type=None, residual_approach='classical'):
        i = self.p0test
        u = self.fields['velocity']
        nu = self.fields['diffusivity']
        assert residual_approach in ('classical', 'difference_quotient')
        sol = self.adjoint_solution if residual_approach == 'classical' else self.solution

        # Edge residual
        mass_term = i*self.p0trial*dx
        flux = -(sol*dot(u, self.n) + nu*dot(self.n, nabla_grad(sol)))
        if norm_type is None:
            flux_terms = ((i*flux)('+') + (i*flux)('-'))*dS
        elif norm_type == 'L1':
            flux_terms = ((i*abs(flux))('+') + (i*abs(flux))('-'))*dS
        elif norm_type == 'L2':
            flux_terms = ((i*flux*flux)('+') + (i*flux*flux)('-'))*dS
        else:
            raise ValueError("Norm should be chosen from {None, 'L1' or 'L2'}.")

        # Account for boundary conditions
        bcs = self.boundary_conditions
        for j in bcs.keys():
            if 'value' not in bcs[j]:  # Robin BC in adjoint
                if norm_type is None:
                    flux_terms += i*flux*ds(j)
                elif norm_type == 'L1':
                    flux_terms += i*abs(flux)*ds(j)
                elif norm_type == 'L2':
                    flux_terms += i*flux*flux*ds(j)

        # Solve auxiliary FEM problem
        name = 'flux_adjoint'
        self.indicators[name] = Function(self.P0)
        solve(mass_term == flux_terms, self.indicators[name])
        self.estimate_error(name)
        return name

    def get_dwr_residual_forward(self):
        tpe = self.tp_enriched
        tpe.project_solution(self.solution)  # FIXME: prolong
        u = tpe.fields['velocity']
        nu = tpe.fields['diffusivity']
        source = tpe.fields['source']

        strong_residual = source - dot(u, grad(tpe.solution)) + div(nu*grad(tpe.solution))
        dwr = strong_residual*self.adjoint_error
        if self.stabilisation == 'SUPG':  # Account for stabilisation error
            dwr += strong_residual*tpe.stabilisation_parameter*dot(u, grad(self.adjoint_error))

        name = 'dwr_cell'
        self.indicators[name] = project(assemble(tpe.p0test*dwr*dx), self.P0)
        self.estimate_error(name)
        return name

    def get_dwr_flux_forward(self):
        tpe = self.tp_enriched
        i = tpe.p0test
        tpe.project_solution(self.solution)  # FIXME: prolong
        nu = tpe.fields['diffusivity']

        # Flux terms (arising from integration by parts)
        mass_term = i*tpe.p0trial*dx
        flux = -nu*dot(tpe.n, nabla_grad(tpe.solution))
        dwr = flux*self.adjoint_error
        flux_terms = ((i*dwr)('+') + (i*dwr)('-'))*dS

        # Account for boundary conditions
        # NOTES:
        #   * For CG methods, Dirichlet error is zero, by construction.
        #   * Negative sign in `flux`.
        bcs = tpe.boundary_conditions
        for j in bcs:
            if 'diff_flux' in bcs[j]:
                flux_terms += i*(dwr + bcs[j]['diff_flux']*self.adjoint_error)*ds(j)

        # Solve auxiliary FEM problem
        name = 'dwr_flux'
        edge_res = Function(tpe.P0)
        solve(mass_term == flux_terms, edge_res)
        self.indicators[name] = project(edge_res, self.P0)
        self.estimate_error(name)
        return name

    def get_dwr_residual_adjoint(self):
        tpe = self.tp_enriched
        tpe.project_solution(self.adjoint_solution, adjoint=True)  # FIXME: prolong
        u = tpe.fields['velocity']
        nu = tpe.fields['diffusivity']
        kernel = tpe.kernel

        strong_residual = kernel + div(u*tpe.adjoint_solution) + div(nu*grad(tpe.adjoint_solution))
        dwr = strong_residual*self.error

        name = 'dwr_cell_adjoint'
        self.indicators[name] = project(assemble(tpe.p0test*dwr*dx), self.P0)
        self.estimate_error(name)
        return name

    def get_dwr_flux_adjoint(self):
        tpe = self.tp_enriched
        i = tpe.p0test
        tpe.project_solution(self.adjoint_solution, adjoint=True)  # FIXME: prolong
        u = tpe.fields['velocity']
        nu = tpe.fields['diffusivity']
        n = tpe.n

        # Edge residual
        mass_term = i*tpe.p0trial*dx
        flux = -(tpe.adjoint_solution*dot(u, n) + nu*dot(n, nabla_grad(tpe.adjoint_solution)))
        dwr = flux*self.error
        flux_terms = ((i*dwr)('+') + (i*dwr)('-'))*dS

        # Account for boundary conditions
        bcs = tpe.boundary_conditions
        for j in bcs.keys():
            if 'value' not in bcs[j]:
                flux_terms += i*dwr*ds(j)  # Robin BC in adjoint

        # Solve auxiliary FEM problem
        name = 'dwr_flux_adjoint'
        edge_res_adjoint = Function(tpe.P0)
        solve(mass_term == flux_terms, edge_res_adjoint)
        self.indicators[name] = project(edge_res_adjoint, self.P0)
        self.estimate_error(name)
        return name

    def get_hessian_metric(self, adjoint=False, normalise=True):
        sol = self.get_solution(adjoint)
        self.M = steady_metric(sol, mesh=self.mesh, normalise=normalise, op=self.op)

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

        nu = self.fields['diffusivity']
        u = self.fields['velocity']

        # Get potential to take Hessian w.r.t.
        # x, y = SpatialCoordinate(self.mesh)
        if adjoint:
            source = self.kernel
            # F1 = -sol*u[0] - nu*sol.dx(0) - source*x
            # F2 = -sol*u[1] - nu*sol.dx(1) - source*y
            F1 = -sol*u[0] - nu*sol.dx(0)
            F2 = -sol*u[1] - nu*sol.dx(1)
        else:
            source = self.fields['source']
            # F1 = sol*u[0] - nu*sol.dx(0) - source*x
            # F2 = sol*u[1] - nu*sol.dx(1) - source*y
            F1 = sol*u[0] - nu*sol.dx(0)
            F2 = sol*u[1] - nu*sol.dx(1)

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
        H1 = steady_metric(F1, mesh=self.mesh, normalise=False, op=self.op)
        H1.rename("Hessian for x-component")
        H2 = steady_metric(F2, mesh=self.mesh, normalise=False, op=self.op)
        H2.rename("Hessian for y-component")
        Hf = steady_metric(source, mesh=self.mesh, normalise=False, op=self.op)
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
