from firedrake import *

import warnings

from adapt_utils.steady.solver import SteadyProblem
from adapt_utils.adapt.metric import isotropic_metric, steady_metric


__all__ = ["SpaceTimeShallowWaterProblem", "SpaceTimeDispersiveShallowWaterProblem"]


class SpaceTimeShallowWaterProblem(SteadyProblem):
    """
    Class for solving shallow water problems discretised using space-time FEM.
    """
    def __init__(self, op, mesh=None, **kwargs):
        # TODO: FunctionSpace is currently hard-coded
        super(SpaceTimeShallowWaterProblem, self).__init__(op, mesh, None, **kwargs)

        # Apply initial condition
        self.set_start_condition(adjoint=False)

        # Classification
        self.nonlinear = False

    def create_function_spaces(self):
        """
        Build the finite element space, `V`, for the prognostic solution, along with various other
        useful spaces and test functions and trial functions based upon them.
        """
        super(SpaceTimeShallowWaterProblem, self).create_function_spaces()
        self.V = VectorFunctionSpace(self.mesh, "CG", 2, dim=self.dim-1)*self.P1
        self.test = TestFunction(self.V)
        self.tests = TestFunctions(self.V)
        self.trial = TrialFunction(self.V)
        self.trials = TrialFunctions(self.V)

    def set_fields(self, adapted=False):
        self.fields = {}
        self.fields['viscosity'] = self.op.set_viscosity(self.P1)
        self.fields['bathymetry'] = self.op.set_bathymetry(self.P0)
        self.fields['coriolis'] = self.op.set_coriolis(self.P1)

    def set_stabilisation(self):
        self.stabilisation = self.stabilisation or 'no'
        try:
            assert self.stabilisation == 'no'
        except AssertionError:
            raise NotImplementedError  # TODO

    def set_start_condition(self, adjoint=False):
        self.set_solution(self.op.set_start_condition(self.V, adjoint=adjoint), adjoint)

    def setup_solver_forward(self):
        u, η = self.trials
        v, θ = self.tests

        # Adjust for length and time scales
        T = Constant(1.0 if not hasattr(self.op, 'T') else self.op.T)
        L = Constant(1.0 if not hasattr(self.op, 'L') else self.op.L)

        # Parameters
        g = self.op.g
        if self.dim == 3:
            f = self.fields['coriolis']
        b = self.fields['bathymetry']
        nu = self.fields['viscosity']

        # Operators
        if self.dim == 2:
            grad_x = lambda F: as_vector([F.dx(0), ])/L
            ddt = lambda F: F.dx(1)/T
            n = as_vector([self.n[0], ])
        elif self.dim == 3:
            grad_x = lambda F: as_vector([F.dx(0), F.dx(1)])/L
            perp = lambda F: as_vector([-F[1], F[0]])
            ddt = lambda F: F.dx(2)/T
            n = as_vector([self.n[0], self.n[1]])
        else:
            raise ValueError("Only 1+1 and 2+1 dimensional problems allowed.")

        # Initial and final time tags
        t0_tag = self.op.t_init_tag
        tf_tag = self.op.t_final_tag

        # Momentum equation
        self.lhs = inner(v, ddt(u))*dx                  # Time derivative
        self.lhs += inner(v, g*grad_x(η))*dx            # Pressure gradient term
        if self.dim == 3:
            self.lhs += inner(v, f*perp(u))*dx          # Coriolis term
        self.lhs += nu*inner(grad_x(v), grad_x(u))*dx   # Viscosity term

        # Integration by parts for viscosity
        for i in self.mesh.exterior_facets.unique_markers:
            if i not in (t0_tag, tf_tag):
                self.lhs += -nu*inner(v, dot(grad_x(u), n))*ds(i)

        # Continuity equation
        self.lhs += inner(θ, ddt(η))*dx                 # Time derivative
        self.lhs += -inner(grad_x(θ), b*u)*dx           # Continuity term
        self.rhs = inner(θ, b*Constant(0.0))*ds

        # TODO: Enable different BCs

        # Initial conditions
        u0, eta0 = self.solution.copy(deepcopy=True).split()
        self.dbcs = [DirichletBC(self.V.sub(0), u0, t0_tag),
                     DirichletBC(self.V.sub(1), eta0, t0_tag)]

    def setup_solver_adjoint(self):
        z, ζ = self.trials
        v, θ = self.tests

        # Adjust for length and time scales
        T = Constant(1.0 if not hasattr(self.op, 'T') else self.op.T)
        L = Constant(1.0 if not hasattr(self.op, 'L') else self.op.L)

        # Parameters
        g = self.op.g
        if self.dim == 3:
            f = self.fields['coriolis']
        b = self.fields['bathymetry']
        warnings.warn("#### TODO: Viscosity ignored")
        # nu = self.fields['viscosity']

        # Operators
        if self.dim == 2:
            grad_x = lambda F: as_vector([F.dx(0), ])/L
            ddt = lambda F: F.dx(1)/T
        elif self.dim == 3:
            grad_x = lambda F: as_vector([F.dx(0), F.dx(1)])/L
            perp = lambda F: as_vector([-F[1], F[0]])
            ddt = lambda F: F.dx(2)/T
        else:
            raise ValueError("Only 1+1 and 2+1 dimensional problems allowed.")

        # Initial and final time tags
        tf_tag = self.op.t_final_tag

        # Momentum equation
        self.lhs_adjoint = -inner(v, ddt(z))*dx
        # self.lhs_adjoint += -b*inner(v, grad_x(ζ))*dx
        self.lhs_adjoint += -inner(v, grad_x(b*ζ))*dx
        if self.dim == 3:
            self.lhs_adjoint += inner(v, f*perp(z))*dx
        # TODO: Viscosity term

        # Continuity equation
        self.lhs_adjoint += -inner(θ, ddt(ζ))*dx
        self.lhs_adjoint += g*inner(grad_x(θ), z)*dx

        # RHS
        self.get_qoi_kernel()
        k_u, k_eta = self.kernel.copy(deepcopy=True).split()
        self.rhs_adjoint = Constant(0.0)*(inner(v, k_u) + inner(θ, k_eta))*dx

        # TODO: Enable different BCs

        # Final time conditions
        self.dbcs_adjoint = [DirichletBC(self.V.sub(0), k_u, tf_tag),
                             DirichletBC(self.V.sub(1), k_eta, tf_tag)]

    def get_dwr_residual_forward(self):
        tpe = self.tp_enriched
        tpe.project_solution(self.solution)  # FIXME: prolong
        u, eta = tpe.solution.split()
        z, zeta = self.adjoint_error.split()

        # Adjust for length and time scales
        T = Constant(1.0 if not hasattr(self.op, 'T') else self.op.T)
        L = Constant(1.0 if not hasattr(self.op, 'L') else self.op.L)

        # Physical parameters
        g = self.op.g
        b = tpe.fields['bathymetry']
        warnings.warn("#### TODO: Viscosity ignored")
        warnings.warn("#### TODO: Coriolis ignored")

        # Operators
        if self.dim == 2:
            grad_x = lambda F: as_vector([F.dx(0), ])/L
            ddt = lambda F: F.dx(1)/T
            div_x = lambda F: F[0].dx(0)
        elif self.dim == 3:
            grad_x = lambda F: as_vector([F.dx(0), F.dx(1)])/L
            ddt = lambda F: F.dx(2)/T
            div_x = lambda F: F[0].dx(0) + F[1].dx(1)  # TODO: test
        else:
            raise ValueError("Only 1+1 and 2+1 dimensional problems allowed.")

        dwr = -inner(self.adjoint_error, ddt(tpe.solution))
        dwr += -inner(z, g*grad_x(eta))
        dwr += -inner(zeta, div_x(b*u))

        self.indicators['dwr_cell'] = project(assemble(tpe.p0test*abs(dwr)*dx), self.P0)
        self.estimate_error('dwr_cell')

    def get_dwr_flux_forward(self):
        tpe = self.tp_enriched
        i = tpe.p0test
        tpe.project_solution(self.solution)  # FIXME: prolong
        u, eta = tpe.solution.split()
        z, zeta = self.adjoint_error.split()

        # Physical parameters
        b = tpe.fields['bathymetry']
        warnings.warn("#### TODO: Viscosity ignored")

        # Normal vector
        if self.dim == 2:
            n = as_vector([tpe.n[0], ])
        elif self.dim == 3:
            n = as_vector([tpe.n[0], tpe.n[1]])
        else:
            raise ValueError("Only 1+1 and 2+1 dimensional problems allowed.")

        # Flux term resulting from integration by parts
        loc = i*inner(zeta, dot(b*u, n))
        flux_terms = (loc('+') + loc('-'))*dS + loc*ds

        # Solve auxiliary finite element problem to get traces on particular element
        mass_term = i*tpe.p0trial*dx
        res = Function(tpe.P0)
        solve(mass_term == flux_terms, res)
        self.indicators['dwr_flux'] = project(assemble(i*res*dx), self.P0)
        self.estimate_error('dwr_flux')

    def get_dwr_residual_adjoint(self):
        tpe = self.tp_enriched
        tpe.project_solution(self.solution, adjoint=True)  # FIXME: prolong
        u, eta = self.error.split()
        z, zeta = tpe.adjoint_solution.split()

        # Adjust for length and time scales
        T = Constant(1.0 if not hasattr(self.op, 'T') else self.op.T)
        L = Constant(1.0 if not hasattr(self.op, 'L') else self.op.L)

        # Physical parameters
        g = self.op.g
        b = tpe.fields['bathymetry']
        warnings.warn("#### TODO: Viscosity ignored")
        warnings.warn("#### TODO: Coriolis ignored")

        # Operators
        if self.dim == 2:
            grad_x = lambda F: as_vector([F.dx(0), ])/L
            ddt = lambda F: F.dx(1)/T
            div_x = lambda F: F[0].dx(0)
        elif self.dim == 3:
            grad_x = lambda F: as_vector([F.dx(0), F.dx(1)])/L
            ddt = lambda F: F.dx(2)/T
            div_x = lambda F: F[0].dx(0) + F[1].dx(1)  # TODO: test
        else:
            raise ValueError("Only 1+1 and 2+1 dimensional problems allowed.")

        dwr = inner(self.error, ddt(tpe.adjoint_solution))
        dwr += inner(u, grad_x(b*zeta))
        dwr += inner(eta, g*div_x(z))
        # NOTE: Kernel is DirichletBC so ignored here

        self.indicators['dwr_cell_adjoint'] = project(assemble(tpe.p0test*abs(dwr)*dx), self.P0)
        self.estimate_error('dwr_cell_adjoint')

    def get_dwr_flux_adjoint(self):
        tpe = self.tp_enriched
        i = tpe.p0test
        tpe.project_solution(self.solution, adjoint=True)  # FIXME: prolong
        u, eta = self.error.split()
        z, zeta = tpe.adjoint_solution.split()

        # Physical parameters
        g = self.op.g
        warnings.warn("#### TODO: Viscosity ignored")

        # Normal vector
        if self.dim == 2:
            n = as_vector([tpe.n[0], ])
        elif self.dim == 3:
            n = as_vector([tpe.n[0], tpe.n[1]])
        else:
            raise ValueError("Only 1+1 and 2+1 dimensional problems allowed.")

        # Flux term resulting from integration by parts
        loc = -i*inner(eta, g*dot(z, n))
        flux_terms = (loc('+') + loc('-'))*dS + loc*ds

        # Solve auxiliary finite element problem to get traces on particular element
        mass_term = i*tpe.p0trial*dx
        res = Function(tpe.P0)
        solve(mass_term == flux_terms, res)
        self.indicators['dwr_flux_adjoint'] = project(assemble(i*res*dx), self.P0)
        self.estimate_error('dwr_flux_adjoint')

    def get_isotropic_metric(self):
        """Accounts for length and time scales."""  # FIXME: Is this the right thing to do?
        el = self.indicator.ufl_element()
        name = self.indicator.dat.name
        if (el.family(), el.degree()) != ('Lagrange', 1):
            self.indicator = project(self.indicator, self.P1)
            self.indicator.rename(name)
        self.M = isotropic_metric(self.indicator, normalise=False, op=self.op)
        if hasattr(self.op, 'L') and hasattr(self.op, 'T'):
            L, T = self.op.L, self.op.T
            self.M.interpolate(as_matrix([[self.M[0, 0]/L, 0.0], [0.0, self.M[1, 1]/T]]))
        self.M = steady_metric(H=self.M, op=self.op)

    def plot_solution(self, adjoint=False):  # FIXME: Can't seem to plot vector fields
        if not self.op.plot_pvd:
            return
        if adjoint:
            z, zeta = self.adjoint_solution.split()
            zeta.rename("Adjoint elevation")
            spd = interpolate(sqrt(dot(z, z)), self.P1)
            spd.rename("Adjoint fluid speed")
            self.adjoint_solution_file.write(spd, zeta)
        else:
            u, eta = self.solution.split()
            eta.rename("Elevation")
            spd = interpolate(sqrt(dot(u, u)), self.P1)
            spd.rename("Fluid speed")
            self.solution_file.write(spd, eta)

    def checkpoint_solution(self, adjoint=False):
        if not self.op.save_hdf5:
            return
        raise NotImplementedError  # TODO: Save fields to HDF5 and plot later

    def get_qoi_kernel(self):
        self.kernel = self.op.set_qoi_kernel(self.V)

    def quantity_of_interest(self):
        if not hasattr(self, 'kernel'):
            self.get_qoi_kernel()
        self.qoi = assemble(inner(self.kernel, self.solution)*dx)
        return self.qoi

    def quantity_of_interest_form(self):
        if not hasattr(self, 'kernel'):
            self.get_qoi_kernel()
        return inner(self.kernel, self.solution)*dx


class SpaceTimeDispersiveShallowWaterProblem(SteadyProblem):
    """
    Class for solving dispersive shallow water problems discretised using space-time FEM.

    NOTES:
      * Unlike with other shallow water problems, we solve for b*u instead of u in the momentum eqn.
      * Due to the presence of the third mixed derivative, we solve the system as 2+1+1 mixed system,
        where the final scalar equation solves for the divergence of b*u.
      * We use Taylor-Hood for the momentum-continuity pair and P0 space for the auxiliary equation.
    """
    def __init__(self, op, mesh=None, **kwargs):
        # TODO: FunctionSpace is currently hard-coded
        super(SpaceTimeDispersiveShallowWaterProblem, self).__init__(op, mesh, None, **kwargs)
        try:
            assert self.mesh.topological_dimension() == 3
        except AssertionError:
            raise ValueError("We consider 2 spatial dimensions and 1 time.")

        # Apply initial condition
        self.set_start_condition(adjoint=False)

        # Classification
        self.nonlinear = False

    def create_function_spaces(self):
        """
        Build the finite element space, `V`, for the prognostic solution, along with various other
        useful spaces and test functions and trial functions based upon them.
        """
        super(SpaceTimeDispersiveShallowWaterProblem, self).create_function_spaces()
        self.V = VectorFunctionSpace(self.mesh, "CG", 2, dim=2)*self.P1*self.P1
        self.test = TestFunction(self.V)
        self.tests = TestFunctions(self.V)
        self.trial = TrialFunction(self.V)
        self.trials = TrialFunctions(self.V)

    def set_fields(self, adapted=False):
        self.fields = {}
        self.fields['bathymetry'] = self.op.set_bathymetry(self.P1)

    def set_stabilisation(self):
        self.stabilisation = self.stabilisation or 'no'
        try:
            assert self.stabilisation == 'no'
        except AssertionError:
            raise NotImplementedError  # TODO

    def set_start_condition(self, adjoint=False):
        self.set_solution(self.op.set_start_condition(self.V, adjoint=adjoint), adjoint)

    def setup_solver_forward(self):
        u, η, λ = self.trials
        v, θ, μ = self.tests

        # Parameters
        g = self.op.g
        b = self.fields['bathymetry']

        # Operators
        grad_x = lambda F: as_vector([F.dx(0), F.dx(1)])
        div_x = lambda F: F[0].dx(0) + F[1].dx(1)
        # perp = lambda F: as_vector([-F[1], F[0]])
        ddt = lambda F: F.dx(2)
        n = as_vector([self.n[0], self.n[1]])

        # Initial and final time tags
        t0_tag = self.op.t_init_tag
        tf_tag = self.op.t_final_tag

        # Momentum equation
        self.lhs = inner(v, ddt(u))*dx                # Time derivative
        self.lhs += inner(v, g*b*grad_x(η))*dx        # Pressure gradient term
        self.lhs += inner(div_x(v), b*b*ddt(λ)/3)*dx

        # Continuity equation
        self.lhs += inner(θ, ddt(η))*dx               # Time derivative
        self.lhs += -inner(grad_x(θ), u)*dx           # Continuity term

        # Auxiliary equation, λ = div(u)
        self.lhs += inner(μ, λ)*dx + inner(grad_x(μ), u)*dx
        # self.lhs += -jump(μ*u, n)*dS

        # Boundary terms resulting from integration by parts
        self.rhs = 0
        for i in self.mesh.exterior_facets.unique_markers:
            if i not in (t0_tag, tf_tag):
                self.lhs += -inner(dot(v, n), b*b*ddt(λ)/3)*ds(i)
                self.rhs += -inner(θ, Constant(0.0))*ds(i)
                self.rhs += inner(μ, Constant(0.0))*ds(i)

        # TODO: Enable different BCs

        # Initial conditions
        u0, eta0, udiv0 = self.solution.copy(deepcopy=True).split()
        self.dbcs = [DirichletBC(self.V.sub(0), u0, t0_tag),
                     DirichletBC(self.V.sub(1), eta0, t0_tag),
                     DirichletBC(self.V.sub(2), udiv0, t0_tag)]

    def plot_solution(self, adjoint=False):  # FIXME: Can't seem to plot vector fields
        if adjoint:
            z, zeta, zdiv = self.adjoint_solution.split()
            spd = interpolate(sqrt(dot(u, u)), self.P1)
            spd.rename("Adjoint of fluid speed scaled by bathymetry")
            zeta.rename("Adjoint elevation")
            zdiv.rename("Divergence")
            self.adjoint_solution_file.write(spd, zeta, zdiv)
        else:
            u, eta, udiv = self.solution.split()
            spd = interpolate(sqrt(dot(u, u)), self.P1)
            spd.rename("Fluid speed scaled by bathymetry")
            eta.rename("Elevation")
            udiv.rename("Divergence")
            self.solution_file.write(spd, eta, udiv)
