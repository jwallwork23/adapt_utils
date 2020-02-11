from firedrake import *

from adapt_utils.solver import SteadyProblem


__all__ = ["SpaceTimeShallowWaterProblem", "SpaceTimeDispersiveShallowWaterProblem"]


class SpaceTimeShallowWaterProblem(SteadyProblem):
    """
    Class for solving shallow water problems discretised using space-time FEM.
    """
    def __init__(self, op, mesh=None, discrete_adjoint=True, prev_solution=None, levels=0):
        # TODO: FunctionSpace is currently hard-coded
        super(SpaceTimeShallowWaterProblem, self).__init__(op, mesh, None, discrete_adjoint, prev_solution, levels)
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
        self.P0 = FunctionSpace(self.mesh, "DG", 0)
        self.P1 = FunctionSpace(self.mesh, "CG", 1)
        self.P2 = FunctionSpace(self.mesh, "CG", 2)
        self.P1DG = FunctionSpace(self.mesh, "DG", 1)
        self.P1_vec = VectorFunctionSpace(self.mesh, "CG", 1)
        self.P1_ten = TensorFunctionSpace(self.mesh, "CG", 1)
        self.V = VectorFunctionSpace(self.mesh, "CG", 2, dim=2)*self.P1
        self.test = TestFunction(self.V)
        self.tests = TestFunctions(self.V)
        self.trial = TrialFunction(self.V)
        self.trials = TrialFunctions(self.V)
        self.p0test = TestFunction(self.P0)
        self.p0trial = TrialFunction(self.P0)


    def set_fields(self):
        self.fields = {}
        self.fields['viscosity'] = self.op.set_viscosity(self.P1)
        self.fields['bathymetry'] = self.op.set_bathymetry(self.P1)
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

        # Parameters
        g = self.op.g
        f = self.fields['coriolis']
        b = self.fields['bathymetry']
        nu = self.fields['viscosity']

        # Operators
        grad_x = lambda F: as_vector([F.dx(0), F.dx(1)])
        perp = lambda F: as_vector([-F[1], F[0]])
        ddt = lambda F: F.dx(2)
        n = as_vector([self.n[0], self.n[1]])

        # Initial and final time tags
        t0_tag = self.op.t_init_tag
        tf_tag = self.op.t_final_tag

        # Momentum equation
        self.lhs = inner(v, ddt(u))*dx                  # Time derivative
        self.lhs += inner(v, g*grad_x(η))*dx            # Pressure gradient term
        self.lhs += inner(v, f*perp(u))*dx              # Coriolis term
        self.lhs += nu*inner(grad_x(v), grad_x(u))*dx   # Viscosity term


        # Integration by parts for viscosity
        for i in self.mesh.exterior_facets.unique_markers:
            if not i in (t0_tag, tf_tag):
                self.lhs += -nu*inner(v, dot(grad_x(u), n))*ds(i)

        # Continuity equation
        self.lhs += inner(θ, ddt(η))*dx                 # Time derivative
        self.lhs += -inner(grad_x(θ), b*u)*dx           # Continuity term
        self.rhs = inner(grad_x(θ), b*Constant(as_vector([0.0, 0.0])))*dx

        # TODO: Enable different BCs

        # Initial conditions
        u0, eta0 = self.solution.copy(deepcopy=True).split()
        self.dbcs = [DirichletBC(self.V.sub(0), u0, t0_tag),
                     DirichletBC(self.V.sub(1), eta0, t0_tag)]

    def solve_forward(self):
        self.setup_solver_forward()
        if self.nonlinear:
            self.rhs = 0
        sol = Function(self.V*self.P1)
        self.op.print_debug("Solver parameters for forward: {:}".format(self.op.params))
        solve(self.lhs == self.rhs, sol, bcs=self.dbcs, solver_parameters=self.op.params)
        u_sol, eta_sol, lam_sol = sol.split()
        u, eta = self.solution.split()
        u.assign(u_sol)
        eta.assign(eta_sol)
        self.plot_solution(adjoint=False)

    def plot_solution(self, adjoint=False):  # FIXME: Can't seem to plot vector fields
        if adjoint:
            z, zeta = self.adjoint_solution.split()
            zeta.rename("Adjoint elevation")
            spd = interpolate(sqrt(dot(u, u)), self.P1)
            spd.rename("Adjoint fluid speed")
            self.adjoint_solution_file.write(spd, zeta)
        else:
            u, eta = self.solution.split()
            eta.rename("Elevation")
            spd = interpolate(sqrt(dot(u, u)), self.P1)
            spd.rename("Fluid speed")
            self.solution_file.write(spd, eta)


class SpaceTimeDispersiveShallowWaterProblem(SteadyProblem):
    """
    Class for solving dispersive shallow water problems discretised using space-time FEM.
    """
    def __init__(self, op, mesh=None, discrete_adjoint=True, prev_solution=None, levels=0):
        # TODO: FunctionSpace is currently hard-coded
        super(SpaceTimeDispersiveShallowWaterProblem, self).__init__(op, mesh, None, discrete_adjoint, prev_solution, levels)
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
        self.P0 = FunctionSpace(self.mesh, "DG", 0)
        self.P1 = FunctionSpace(self.mesh, "CG", 1)
        self.P2 = FunctionSpace(self.mesh, "CG", 2)
        self.P1DG = FunctionSpace(self.mesh, "DG", 1)
        self.P1_vec = VectorFunctionSpace(self.mesh, "CG", 1)
        self.P1_ten = TensorFunctionSpace(self.mesh, "CG", 1)
        self.V = VectorFunctionSpace(self.mesh, "CG", 2, dim=2)*self.P1*self.P1
        self.test = TestFunction(self.V)
        self.tests = TestFunctions(self.V)
        self.trial = TrialFunction(self.V)
        self.trials = TrialFunctions(self.V)
        self.p0test = TestFunction(self.P0)
        self.p0trial = TrialFunction(self.P0)


    def set_fields(self):
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
        perp = lambda F: as_vector([-F[1], F[0]])
        ddt = lambda F: F.dx(2)
        n = as_vector([self.n[0], self.n[1]])

        # Initial and final time tags
        t0_tag = self.op.t_init_tag
        tf_tag = self.op.t_final_tag

        # Momentum equation
        self.lhs = inner(v, ddt(u))*dx                # Time derivative
        self.lhs += inner(v, g*b*grad_x(η))*dx        # Pressure gradient term
        self.lhs += inner(div_x(v), b*b*ddt(λ)/3)*dx

        # Boundary terms resulting from integration by parts
        self.rhs = 0
        for i in self.mesh.exterior_facets.unique_markers:
            if not i in (t0_tag, tf_tag):
                self.lhs += -inner(dot(v, n), b*b*ddt(λ)/3)*ds(i)
                self.rhs += -inner(grad_x(θ), Constant(as_vector([0.0, 0.0])))*ds(i)
                self.rhs += inner(μ, Constant(0.0))*ds(i)

        # Continuity equation
        self.lhs += inner(θ, ddt(η))*dx               # Time derivative
        self.lhs += -inner(grad_x(θ), u)*dx           # Continuity term

        # Auxiliary equation, λ = div(u)
        self.lhs += inner(μ, λ)*dx + inner(grad_x(μ), u)*dx

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
            spd.rename("Adjoint fluid speed")
            zeta.rename("Adjoint elevation")
            zdiv.rename("Divergence of adjoint fluid velocity")
            self.adjoint_solution_file.write(spd, zeta, zdiv)
        else:
            u, eta, udiv = self.solution.split()
            spd = interpolate(sqrt(dot(u, u)), self.P1)
            spd.rename("Fluid speed")
            eta.rename("Elevation")
            udiv.rename("Divergence of fluid velocity")
            self.solution_file.write(spd, eta, udiv)
