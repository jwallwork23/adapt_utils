from firedrake import *

from adapt_utils.options import Options


__all__ = ["construct_gradient", "construct_hessian", "construct_boundary_hessian",
           "L2ProjectorGradient", "DoubleL2ProjectorHessian"]


# --- Use the following drivers if only doing a single L2 projection on the current mesh

def construct_gradient(*args, **kwargs):
    r"""
    Assuming the function `f` is P1 (piecewise linear and continuous), direct differentiation will
    give a gradient which is P0 (piecewise constant and discontinuous). Since we would prefer a
    smooth gradient, we solve an auxiliary finite element problem in P1 space. This "L2 projection"
    gradient recovery technique makes use of the Cl\'ement interpolation operator. That `f` is P1
    is not actually a requirement.

    :arg f: (scalar) P1 solution field.
    :kwarg mesh: mesh upon which Hessian is to be constructed. This must be applied if `f` is not a
                 Function, but a ufl expression.
    :kwarg bcs: boundary conditions for L2 projection.
    :param op: `Options` class object providing min/max cell size values.
    :return: reconstructed gradient associated with `f`.
    """
    return L2ProjectorGradient(*args, **kwargs).project()


def construct_hessian(*args, **kwargs):
    r"""
    Assuming the smooth solution field has been approximated by a function `f` which is P1, all
    second derivative information has been lost. As such, the Hessian of `f` cannot be directly
    computed. We provide two means of recovering it, as follows. That `f` is P1 is not actually
    a requirement.

    (1) "Integration by parts" ('parts'):
    This involves solving the PDE $H = \nabla^T\nabla f$ in the weak sense. Code is based on the
    Monge-Amp\`ere tutorial provided on the Firedrake website:
    https://firedrakeproject.org/demos/ma-demo.py.html.

    (2) "Double L2 projection" ('dL2'):
    This involves two applications of the L2 projection operator. In this mode, we are permitted
    to recover the Hessian of a P0 field, since no derivatives of `f` are required.

    :arg f: P1 solution field.
    :kwarg mesh: mesh upon which Hessian is to be constructed. This must be applied if `f` is not a
                 Function, but a ufl expression.
    :kwarg degree: polynomial degree of Hessian.
    :param op: `Options` class object providing min/max cell size values.
    :return: reconstructed Hessian associated with `f`.
    """
    return DoubleL2ProjectorHessian(*args, boundary=False, **kwargs).project()


def construct_boundary_hessian(f, mesh=None, degree=1, op=Options()):
    """
    Recover the Hessian of `f` on the domain boundary. That is, the Hessian in the direction
    tangential to the boundary. In two dimensions this gives a scalar field, whereas in three
    dimensions it gives a 2D field on the surface. The resulting field should only be considered on
    the boundary and is set arbitrarily to 1/h_max in the interior.

    :arg f: Scalar solution field.
    :kwarg mesh: Mesh upon which Hessian is to be constructed. This must be applied if `f` is not a
                 Function, but a ufl expression.
    :kwarg degree: Polynomial degree of Hessian.
    :param op: `Options` class object providing max cell size value.
    :return: reconstructed boundary Hessian associated with `f`.
    """
    return DoubleL2ProjectorHessian(*args, boundary=True, **kwargs).project()


# --- Use the following drivers if doing multiple L2 projections on the current mesh


class L2Projector():

    def __init__(self, field, mesh=None, bcs=None, op=Options()):
        self.field = field
        self.mesh = mesh or field.function_space().mesh()
        self.n = FacetNormal(self.mesh)
        self.bcs = bcs
        self.kwargs = {
            'solver_parameters': op.hessian_solver_parameters,
        }

    def setup(self):
        pass

    def project(self):
        if not hasattr(self, 'projector'):
            self.setup()
        self.projector.solve()
        return self.l2_projection


class L2ProjectorGradient(L2Projector):

    def setup(self):
        P1_vec = VectorFunctionSpace(self.mesh, "CG", 1)
        g, φ = TrialFunction(P1_vec), TestFunction(P1_vec)

        a = inner(φ, g)*dx
        # L = inner(φ, grad(self.f))*dx
        L = f*dot(φ, n)*ds - div(φ)*self.f*dx  # Enables field to be P0
        self.l2_projection = Function(P1_vec, name="Recovered gradient")
        prob = LinearVariationalProblem(a, L, self.l2_projection, bcs=self.bcs)
        self.projector = LinearVariationalSolver(prob, **self.kwargs)


class DoubleL2ProjectorHessian(L2Projector):

    def __init__(self, *args, boundary=False, **kwargs):
        super(DoubleL2ProjectorHessian, self).__init__(*args, **kwargs)
        self.boundary = boundary
        self.dim = self.mesh.topological_dimension()
        assert self.dim in (2, 3)
        if self.boundary:
            try:
                assert self.dim == 2
            except AssertionError:
                raise NotImplementedError  # TODO
        op = kwargs.get('op')
        self.hessian_recovery = op.hessian_recovery
        self.h_max = op.h_max

    def setup(self):
        if self.boundary:
            self._setup_boundary_projector()
        else:
            self._setup_interior_projector()

    def _setup_interior_projector(self):
        P1_ten = TensorFunctionSpace(self.mesh, "CG", 1)

        # Integration by parts applied to the Hessian definition
        if self.hessian_recovery == 'parts':
            H, τ = TrialFunction(P1_ten), TestFunction(P1_ten)
            self.l2_projection = Function(P1_ten)

            a = inner(τ, H)*dx
            L = -inner(div(τ), grad(self.field))*dx
            for i in range(self.dim):
                for j in range(self.dim):
                    L += τ[i, j]*self.n[j]*self.field.dx(i)*ds

        # Double L2 projection, using a mixed formulation for the gradient and Hessian
        elif self.hessian_recovery == 'dL2':
            P1_vec = VectorFunctionSpace(self.mesh, "CG", 1)
            W = P1_ten*P1_vec
            H, g = TrialFunctions(W)
            τ, φ = TestFunctions(W)
            self.l2_projection = Function(W)

            a = inner(τ, H)*dx
            a += inner(φ, g)*dx
            a += inner(div(τ), g)*dx
            for i in range(self.dim):
                for j in range(self.dim):
                    a += -g[i]*τ[i, j]*self.n[j]*ds

            # L = inner(grad(self.field), φ)*dx
            L = self.field*dot(φ, self.n)*ds - self.field*div(φ)*dx  # Enables field to be P0

            self.kwargs.pop('solver_parameters')  # TODO

        prob = LinearVariationalProblem(a, L, self.l2_projection, bcs=self.bcs)
        self.projector = LinearVariationalSolver(prob, **self.kwargs)

    def _setup_boundary_projector(self):
        P1 = FunctionSpace(mesh, "CG", degree)
        h, v = TrialFunction(P1), TestFunction(P1)
        self.l2_projection = Function(P1, name="Recovered boundary Hessian")

        # Arbitrary value in domain interior
        a = v*h*dx
        L = v*Constant(pow(self.h_max, -2))*dx

        # Hessian on boundary
        if self.bcs is None:
            a_bc = v*h*ds
            s = perp(self.n)  # Tangent vector
            L_bc = -(s[0]*v.dx(0)*f.dx(0) + s[1]*v.dx(1)*f.dx(1))*ds
            self.bcs = EquationBC(a_bc == L_bc, self.l2_projection, 'on_boundary')

        prob = LinearVariationalProblem(a, L, self.l2_projection, bcs=self.bcs)
        self.projector = LinearVariationalSolver(prob, **self.kwargs)

    def project(self):
        if not self.boundary and self.hessian_recovery == 'dL2':
            return self._project_interior()
        else:
            return super(DoubleL2ProjectorHessian, self).project()

    def _project_interior(self):
        if not hasattr(self, 'projector'):
            self.setup()
        self.projector.solve()
        H, g = self.l2_projection.split()
        H.rename("Recovered Hessian")
        return H
