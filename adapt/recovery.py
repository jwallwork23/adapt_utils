from firedrake import *

from adapt_utils.options import Options


__all__ = ["recover_gradient", "recover_hessian", "recover_boundary_hessian",
           "L2ProjectorGradient", "DoubleL2ProjectorHessian"]


# --- Use the following drivers if only doing a single L2 projection on the current mesh

def recover_gradient(f, **kwargs):
    r"""
    Assuming the function `f` is P1 (piecewise linear and continuous), direct differentiation will
    give a gradient which is P0 (piecewise constant and discontinuous). Since we would prefer a
    smooth gradient, we solve an auxiliary finite element problem in P1 space. This 'L2 projection'
    gradient recovery technique makes use of the Cl\'ement interpolation operator.

    That `f` is P1 is not actually a requirement. In fact, this implementation supports an argument
    which is a scalar UFL expression.

    :arg f: field which we seek the gradient of.
    :kwarg bcs: boundary conditions for L2 projection.
    :param op: `Options` class object providing min/max cell size values.
    :return: reconstructed gradient associated with `f`.
    """
    kwargs.setdefault('op', Options())

    # Argument is a Function
    if isinstance(f, Function):
        return L2ProjectorGradient(f.function_space(), **kwargs).project(f)
    op = kwargs.get('op')
    op.print_debug("RECOVERY: Recovering gradient on domain interior...")

    # Argument is a UFL expression
    bcs = kwargs.get('bcs')
    mesh = kwargs.get('mesh', op.default_mesh)
    P1_vec = VectorFunctionSpace(mesh, "CG", 1)
    g, φ = TrialFunction(P1_vec), TestFunction(P1_vec)
    n = FacetNormal(mesh)
    a = inner(φ, g)*dx
    # L = inner(φ, grad(f))*dx
    L = f*dot(φ, n)*ds - div(φ)*f*dx  # Enables field to be P0
    l2_proj = Function(P1_vec, name="Recovered gradient")
    solve(a == L, l2_proj, bcs=bcs, solver_parameters=op.gradient_solver_parameters)
    return l2_proj


def recover_hessian(f, **kwargs):
    r"""
    Assuming the smooth solution field has been approximated by a function `f` which is P1, all
    second derivative information has been lost. As such, the Hessian of `f` cannot be directly
    computed. We provide two means of recovering it, as follows.

    That `f` is P1 is not actually a requirement. In fact, this implementation supports an argument
    which is a scalar UFL expression.

    (1) "Integration by parts" ('parts'):
    This involves solving the PDE $H = \nabla^T\nabla f$ in the weak sense. Code is based on the
    Monge-Amp\`ere tutorial provided on the Firedrake website:
    https://firedrakeproject.org/demos/ma-demo.py.html.

    (2) "Double L2 projection" ('dL2'):
    This involves two applications of the L2 projection operator. In this mode, we are permitted
    to recover the Hessian of a P0 field, since no derivatives of `f` are required.

    :arg f: field which we seek the Hessian of.
    :param op: `Options` class object providing min/max cell size values.
    :return: reconstructed Hessian associated with `f`.
    """
    kwargs.setdefault('op', Options())

    # Argument  is a Function
    if isinstance(f, Function):
        return DoubleL2ProjectorHessian(f.function_space(), boundary=False, **kwargs).project(f)
    op = kwargs.get('op')
    op.print_debug("RECOVERY: Recovering Hessian on domain interior...")

    # Argument is a UFL expression
    bcs = kwargs.get('bcs')
    mesh = kwargs.get('mesh', op.default_mesh)
    P1_ten = TensorFunctionSpace(mesh, "CG", 1)
    n = FacetNormal(mesh)
    solver_parameters = op.hessian_solver_parameters[op.hessian_recovery]

    # Integration by parts
    if op.hessian_recovery == 'parts':
        H, τ = TrialFunction(P1_ten), TestFunction(P1_ten)
        l2_proj = Function(P1_ten, name="Recovered Hessian")
        a = inner(τ, H)*dx
        L = -inner(div(τ), grad(f))*dx
        L += dot(grad(f), dot(τ, n))*ds
        solve(a == L, l2_proj, bcs=bcs, solver_parameters=solver_parameters)
        return l2_proj

    # Double L2 projection
    P1_vec = VectorFunctionSpace(mesh, "CG", 1)
    W = P1_vec*P1_ten
    g, H = TrialFunctions(W)
    φ, τ = TestFunctions(W)
    l2_proj = Function(W)
    a = inner(τ, H)*dx
    a += inner(φ, g)*dx
    a += inner(div(τ), g)*dx
    a += -dot(g, dot(τ, n))*ds
    # L = inner(grad(f), φ)*dx
    L = f*dot(φ, n)*ds - f*div(φ)*dx  # Enables field to be P0
    solve(a == L, l2_proj, bcs=bcs, solver_parameters=solver_parameters)
    g, H = l2_proj.split()
    H.rename("Recovered Hessian")
    return H


def recover_boundary_hessian(f, **kwargs):
    """
    Recover the Hessian of `f` on the domain boundary. That is, the Hessian in the direction
    tangential to the boundary. In two dimensions this gives a scalar field, whereas in three
    dimensions it gives a 2D field on the surface. The resulting field should only be considered on
    the boundary and is set arbitrarily to 1/h_max in the interior.

    :arg f: field which we seek the Hessian of.
    :param op: `Options` class object providing max cell size value.
    :return: reconstructed boundary Hessian associated with `f`.
    """
    kwargs.setdefault('op', Options())

    # Argument is a Function
    if isinstance(f, Function):
        return DoubleL2ProjectorHessian(f.function_space(), boundary=True, **kwargs).project(f)
    op = kwargs.get('op')
    op.print_debug("RECOVERY: Recovering Hessian on domain boundary...")

    # Argument is a UFL expression
    bcs = kwargs.get('bcs')
    mesh = kwargs.get('mesh', op.default_mesh)
    P1 = FunctionSpace(mesh, "CG", 1)
    n = FacetNormal(mesh)
    Hs, v = TrialFunction(P1), TestFunction(P1)
    l2_proj = Function(P1, name="Recovered boundary Hessian")

    # Arbitrary value in domain interior
    a = v*Hs*dx
    L = v*Constant(1/op.h_max**2)*dx

    # Hessian on boundary
    if bcs is None:
        s = perp(n)  # Tangent vector
        a_bc = v*Hs*ds
        L_bc = -dot(s, grad(v))*dot(s, grad(f))*ds
        # TODO: bbcs?
        bcs = EquationBC(a_bc == L_bc, l2_proj, 'on_boundary')

    solver_parameters = op.hessian_solver_parameters['parts']
    nullspace = VectorSpaceBasis(constant=True)
    solve(a == L, l2_proj, bcs=bcs, nullspace=nullspace, solver_parameters=solver_parameters)
    return l2_proj


# --- Use the following drivers if doing multiple L2 projections on the current mesh

class L2Projector():
    """
    Base class for performing L2 projections.

    Inherited classes must implement the :attr:`setup` method
    """

    def __init__(self, function_space, bcs=None, op=Options(), **kwargs):
        self.op = op
        self.field = Function(function_space)
        self.mesh = function_space.mesh()
        self.n = FacetNormal(self.mesh)
        self.bcs = bcs
        self.kwargs = {
            'solver_parameters': op.hessian_solver_parameters[op.hessian_recovery],
        }

    def setup(self):
        raise NotImplementedError("Should be implemented in derived class.")

    def project(self, f):
        self.op.print_debug("RECOVERY: L2 projecting {:s}...".format(self.name))
        assert f.function_space() == self.field.function_space()
        self.field.assign(f)
        if not hasattr(self, 'projector'):
            self.setup()
        if not hasattr(self, 'l2_projection'):
            raise ValueError("`setup` method should define `l2_projection` output field")
        self.projector.solve()
        return self.l2_projection


class L2ProjectorGradient(L2Projector):
    """
    Class for L2 projecting a scalar field to obtain its gradient in P1 space.

    Note that the field itself need not be continuously differentiable at all.
    """
    name = 'gradient'

    def __init__(self, *args, **kwargs):
        super(L2ProjectorGradient, self).__init__(*args, **kwargs)
        self.kwargs = {
            'solver_parameters': self.op.gradient_solver_parameters,
        }

    def setup(self):
        P1_vec = VectorFunctionSpace(self.mesh, "CG", 1)
        g, φ = TrialFunction(P1_vec), TestFunction(P1_vec)
        n = FacetNormal(self.mesh)

        a = inner(φ, g)*dx
        # L = inner(φ, grad(self.field))*dx
        L = self.field*dot(φ, n)*ds - div(φ)*self.field*dx  # Enables field to be P0
        self.l2_projection = Function(P1_vec, name="Recovered gradient")
        prob = LinearVariationalProblem(a, L, self.l2_projection, bcs=self.bcs)
        self.projector = LinearVariationalSolver(prob, **self.kwargs)


class DoubleL2ProjectorHessian(L2Projector):
    """
    Class for L2 projecting a scalar field to obtain its Hessian in P1 space.

    This can either be achieved by a direct integration by parts, or by a double L2 projection, with
    the gradient as an intermediate variable. The former approach may be specified by setting
    `op.hessian_recovery = 'parts'` and the latter by setting `op.hessian_recovery = 'dL2'`. For
    double L2 projection, a mixed finite element method is used.

    Note that the field itself need not be continuously differentiable at all.
    """
    name = 'Hessian'

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

    def setup(self):
        if self.boundary:
            self.name += ' on domain boundary'
            self._setup_boundary_projector()
        else:
            self.name += ' on domain interior'
            self._setup_interior_projector()

    def _setup_interior_projector(self):
        P1_ten = TensorFunctionSpace(self.mesh, "CG", 1)

        # Integration by parts applied to the Hessian definition
        if self.op.hessian_recovery == 'parts':
            H, τ = TrialFunction(P1_ten), TestFunction(P1_ten)
            self.l2_projection = Function(P1_ten)

            a = inner(τ, H)*dx
            L = -inner(div(τ), grad(self.field))*dx
            L += dot(grad(self.field), dot(τ, self.n))*ds

        # Double L2 projection, using a mixed formulation for the gradient and Hessian
        elif self.op.hessian_recovery == 'dL2':
            P1_vec = VectorFunctionSpace(self.mesh, "CG", 1)
            W = P1_vec*P1_ten
            g, H = TrialFunctions(W)
            φ, τ = TestFunctions(W)
            self.l2_projection = Function(W)

            a = inner(τ, H)*dx
            a += inner(φ, g)*dx
            a += inner(div(τ), g)*dx
            a += -dot(g, dot(τ, self.n))*ds

            # L = inner(grad(self.field), φ)*dx
            L = self.field*dot(φ, self.n)*ds - self.field*div(φ)*dx  # Enables field to be P0

            self.kwargs.pop('solver_parameters')  # TODO

        prob = LinearVariationalProblem(a, L, self.l2_projection, bcs=self.bcs)
        self.projector = LinearVariationalSolver(prob, **self.kwargs)

    # TODO: Couple with the above instead of setting arbitrary interior value
    def _setup_boundary_projector(self):
        P1 = FunctionSpace(mesh, "CG", 1)
        Hs, v = TrialFunction(P1), TestFunction(P1)
        self.l2_projection = Function(P1, name="Recovered boundary Hessian")

        # Arbitrary value in domain interior
        a = v*Hs*dx
        L = v*Constant(pow(self.op.h_max, -2))*dx

        # Hessian on boundary
        if self.bcs is None:
            a_bc = v*Hs*ds
            s = perp(self.n)  # Tangent vector
            L_bc = -dot(s, grad(v))*dot(s, grad(self.field))*ds
            # TODO: Account for nullspace
            self.bcs = EquationBC(a_bc == L_bc, self.l2_projection, 'on_boundary')

        prob = LinearVariationalProblem(a, L, self.l2_projection, bcs=self.bcs)
        self.projector = LinearVariationalSolver(prob, **self.kwargs)

    def project(self, f):
        assert f.function_space() == self.field.function_space()
        self.field.assign(f)
        if not self.boundary and self.op.hessian_recovery == 'dL2':
            return self._project_interior()
        else:
            return super(DoubleL2ProjectorHessian, self).project(f)

    def _project_interior(self):
        self.op.print_debug("RECOVERY: Recovering Hessian on domain interior...")
        if not hasattr(self, 'projector'):
            self.setup()
        self.projector.solve()
        g, H = self.l2_projection.split()
        H.rename("Recovered Hessian")
        return H
