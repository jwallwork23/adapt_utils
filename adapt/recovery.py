from firedrake import *

from adapt_utils.options import *


__all__ = ["construct_gradient", "construct_hessian", "construct_boundary_hessian"]


def construct_gradient(f, mesh=None, op=Options()):
    r"""
    Assuming the function `f` is P1 (piecewise linear and continuous), direct differentiation will
    give a gradient which is P0 (piecewise constant and discontinuous). Since we would prefer a
    smooth gradient, we solve an auxiliary finite element problem in P1 space. This "L2 projection"
    gradient recovery technique makes use of the Cl\'ement interpolation operator. That `f` is P1
    is not actually a requirement.

    :arg f: (scalar) P1 solution field.
    :kwarg mesh: mesh upon which Hessian is to be constructed. This must be applied if `f` is not a 
                 Function, but a ufl expression.
    :param op: `Options` class object providing min/max cell size values.
    :return: reconstructed gradient associated with `f`.
   """
    if mesh is None:
        mesh = f.function_space().mesh()
    P1_vec = VectorFunctionSpace(mesh, "CG", 1)
    g = TrialFunction(P1_vec)
    φ = TestFunction(P1_vec)
    # TODO: include an option to swap between these two: 'parts' vs 'L2'
    a = inner(g, φ)*dx
    # L = inner(grad(f), φ)*dx
    L = f*dot(φ, FacetNormal(mesh))*ds - f*div(φ)*dx  # enables f to be P0
    g = Function(P1_vec)
    solve(a == L, g, solver_parameters=op.hessian_solver_parameters)
    return g


def construct_hessian(f, mesh=None, degree=1, op=Options()):
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
    if mesh is None:
        mesh = f.function_space().mesh()
    dim = mesh.topological_dimension()
    assert dim in (2, 3)
    if degree == 0:
        V = TensorFunctionSpace(mesh, "DG", degree)
    else:
        V = TensorFunctionSpace(mesh, "CG", degree)
    n = FacetNormal(mesh)

    # Integration by parts applied to the Hessian definition
    if op.hessian_recovery == 'parts':
        H = TrialFunction(V)
        τ = TestFunction(V)
        a = inner(tau, H)*dx
        L = -inner(div(τ), grad(f))*dx
        for i in range(dim):
            for j in range(dim):
                L += τ[i, j]*n[j]*f.dx(i)*ds

        H = Function(V)
        solve(a == L, H, solver_parameters=op.hessian_solver_parameters)

    # Double L2 projection, using a mixed formulation for the gradient and Hessian
    elif op.hessian_recovery == 'dL2':
        P1_vec = VectorFunctionSpace(mesh, "CG", 1)
        W = V*P1_vec
        H, g = TrialFunctions(W)
        τ, φ = TestFunctions(W)
        a = inner(τ, H)*dx
        a += inner(φ, g)*dx
        a += inner(div(τ), g)*dx
        for i in range(dim):
            for j in range(dim):
                a += -g[i]*τ[i, j]*n[j]*ds

        # L = inner(grad(f), φ)*dx
        L = f*dot(φ, n)*ds - f*div(φ)*dx  # enables f to be P0

        q = Function(W)
        solve(a == L, q)  # TODO: Solver parameters?
        H = q.split()[0]

    return H

def construct_boundary_hessian(f, mesh=None, op=Options()):
    """
    Recover the Hessian of `f` on the domain boundary. That is, the Hessian in the direction
    tangential to the boundary. In two dimensions this gives a scalar field, whereas in three
    dimensions it gives a 2D field on the surface. The resulting field should only be considered on
    the boundary and is set arbitrarily to 1/h_max in the interior.

    :arg f: scalar solution field.
    :kwarg mesh: mesh upon which Hessian is to be constructed. This must be applied if `f` is not a 
                 Function, but a ufl expression.
    :param op: `Options` class object providing max cell size value.
    :return: reconstructed boundary Hessian associated with `f`.
    """
    if mesh is None:
        mesh = f.function_space().mesh()
    dim = mesh.topological_dimension()
    try:
        # assert dim in (2, 3)
        assert dim == 2
    except AssertionError:
        raise NotImplementedError  # TODO

    P1 = FunctionSpace(mesh, "CG", 1)
    h = TrialFunction(P1)
    v = TestFunction(P1)

    # Normal vector and tangent vector
    n = FacetNormal(mesh)
    s = perp(n)

    # Arbitrary value in domain interior
    a = v*h*dx
    L = v*Constant(pow(op.h_max, -2))*dx
    h_ = Function(P1)

    # Hessian on boundary
    bc = EquationBC(v*h*ds == -(s[0]*v.dx(0)*f.dx(0) + s[1]*v.dx(1)*f.dx(1))*ds, h_, 'on_boundary')
    solve(a == L, h_, bcs=[bc], solver_parameters=op.hessian_solver_parameters)

    return h_
