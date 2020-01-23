from firedrake import *


__all__ = ["local_norm", "local_edge_integral", "local_interior_edge_integral",
           "local_boundary_integral", "local_edge_norm", "local_interior_edge_norm",
           "local_boundary_norm", "frobenius_norm", "local_frobenius_norm"]


def local_norm(f, norm_type='L2'):
    """
    Calculate the `norm_type`-norm of `f` separately on each element of the mesh.
    """
    typ = norm_type.lower()
    mesh = f.function_space().mesh()
    i = TestFunction(FunctionSpace(mesh, "DG", 0))

    if isinstance(f, Function):
        if typ == 'l2':
            form = i*inner(f, f)*dx
        elif typ == 'h1':
            form = i*inner(f, f)*dx + i*inner(grad(f), grad(f))*dx
        elif typ == "hdiv":
            form = i*inner(f, f)*dx + i*div(f)*div(f)*dx
        elif typ == "hcurl":
            form = i*inner(f, f)*dx + i*inner(curl(f), curl(f))*dx
        else:
            raise RuntimeError("Unknown norm type '{:s}'".format(norm_type))
    else:
        if typ == 'l2':
            form = i*sum(inner(fi, fi) for fi in f)*dx
        elif typ == 'h1':
            form = i*sum(inner(fi, fi)*dx + inner(grad(fi), grad(fi)) for fi in f)*dx
        elif typ == "hdiv":
            form = i*sum(inner(fi, fi)*dx + div(fi) * div(fi) for fi in f)*dx
        elif typ == "hcurl":
            form = i*sum(inner(fi, fi)*dx + inner(curl(fi), curl(fi)) for fi in f)*dx
        else:
            raise RuntimeError("Unknown norm type '{:s}'".format(norm_type))
    return sqrt(assemble(form))

def local_edge_integral(f, mesh=None):
    """
    Integrates `f` over all edges elementwise, giving a P0 field. 
    """
    mesh = mesh or f.function_space().mesh()
    P0 = FunctionSpace(mesh, 'DG', 0)
    test, trial, integral = TestFunction(P0), TrialFunction(P0), Function(P0)
    solve(test*trial*dx == ((test*f)('+') + (test*f)('-'))*dS + test*f*ds, integral)
    return integral

def local_interior_edge_integral(f, mesh=None):
    """
    Integrates `f` over all interior edges elementwise, giving a P0 field. 
    """
    mesh = mesh or f.function_space().mesh()
    P0 = FunctionSpace(mesh, 'DG', 0)
    test, trial, integral = TestFunction(P0), TrialFunction(P0), Function(P0)
    solve(test*trial*dx == ((test*f)('+') + (test*f)('-'))*dS, integral)
    return integral

def local_boundary_integral(f, mesh=None):
    """
    Integrates `f` over all exterior edges elementwise, giving a P0 field. 
    """
    mesh = mesh or f.function_space().mesh()
    P0 = FunctionSpace(mesh, 'DG', 0)
    test, trial, integral = TestFunction(P0), TrialFunction(P0), Function(P0)
    solve(test*trial*dx == test*f*ds, integral)
    return integral

def local_edge_norm(f, mesh=None):
    mesh = mesh or f.function_space().mesh()
    return local_edge_integral(f*f, mesh)  # TODO: Norms other than L2

def local_interior_edge_norm(f, mesh=None):
    mesh = mesh or f.function_space().mesh()
    return local_interior_edge_integral(f*f, mesh)  # TODO: Norms other than L2

def local_boundary_norm(f, mesh=None):
    mesh = mesh or f.function_space().mesh()
    return local_boundary_integral(f*f, mesh)  # TODO: Norms other than L2)

def frobenius_norm(matrix, mesh=None):
    mesh = mesh or matrix.function_space().mesh()
    dim = mesh.topological_dimension()
    f = 0
    for i in range(dim):
        for j in range(dim):
            f += matrix[i, j]*matrix[i, j]
    return sqrt(assemble(f*dx))

def local_frobenius_norm(matrix, mesh=None, space=None):
    mesh = mesh or matrix.function_space().mesh()
    space = space or FunctionSpace(mesh, "DG", 0)
    dim = mesh.topological_dimension()
    f = 0
    for i in range(dim):
        for j in range(dim):
            f += matrix[i, j]*matrix[i, j]
    return sqrt(assemble(TestFunction(space)*f*dx))
