from firedrake import *


__all__ = ["local_norm", "local_edge_integral", "local_interior_edge_integral",
           "local_boundary_integral", "local_edge_norm", "local_interior_edge_norm",
           "local_boundary_norm"]


def local_norm(f, norm_type='L2'):
    """
    Calculate the `norm_type`-norm of `f` separately on each element of the mesh.
    """
    typ = norm_type.lower()
    mesh = f.function_space().mesh()
    i = TestFunction(FunctionSpace(mesh, "DG", 0))

    if isinstance(f, FiredrakeFunction):
        if typ == 'l2':
            form = i * inner(f, f) * dx
        elif typ == 'h1':
            form = i * (inner(f, f) * dx + inner(grad(f), grad(f))) * dx
        elif typ == "hdiv":
            form = i * (inner(f, f) * dx + div(f) * div(f)) * dx
        elif typ == "hcurl":
            form = i * (inner(f, f) * dx + inner(curl(f), curl(f))) * dx
        else:
            raise RuntimeError("Unknown norm type '%s'" % norm_type)
    else:
        if typ == 'l2':
            form = i * sum(inner(fi, fi) for fi in f) * dx
        elif typ == 'h1':
            form = i * sum(inner(fi, fi) * dx + inner(grad(fi), grad(fi)) for fi in f) * dx
        elif typ == "hdiv":
            form = i * sum(inner(fi, fi) * dx + div(fi) * div(fi) for fi in f) * dx
        elif typ == "hcurl":
            form = i * sum(inner(fi, fi) * dx + inner(curl(fi), curl(fi)) for fi in f) * dx
        else:
            raise RuntimeError("Unknown norm type '%s'" % norm_type)
    return sqrt(assemble(form))


def local_edge_integral(f, mesh=None):
    """
    Integrates `f` over all edges elementwise, giving a P0 field. 
    """
    if mesh is None:
        mesh = f.function_space().mesh()
    P0 = FunctionSpace(mesh, 'DG', 0)
    i = TestFunction(P0)
    return project(assemble(((f*i)('+') + (f*i)('-'))*dS + f*i*ds), P0)


def local_interior_edge_integral(f, mesh=None):
    """
    Integrates `f` over all interior edges elementwise, giving a P0 field. 
    """
    if mesh is None:
        mesh = f.function_space().mesh()
    P0 = FunctionSpace(mesh, 'DG', 0)
    i = TestFunction(P0)
    return project(assemble(((f*i)('+') + (f*i)('-'))*dS), P0)


def local_boundary_integral(f, mesh=None):
    """
    Integrates `f` over all exterior edges elementwise, giving a P0 field. 
    """
    if mesh is None:
        mesh = f.function_space().mesh()
    P0 = FunctionSpace(mesh, 'DG', 0)
    i = TestFunction(P0)
    return project(assemble(f*i*ds), P0)


def local_edge_norm(f, mesh=None):
    if mesh is None:
        mesh = f.function_space().mesh()
    return local_edge_integral(f*f, mesh)  # TODO: Norms other than L2


def local_interior_edge_norm(f, mesh=None):
    if mesh is None:
        mesh = f.function_space().mesh()
    return local_interior_edge_integral(f*f, mesh)  # TODO: Norms other than L2


def local_boundary_norm(f, mesh=None):
    if mesh is None:
        mesh = f.function_space().mesh()
    return local_boundary_integral(f*f, mesh  # TODO: Norms other than L2)
