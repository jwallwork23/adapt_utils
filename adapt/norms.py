from firedrake import *


__all__ = ["local_norm"]


def local_norm(f, norm_type='L2'):
    """
    Calculate the `norm_type`-norm of `f` separately on each element of the mesh.
    """
    typ = norm_type.lower()
    mesh = f.function_space().mesh()
    v = TestFunction(FunctionSpace(mesh, "DG", 0))

    if isinstance(f, FiredrakeFunction):
        if typ == 'l2':
            form = v * inner(f, f) * dx
        elif typ == 'h1':
            form = v * (inner(f, f) * dx + inner(grad(f), grad(f))) * dx
        elif typ == "hdiv":
            form = v * (inner(f, f) * dx + div(f) * div(f)) * dx
        elif typ == "hcurl":
            form = v * (inner(f, f) * dx + inner(curl(f), curl(f))) * dx
        else:
            raise RuntimeError("Unknown norm type '%s'" % norm_type)
    else:
        if typ == 'l2':
            form = v * sum(inner(fi, fi) for fi in f) * dx
        elif typ == 'h1':
            form = v * sum(inner(fi, fi) * dx + inner(grad(fi), grad(fi)) for fi in f) * dx
        elif typ == "hdiv":
            form = v * sum(inner(fi, fi) * dx + div(fi) * div(fi) for fi in f) * dx
        elif typ == "hcurl":
            form = v * sum(inner(fi, fi) * dx + inner(curl(fi), curl(fi)) for fi in f) * dx
        else:
            raise RuntimeError("Unknown norm type '%s'" % norm_type)
    return sqrt(assemble(form))


def local_edge_norm(f, mesh=None, flux_jump=False):
    """
    Integrates `f` over all interior edges elementwise, giving a P0 field. 
    """
    if mesh is None:
        mesh = f.function_space().mesh()
    P0 = FunctionSpace(mesh, 'DG', 0)
    edge_function = Function(P0)
    v = TestFunction(P0)
    if flux_jump:
        edge_function = project(assemble(jump(f, FacetNormal(mesh)) * avg(v) * dS), P0)
    else:
        edge_function = project(assemble(0.5 * (f('+') * v('+') + f('-') * v('-')) * dS), P0)

    return edge_function


def local_boundary_norm(f, mesh=None):
    """
    Integrates `f` over all exterior edges elementwise, giving a P0 field. 
    """
    if mesh is None:
        mesh = f.function_space().mesh()
    P0 = FunctionSpace(mesh, 'DG', 0)
    boundary_function = Function(P0)
    v = TestFunction(P0)
    boundary_function.interpolate(assemble(f * v * ds))

    return boundary_function
