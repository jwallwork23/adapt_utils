from firedrake import *

import numpy as np


__all__ = ["lp_norm", "total_variation", "local_norm", "frobenius_norm", "local_frobenius_norm",
           "local_edge_integral", "local_interior_edge_integral", "local_boundary_integral"]


def lp_norm(f, p=2):
    """
    Calculate the lp norm of a 1D array `f`, where `p` is either `None` or `'infty'` (denoting the
    infinity norm), or `p >= 1`.
    """
    if p is None or 'inf' in p:
        return f.max()
    elif p.startswith('l'):
        p = float(p[1:])
        try:
            assert p >= 1
        except AssertionError:
            raise ValueError("Norm type l{:} not recognised.".format(p))
        return pow(np.sum(pow(np.abs(fi), p) for fi in f), 1/p)
    else:
        raise ValueError("Norm type {:} not recognised.".format(p))

def total_variation(f):
    """Calculate the total variation of a 1D array f."""
    n, tv, i0 = len(f), 0.0, 0
    sign_ = np.sign(f[1] - f[i0])
    for i in range(2, n):
        sign = np.sign(f[i] - f[i-1])
        if sign != sign_:
            tv += np.abs(f[i-1] - f[i0])
            i0 = i-1
        if i == n-1:
            tv += np.abs(f[i] - f[i0])
        sign_ = sign
    return tv

def local_norm(f, norm_type='L2'):
    """Calculate the `norm_type`-norm of `f` separately on each element of the mesh."""
    typ = norm_type.lower()
    mesh = f.function_space().mesh()
    i = TestFunction(FunctionSpace(mesh, "DG", 0))
    p = 2
    if typ.startswith('l'):
        try:
            p = int(typ[1:])
            if p < 1:
                raise ValueError
        except ValueError:
            raise ValueError("Don't know how to interpret {:s}-norm.".format(norm_type))
    elif typ not in ('h1', 'hdiv', 'hcurl'):
        raise RuntimeError("Unknown norm type '{:s}'".format(norm_type))

    if isinstance(f, Function):
        if typ == 'h1':
            form = i*inner(f, f)*dx + i*inner(grad(f), grad(f))*dx
        elif typ == 'hdiv':
            form = i*inner(f, f)*dx + i*div(f)*div(f)*dx
        elif typ == 'hcurl':
            form = i*inner(f, f)*dx + i*inner(curl(f), curl(f))*dx
        else:
            expr = inner(f, f)
            form = i*(expr**(p/2))*dx
    else:
        if typ == 'h1':
            form = i*sum(inner(fi, fi)*dx + inner(grad(fi), grad(fi)) for fi in f)*dx
        elif typ == 'hdiv':
            form = i*sum(inner(fi, fi)*dx + div(fi) * div(fi) for fi in f)*dx
        elif typ == 'hcurl':
            form = i*sum(inner(fi, fi)*dx + inner(curl(fi), curl(fi)) for fi in f)*dx
        else:
            expr = sum(inner(fi, fi) for fi in f)
            form = i*(expr**(p/2))*dx
    return assemble(form)**(1/p)

def frobenius_norm(matrix, mesh=None):
    """Calculate the Frobenius norm of `matrix`."""
    mesh = mesh or matrix.function_space().mesh()
    dim = mesh.topological_dimension()
    f = 0
    for i in range(dim):
        for j in range(dim):
            f += matrix[i, j]*matrix[i, j]
    return sqrt(assemble(f*dx))

def local_frobenius_norm(matrix, mesh=None, space=None):
    """Calculate the Frobenius norm of `matrix` separately on each element of the mesh."""
    mesh = mesh or matrix.function_space().mesh()
    space = space or FunctionSpace(mesh, "DG", 0)
    dim = mesh.topological_dimension()
    f = 0
    for i in range(dim):
        for j in range(dim):
            f += matrix[i, j]*matrix[i, j]
    return sqrt(assemble(TestFunction(space)*f*dx))

def local_edge_integral(f, mesh=None):
    """Integrate `f` over all edges elementwise, giving a P0 field."""
    mesh = mesh or f.function_space().mesh()
    P0 = FunctionSpace(mesh, 'DG', 0)
    test, trial, integral = TestFunction(P0), TrialFunction(P0), Function(P0)
    solve(test*trial*dx == ((test*f)('+') + (test*f)('-'))*dS + test*f*ds, integral)
    return integral

def local_interior_edge_integral(f, mesh=None):
    """Integrate `f` over all interior edges elementwise, giving a P0 field."""
    mesh = mesh or f.function_space().mesh()
    P0 = FunctionSpace(mesh, 'DG', 0)
    test, trial, integral = TestFunction(P0), TrialFunction(P0), Function(P0)
    solve(test*trial*dx == ((test*f)('+') + (test*f)('-'))*dS, integral)
    return integral

def local_boundary_integral(f, mesh=None):
    """Integrate `f` over all exterior edges elementwise, giving a P0 field."""
    mesh = mesh or f.function_space().mesh()
    P0 = FunctionSpace(mesh, 'DG', 0)
    test, trial, integral = TestFunction(P0), TrialFunction(P0), Function(P0)
    solve(test*trial*dx == test*f*ds, integral)
    return integral
