from firedrake import *
import firedrake.supermeshing as supermesh


__all__ = ["supermesh_project", "point_interpolate", "adjoint_supermesh_project"]


def supermesh_project(src, tgt, check_mass=False, mixed_mass_matrix=None, solver=None):
    """
    Hand-coded supermesh projection.

    :arg src: source field.
    :arg tgt: target field.
    """
    source_space = src.function_space()
    target_space = tgt.function_space()

    # Step 1: Form the RHS:
    #    rhs := Mst * src
    Mst = mixed_mass_matrix or supermesh.assemble_mixed_mass_matrix(source_space, target_space)
    with tgt.dat.vec_ro as vt:
        rhs = vt.copy()
    with src.dat.vec_ro as vs:
        Mst.mult(vs, rhs)

    # Step 2: Solve the linear system for the target:
    #    Mt * tgt = rhs
    ksp = solver or PETSc.KSP().create()
    if solver is None:
        Mt = assemble(inner(TrialFunction(target_space), TestFunction(target_space))*dx).M.handle
        ksp.setOperators(Mt)
        ksp.setFromOptions()
    with tgt.dat.vec as vt:
        ksp.solve(rhs, vt)

    if check_mass:
        assert np.allclose(assemble(src*dx), assemble(tgt*dx))
    return tgt


def point_interpolate(src, tgt, tol=1.0e-10):
    """
    Hand-coded point interpolation operator.

    :arg src: source field.
    :arg tgt: target field.
    """
    try:
        assert src.ufl_element().family() == 'Lagrange'
        assert src.ufl_element().degree() == 1
        assert tgt.ufl_element().family() == 'Lagrange'
        assert tgt.ufl_element().degree() == 1
    except AssertionError:
        raise NotImplementedError
    if not hasattr(tgt, 'function_space'):
        tgt = Function(tgt)
    mesh = tgt.function_space().mesh()
    target_coords = mesh.coordinates.dat.data
    for i in range(mesh.num_vertices()):
        tgt.dat.data[i] = src.at(target_coords[i], tolerance=tol)
    return tgt


def adjoint_supermesh_project(tgt_b, src_b, mixed_mass_matrix=None, solver=None):
    """
    Hand-coded adjoint of a supermesh projection.

    :arg tgt_b: seed vector in target space.
    :arg src_b: adjoint supermesh projection into source space.
    """
    source_space = src_b.function_space()
    target_space = tgt_b.function_space()

    # Adjoint of step 2: Solve the linear system for the target:
    #    Mt^T * sol = tgt_b
    ksp = solver or PETSc.KSP().create()
    if solver is None:
        Mt = assemble(inner(TrialFunction(target_space), TestFunction(target_space))*dx).M.handle
        ksp.setOperators(Mt.transpose())
        ksp.setFromOptions()
    with tgt_b.dat.vec_ro as rhs:
        sol = rhs.copy()
        ksp.solve(rhs, sol)

    # Adjoint of 1: Multiply with the tranpose mass matrix
    #    src_b := Mst^T * sol
    Mst = mixed_mass_matrix or supermesh.assemble_mixed_mass_matrix(source_space, target_space)
    with src_b.dat.vec_ro as vs:
        Mst.multTranspose(sol, vs)
    return src_b
