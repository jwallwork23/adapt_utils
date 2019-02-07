from thetis import *


__all__ = ["interp", "mixed_pair_interp"]


def interp(mesh, *fields):
    """
    Transfer solution fields from the old mesh to the new mesh. Based around the function
    ``transfer_solution`` by Nicolas Barral, 2017. The main modification is to manually 'push'
    vertices back into the domain when they are located either outside the domain, or on the domain
    boundary but considered as an interior node. This is achieved by sequentially increasing the
    tolerance involved involved in point evaluation.

    :arg mesh: new mesh onto which fields are to be interpolated.
    :arg fields: tuple of functions defined on the old mesh that one wants to transfer
    :return: interpolated fields.
    """
    try:
        assert mesh._topological_dimension == 2
    except:
        raise NotImplementedError('3D implementation not yet considered.')
    fields_new = ()
    for f in fields:
        element = f.function_space().ufl_element()
        family = element.family()
        degree = element.degree()
        f_new = Function(FunctionSpace(mesh, element), name=f.dat.name)
        notInDomain = []
        if family == 'Lagrange' and degree == 1:
            coords = mesh.coordinates.dat.data      # Vertex/node coords
        elif family in ('Lagrange', 'Discontinuous Lagrange'):
            C = VectorFunctionSpace(mesh, family, degree)
            interp_coordinates = Function(C).interpolate(mesh.coordinates)
            coords = interp_coordinates.dat.data    # Node coords (NOT just vertices)
        else:
            raise NotImplementedError('Interpolation not currently supported on requested field type.')

        # Establish which vertices fall outside the domain
        # TODO: Figure out how to do this in a less hacky way
        for x in range(len(coords)):  # This looping is unnecessary in some cases
            try:
                val = f.at(coords[x])
            except PointNotInDomainError:
                # print("#### DEBUG: offending coordinates = ", coords[x])
                val = None
                notInDomain.append(x)
            finally:
                f_new.dat.data[x] = val
        eps = 1e-6   # Tolerance to be increased
        while len(notInDomain) > 0:
            eps *= 10
            for x in notInDomain:
                try:
                    val = f.at(coords[x], tolerance=eps)
                except PointNotInDomainError(mesh, coords[x]):
                    val = None
                finally:
                    f_new.dat.data[x] = val
                    notInDomain.remove(x)
            if eps >= 1e8:
                raise PointNotInDomainError(mesh, notInDomain)
        fields_new += (f_new,)

    if len(fields_new) == 1:
        return fields_new[0]
    else:
        return fields_new


def mixed_pair_interp(mesh, V, *fields):
    """
    Interpolate mixed function space pairs onto a new mesh.

    :arg mesh: new mesh to be interpolated onto.
    :arg V: mixed function space defined on new mesh.
    :arg fields: (mixed function space) fields to be interpolated.
    :return: interpolated function pair(s).
    """
    fields_new = ()
    for q in fields:
        p = Function(V)
        p0, p1 = p.split()
        q0, q1 = q.split()
        q0, q1 = interp(mesh, q0, q1)
        p0.assign(q0), p1.assign(q1)
        fields_new += (p,)
    if len(fields_new) == 1:
        return fields_new[0]
    else:
        return fields_new
