from firedrake import *
import numpy as np
import numpy.linalg as la


__all__ = ["get_min_angle", "sipg_parameter", "index_string", "subdomain_indicator",
           "get_boundary_nodes", "print_doc", "bessi0", "bessk0"]


def get_min_angle(mesh):
    """
    Compute the minimum angle in each mesh element.
    """
    plex = mesh._plex

    # Ensure correct section
    dim = mesh._topological_dimension
    entity_dofs = np.zeros(dim+1, dtype=np.int32)
    entity_dofs[0] = mesh.geometric_dimension()
    coordSection = dmplex.create_section(mesh, entity_dofs)
    dmCoords = plex.getCoordinateDM()
    dmCoords.setDefaultSection(coordSection)
    coords_local = dmCoords.createLocalVec()
    coords_local.array[:] = np.reshape(mesh.coordinates.dat.data_ro_with_halos, coords_local.array.shape)
    plex.setCoordinatesLocal(coords_local)

    # Loop over all cells
    cells = plex.getDepthStratum(2)
    coords = mesh.coordinates.dat.data_ro_with_halos
    min_angles = np.zeros(cells[1])
    for c in range(cells[0], cells[1]):
        local_vertices = plex.getTransitiveClosure(c)[0][4:]
        endpoints = [np.array(coords[coordSection.getOffset(v)//dim]) for v in local_vertices]
        dat = {0: {}, 1: {}, 2: {}}
        dat[0]['vector'] = endpoints[1]-endpoints[0]
        dat[0]['length'] = la.norm(dat[0]['vector'])
        dat[1]['vector'] = endpoints[2]-endpoints[1]
        dat[1]['length'] = la.norm(dat[1]['vector'])
        dat[2]['vector'] = endpoints[0]-endpoints[2]
        dat[2]['length'] = la.norm(dat[2]['vector'])
        lmin = min(dat[0]['length'], dat[1]['length'], dat[2]['length'])
        for i in dat:
            if np.abs(dat[i]['length'] - lmin) < 1e-8:
                dat.pop(i)
                break
        normalised = []
        for i in dat:
            normalised.append(dat[i]['vector']/dat[i]['length'])
        min_angles[c] = acos(np.abs(np.dot(normalised[0], normalised[1])))
    return min_angles

def sipg_parameter(mesh, nu, constant=True, p=1):
    """
    Compute SIPG parameter for a given mesh and viscosity/diffusivity :math:`nu`.

    Epshteyn and Riviere (2007). Estimation of penalty parameters for symmetric
    interior penalty Galerkin methods. Journal of Computational and Applied
    Mathematics, 206(2):843-872. http://dx.doi.org/10.1016/j.cam.2006.08.029

    :arg nu: viscosity/diffusivity of problem.
    :kwarg constant: toggle whether we want a spatially varying coefficient.
    :kwarg p: degree of function space used to solve problem.
    """
    min_angles = get_min_angle(mesh)
    assert p > 0
    if constant:
        if isinstance(nu, Constant):
            nu_max = nu.dat.data[0]
        else:
            nu_max = np.max(nu.dat.data)
        return Constant(3*p*(p+1)*nu_max/tan(np.min(min_angles)))
    sigma = Function(FunctionSpace(mesh, "DG", 0))
    sigma.interpolate(3*p*(p+1)*nu)  # FIXME: assumes viscosity constant in each element
    for i in range(len(sigma.dat.data)):
        sigma.dat.data[i] /= tan(min_angles[i])  # TODO: check numbering is consistent
    return sigma

def index_string(index):
    """
    :arg index: integer form of index.
    :return: five-digit string form of index.
    """
    return (5 - len(str(index)))*'0' + str(index)

def subdomain_indicator(mesh, subdomain_id):
    """
    Creates a P0 indicator function relating with `subdomain_id`.
    """
    return assemble(TestFunction(FunctionSpace(mesh, "DG", 0))*dx(subdomain_id))

def get_boundary_nodes(fs, segment='on_boundary'):
    """
    :arg fs: function space to get boundary nodes for.
    :kwarg segment: segment of boundary to get nodes of (default 'on_boundary').
    """
    return fs.boundary_nodes(segment, 'topological')

def print_doc(anything):
    """
    Print the docstring of any class or function.
    """
    print(anything.__doc__)

def bessi0(x):
    """
    Modified Bessel function of the first kind. Code taken from 'Numerical recipes in C'.
    """
    ax = abs(x)
    y1 = x/3.75
    y1 *= y1
    expr1 = 1.0 + y1*(3.5156229 + y1*(3.0899424 + y1*(1.2067492 + y1*(0.2659732 + y1*(0.360768e-1 + y1*0.45813e-2)))))
    y2 = 3.75/ax
    expr2 = (exp(ax)/sqrt(ax))*(0.39894228 + y2*(0.1328592e-1 + y2*(0.225319e-2 + y2*(-0.157565e-2 + y2*(0.916281e-2 + y2*(-0.2057706e-1 + y2*(0.2635537e-1 + y2*(-0.1647633e-1 + y2*0.392377e-2))))))))
    return conditional(le(ax, 3.75), expr1, expr2)

def bessk0(x):
    """
    Modified Bessel function of the second kind. Code taken from 'Numerical recipes in C'.
    """
    y1 = x*x/4.0
    expr1 = (-ln(x/2.0)*bessi0(x)) + (-0.57721566 + y1*(0.42278420 + y1*(0.23069756 + y1*(0.3488590e-1 + y1*(0.262698e-2 + y1*(0.10750e-3 + y1*0.74e-5))))))
    y2 = 2.0/x
    expr2 = (exp(-x)/sqrt(x))*(1.25331414 + y2*(-0.7832358e-1 + y2*(0.2189568e-1 + y2*(-0.1062446e-1 + y2*(0.587872e-2 + y2*(-0.251540e-2 + y2*0.53208e-3))))))
    return conditional(ge(x, 2), expr2, expr1)
