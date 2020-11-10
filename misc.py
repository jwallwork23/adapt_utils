from thetis import *

import fnmatch
import os

from adapt_utils.linalg import rotate


__all__ = ["prod", "combine", "integrate_boundary", "copy_mesh", "get_boundary_nodes",
           "box", "ellipse", "bump", "circular_bump", "gaussian",
           "num_days", "find", "knownargs2dict", "unknownargs2dict"]


# --- UFL

def abs(u):
    """
    Hack due to the fact `abs` seems to be broken in conditional statements.
    """
    return conditional(u < 0, -u, u)


def prod(arr):
    """
    Helper function for taking the product of an array (similar to `sum`).
    """
    n = len(arr)
    if n == 0:
        raise ValueError
    elif n == 1:
        return arr[0]
    else:
        return arr[0]*prod(arr[1:])


def combine(operator, *args):
    """
    Helper function for repeatedly application of binary operators.
    """
    n = len(args)
    if n == 0:
        raise ValueError
    elif n == 1:
        return args[0]
    else:
        return operator(args[0], combine(operator, *args[1:]))


# --- Utility functions

def box(locs, mesh, scale=1.0, rotation=None):
    r"""
    Rectangular indicator functions associated with a list of region of interest tuples.

    Takes the value `scale` in the region

  ..math::
        (|x - x0| < r_x) && (|y - y0| < r_y)

    centred about (x0, y0) and zero elsewhere. Similarly for other dimensions.

    :kwarg scale: scale factor for indicator.
    :kwarg rotation: angle by which to rotate.
    """
    d = mesh.topological_dimension()
    dims, L = range(d), range(len(locs))  # Itersets
    x = SpatialCoordinate(mesh)

    # Get distances from origins and RHS values
    X = [[x[i] - locs[j][i] for i in dims] for j in L]
    r = [[locs[j][d] if len(locs[j]) == d+1 else locs[j][d+i] for i in dims] for j in L]

    # Apply rotations
    if rotation is not None:
        rotate(X, rotation)

    # Combine to get indicator
    expr = [combine(And, *[lt(abs(X[j][i]), r[j][i]) for i in dims]) for j in L]
    return conditional(combine(Or, *expr), scale, 0.0)


def ellipse(locs, mesh, scale=1.0, rotation=None):
    r"""
    Ellipse indicator function associated with a list of region of interest tuples.

    Takes the value `scale` in the region

  ..math::
        (x - x_0)^2/r_x^2 + (y - y_0)^2/r_y^2 < 1

    and zero elsewhere. Similarly for other dimensions.

    :kwarg scale: scale factor for indicator.
    :kwarg rotation: angle by which to rotate.
    """
    d = mesh.topological_dimension()
    dims, L = range(d), range(len(locs))  # Itersets
    x = SpatialCoordinate(mesh)

    # Get distances from origins and RHS values
    X = [[x[i] - locs[j][i] for i in dims] for j in L]
    r = [[locs[j][d] if len(locs[j]) == d+1 else locs[j][d+i] for i in dims] for j in L]

    # Apply rotations
    if rotation is not None:
        rotate(X, rotation)

    # Combine to get indicator
    conditions = [lt(sum((X[j][i]/r[j][i])**2 for i in dims), 1) for j in L]
    return conditional(combine(Or, *conditions), scale, 0)


def bump(locs, mesh, scale=1.0, rotation=None):
    r"""
    Rectangular bump function associated with a list of region of interest tuples.
    (A smooth approximation to the box function.)

    Takes the form

  ..math::
        \exp\left(1 - \frac1{\left(1 - \left(\frac{x - x_0}{r_x}\right)^2\right)}\right)
        * \exp\left(1 - \frac1{\left(1 - \left(\frac{y - y_0}{r_y}\right)^2\right)}\right)

    scaled by `scale` inside the box region. Similarly for other dimensions.

    Note that we assume the provided regions are disjoint for this indicator.

    :kwarg scale: scale factor for indicator.
    :kwarg rotation: angle by which to rotate.
    """
    d = mesh.topological_dimension()
    dims, L = range(d), range(len(locs))  # Itersets
    x = SpatialCoordinate(mesh)

    # Get distances from origins and RHS values
    X = [[x[i] - locs[j][i] for i in dims] for j in L]
    r = [[locs[j][d] if len(locs[j]) == d+1 else locs[j][d+i] for i in dims] for j in L]

    # Apply rotations
    if rotation is not None:
        rotate(X, rotation)

    # Combine to get indicator
    q = [[(X[j][i]/r[j][i])**2 for i in dims] for j in L]  # Quotients of squared distances
    conditions = [combine(And, *[lt(q[j][i], 1) for i in dims]) for j in L]
    bumps = [prod([exp(1 - 1/(1 - q[j][i])) for i in dims]) for j in L]
    return sum([conditional(conditions[j], scale*bumps[j], 0) for j in L])


# TODO: Elliptical bump
def circular_bump(locs, mesh, scale=1.0, rotation=None):
    r"""
    Circular bump function associated with a list of region of interest tuples.
    (A smooth approximation to the ball function.)

    Defining the radius :math:`r^2 := (x - x_0)^2 + (y - y_0)^2`, the circular bump takes the
    form

  ..math::
        \exp\left(1 - \frac1{\left1 - \frac{r^2}{r_0^2}\right)}\right)

    scaled by `scale` inside the ball region. Similarly for other dimensions.

    :kwarg scale: scale factor for indicator.
    :kwarg rotation: angle by which to rotate.
    """
    d = mesh.topological_dimension()
    dims, L = range(d), range(len(locs))  # Itersets
    x = SpatialCoordinate(mesh)

    # Get distances from origins and RHS values
    X = [[x[i] - locs[j][i] for i in dims] for j in L]
    r_sq = [[locs[j][d]**2 if len(locs[j]) == d+1 else locs[j][d+i]**2 for i in dims] for j in L]

    # Apply rotations
    if rotation is not None:
        rotate(X, rotation)

    # Combine to get indicator
    q = [sum([X[j][i]**2 for i in dims])/sum(r_sq[j]) for j in L]  # Quotient of squared 2-norms
    return sum([conditional(lt(q[j], 1), scale*exp(1 - 1/(1 - q[j])), 0) for j in L])


def gaussian(locs, mesh, scale=1.0, rotation=None):
    r"""
    Gaussian bell associated with a list of region of interest tuples.

    Takes the form

  ..math::
        \exp\left(- \left(\frac{x^2}{r_x^2} + \frac{y^2}{r_y^2}\right)\right)

    scaled by `scale` inside the ball region. Similarly for other dimensions.

    :kwarg scale: scale factor for indicator.
    :kwarg rotation: angle by which to rotate.
    """
    d = mesh.topological_dimension()
    dims, L = range(d), range(len(locs))  # Itersets
    x = SpatialCoordinate(mesh)

    # Get distances from origins and RHS values
    X = [[x[i] - locs[j][i] for i in dims] for j in L]
    r = [[locs[j][d] if len(locs[j]) == d+1 else locs[j][d+i] for i in dims] for j in L]

    # Apply rotations
    if rotation is not None:
        rotate(X, rotation)

    # Combine to get indicator
    q_sq = [sum((X[j][i]/r[j][i])**2 for i in dims) for j in L]  # Quotient of squares
    # return sum(scale*conditional(lt(q_sq[j], 1), exp(-q_sq[j]), 0) for j in L)
    return sum(scale*exp(-q_sq[j]) for j in L)


# --- Extraction from Firedrake objects

def copy_mesh(mesh):
    """
    Deepcopy a mesh.
    """
    return Mesh(Function(mesh.coordinates))


def get_boundary_nodes(fs, segment='on_boundary'):
    """
    :arg fs: function space to get boundary nodes for.
    :kwarg segment: segment of boundary to get nodes of (default 'on_boundary').
    """
    return fs.boundary_nodes(segment, 'topological')


def integrate_boundary(mesh):
    """
    Integrates over domain boundary.

    Extension of `thetis.utility.compute_boundary_length` to
    arbitrary dimensions.
    """
    P1 = FunctionSpace(mesh, 'CG', 1)
    boundary_markers = sorted(mesh.exterior_facets.unique_markers)
    boundary_len = OrderedDict()
    for i in boundary_markers:
        ds_restricted = ds(int(i))
        one_func = Function(P1).assign(1.0)
        boundary_len[i] = assemble(one_func*ds_restricted)
    return boundary_len


# --- Non-Firedrake specific

def num_days(month, year):
    """Get the number of days in a month"""
    if month in (4, 6, 9, 11):
        return 30
    elif month in (1, 3, 5, 7, 8, 10, 12):
        return 31
    elif month == 2:
        return 29 if year % 4 == 0 else 28


def find(pattern, path):
    """
    Find all files with a specified pattern.
    """
    result = []
    for root, dirs, files in os.walk(path):
        for name in files:
            if fnmatch.fnmatch(name, pattern):
                result.append(os.path.join(root, name))
    return result


def knownargs2dict(ka):
    """
    Extract all public attributes from namespace `ka` and return as a dictionary.
    """
    out = {}
    for arg in [arg for arg in dir(ka) if arg[0] != '_']:
        attr = ka.__getattribute__(arg)
        if attr is not None:
            if attr == '1':
                out[arg] = None
            # TODO: Account for integers
            # TODO: Account for floats
            else:
                out[arg] = attr
    return out


def unknownargs2dict(ua):
    """
    Extract all public attributes from list `ua` and return as a dictionary.
    """
    out = {}
    for i in range(len(ua)//2):
        key = ua[2*i][1:]
        val = ua[2*i+1]
        if val == '1':
            out[key] = None
        # TODO: Account for integers
        # TODO: Account for floats
        else:
            out[key] = val
    return out
