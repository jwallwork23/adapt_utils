from thetis import *

import numpy as np
import os
import pytest

from adapt_utils.io import *
from adapt_utils.unsteady.options import CoupledOptions


def get_mesh(dim, n=4):
    if dim == 1:
        return UnitIntervalMesh(n)
    elif dim == 2:
        return UnitSquareMesh(n, n)
    elif dim == 3:
        return UnitCubeMesh(n, n, n)
    else:
        raise ValueError("Expected dimension 1, 2 or 3 but got {:d}".format(dim))


def get_function_space(mesh, shape, family, degree):
    if shape == 'scalar':
        constructor = FunctionSpace
    elif shape == 'vector':
        constructor = VectorFunctionSpace
    elif shape == 'tensor':
        constructor = TensorFunctionSpace
    else:
        raise ValueError("Shape {:s} not recognised.".format(shape))
    return constructor(mesh, family, degree)


# ---------------------------
# standard tests for pytest
# ---------------------------

# @pytest.fixture(params=[1, 2, 3])  # FIXME
@pytest.fixture(params=[2, 3])
def dim(request):
    return request.param


def test_mesh_io(dim):

    # Create some mesh
    mesh = get_mesh(dim)

    # Save it to file
    fname = 'myplex'
    fpath = create_directory('tmp')
    save_mesh(mesh, fname, fpath)

    # Read from the file and check consistency
    newmesh = load_mesh(fname, fpath)
    assert np.allclose(mesh.coordinates.dat.data, newmesh.coordinates.dat.data)

    # Clean up
    os.remove(os.path.join(fpath, fname + '.h5'))
    os.rmdir(fpath)


def test_bathymetry_io():

    # Create some bathymetry field
    mesh = get_mesh(2)
    x, y = SpatialCoordinate(mesh)
    P1 = get_function_space(mesh, 'scalar', 'CG', 1)
    b = interpolate(x+1, P1)

    # Save it to file
    fname = 'bathymetry'
    plexname = 'myplex'
    fpath = create_directory('tmp')
    export_bathymetry(b, fpath, plexname=plexname)

    # Read from the file and check consistency
    b_new = initialise_bathymetry(mesh, fpath)  # TODO: Make consistent by reading from plex file
    assert np.allclose(b.dat.data, b_new.dat.data)

    # Clean up
    os.remove(os.path.join(fpath, fname + '.h5'))
    os.remove(os.path.join(fpath, plexname + '.h5'))
    os.remove(os.path.join(fpath, 'bathout.pvd'))
    os.remove(os.path.join(fpath, 'bathout_0.vtu'))
    os.rmdir(fpath)
