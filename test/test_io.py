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


@pytest.fixture(params=['cg', 'dg'])
def family(request):
    return request.param


@pytest.fixture(params=['dg-dg', 'dg-cg', 'cg-cg'])
def pair(request):
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


def test_bathymetry_io(family):
    op = CoupledOptions(bathymetry_family=family)

    # Create some bathymetry field
    mesh = get_mesh(2)
    x, y = SpatialCoordinate(mesh)
    P1 = get_function_space(mesh, 'scalar', family.upper(), 1)
    b = interpolate(x+1, P1)

    # Save it to file
    fname = 'bathymetry'
    plexname = 'myplex'
    fpath = create_directory('tmp')
    export_bathymetry(b, fpath, plexname=plexname, op=op)

    # Read from the file and check consistency
    b_new = initialise_bathymetry(mesh, fpath, outputdir=fpath, op=op)  # TODO: Make consistent by reading from file
    assert np.allclose(b.dat.data, b_new.dat.data)

    # Clean up
    os.remove(os.path.join(fpath, fname + '.h5'))
    os.remove(os.path.join(fpath, plexname + '.h5'))
    os.remove(os.path.join(fpath, 'bathout.pvd'))
    os.remove(os.path.join(fpath, 'bathout_0.vtu'))
    os.remove(os.path.join(fpath, 'bathymetry_imported.pvd'))
    os.remove(os.path.join(fpath, 'bathymetry_imported_0.vtu'))
    os.rmdir(fpath)


def test_hydro_io(pair):
    element_pair = {
        'dg-dg': (('DG', 1), ('DG', 1)),
        'dg-cg': (('DG', 1), ('CG', 2)),
        'cg-cg': (('CG', 2), ('CG', 1)),
    }[pair]
    op = CoupledOptions(family=pair)

    # Create some hydrodynamics fields
    mesh = get_mesh(2)
    x, y = SpatialCoordinate(mesh)
    U = get_function_space(mesh, 'vector', *element_pair[0])
    H = get_function_space(mesh, 'scalar', *element_pair[1])
    uv = interpolate(as_vector([x, y]), U)
    elev = interpolate(x*y, H)

    # Save them to file
    fnames = ('velocity', 'elevation')
    plexname = 'myplex'
    fpath = create_directory('tmp')
    export_hydrodynamics(uv, elev, fpath, plexname=plexname, op=op)

    # Read from the file and check consistency
    uv_new, elev_new = initialise_hydrodynamics(fpath, outputdir=fpath, op=op)
    assert np.allclose(uv.dat.data, uv_new.dat.data)
    assert np.allclose(elev.dat.data, elev_new.dat.data)

    # Clean up
    for fname in fnames:
        os.remove(os.path.join(fpath, fname + '.h5'))
        os.remove(os.path.join(fpath, '{:s}out.pvd'.format(fname)))
        os.remove(os.path.join(fpath, '{:s}out_0.vtu'.format(fname)))
        os.remove(os.path.join(fpath, '{:s}_imported.pvd'.format(fname)))
        os.remove(os.path.join(fpath, '{:s}_imported_0.vtu'.format(fname)))
    os.remove(os.path.join(fpath, plexname + '.h5'))
    os.rmdir(fpath)
