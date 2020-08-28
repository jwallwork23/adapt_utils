from thetis import *

import numpy as np
import os
import pytest

from adapt_utils.io import *


def get_mesh(dim, n=4):
    if dim == 1:
        return UnitIntervalMesh(n)
    elif dim == 2:
        return UnitSquareMesh(n, n)
    elif dim == 3:
        return UnitCubeMesh(n, n, n)
    else:
        raise ValueError("Expected dimension 1, 2 or 3 but got {:d}".format(dim))


# ---------------------------
# standard tests for pytest
# ---------------------------

# @pytest.fixture(params=[1, 2, 3])  # FIXME
@pytest.fixture(params=[2, 3])
def dim(request):
    return request.param


def test_save_load(dim):
    mesh = get_mesh(dim)
    coords = mesh.coordinates.dat.data.copy()
    fname = 'mymesh'
    fpath = create_directory('tmp')
    save_mesh(mesh, fname, fpath)
    newmesh = load_mesh(fname, fpath)
    assert np.allclose(coords, newmesh.coordinates.dat.data)
    os.remove(os.path.join(fpath, fname + '.h5'))
    os.rmdir(fpath)
