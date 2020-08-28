from thetis import *
from firedrake.petsc import PETSc

import os


__all__ = ["save_mesh", "load_mesh", "initialise_bathymetry", "export_bathymetry"]


def save_mesh(mesh, fname, fpath):
    """
    :arg mesh: mesh to be saved in DMPlex format.
    :arg fname: file name (without '.h5' extension).
    :arg fpath: directory to store the file.
    """
    if COMM_WORLD.size > 1:
        raise IOError("Saving a mesh to HDF5 only works in serial.")
    try:
        plex = mesh._topology_dm
    except AttributeError:
        plex = mesh._plex  # Backwards compatability
    viewer = PETSc.Viewer().createHDF5(os.path.join(fpath, fname + '.h5'), 'w')
    viewer(plex)


def load_mesh(fname, fpath):
    """
    :arg fname: file name (without '.h5' extension).
    :arg fpath: directory where the file is stored.
    :return: mesh loaded from DMPlex format.
    """
    if COMM_WORLD.size > 1:
        raise IOError("Loading a mesh from HDF5 only works in serial.")
    newplex = PETSc.DMPlex().create()
    newplex.createFromFile(os.path.join(fpath, fname + '.h5'))
    return Mesh(newplex)


def initialise_bathymetry(mesh, fpath, family='CG', degree=1):
    """
    Initialise bathymetry field with results from a previous simulation.

    :arg mesh: field will be defined in finite element space on this mesh.
    :arg fpath: directory to read the data from.
    :kwarg family: finite element family to use for bathymetry space.
    :kwarg degree: finite element degree to use for bathymetry space.
    """
    fname = 'bathymetry'  # File name
    name = 'bathymetry'   # Name used in metadata
    fs = FunctionSpace(mesh, family, degree)
    with timed_stage('initialising {:s}'.format(name)):
        f = Function(fs, name=name)
        with DumbCheckpoint(os.path.join(fpath, fname), mode=FILE_READ) as chk:
            chk.load(f)
    return f


def export_bathymetry(bathymetry, fpath, plexname=None, plot_pvd=False):
    """
    Export bathymetry field to be used in a subsequent simulation.

    :arg bathymetry: field to be stored.
    :arg fpath: directory to save the data to.
    :kwarg plexname: file name to be used for the DMPlex data file.
    """
    fname = 'bathymetry'  # File name
    name = 'bathymetry'   # Name used in field metadata
    if not os.path.exists(fpath):
        os.makedirs(fpath)
    print_output("Exporting fields for subsequent simulation")

    # Create checkpoint to HDF5
    with DumbCheckpoint(os.path.join(fpath, fname), mode=FILE_CREATE) as chk:
        chk.store(bathymetry, name=name)
    if plot_pvd:
        File(os.path.join(fpath, 'bathout.pvd')).write(bathymetry)

    # Save mesh to DMPlex format
    if plexname is not None:
        save_mesh(bathymetry.function_space().mesh(), plexname, fpath)
