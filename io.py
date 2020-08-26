from thetis import *

import os


__all__ = ["save_mesh", "load_mesh", "initialise_fields", "export_final_state"]


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
    newplex.createFromFile(os.path.join(fpath, fname))
    return Mesh(newplex)


def initialise_fields(mesh, fpath, fname='bathymetry', name='bathymetry'):
    """
    Initialise simulation with results from a previous simulation.

    :arg mesh: field will be defined in a P1 space on this mesh.
    :arg fpath: directory to read the data from.
    :arg fname: file name (without '.h5' extension).
    :kwarg name: field name used in the data file.
    """
    fs = FunctionSpace(mesh, 'CG', 1)  # TODO: Have fs as input, rather than mesh
    with timed_stage('initialising {:s}'.format(name)):
        f = Function(fs, name=name)
        with DumbCheckpoint(os.path.join(fpath, fname), mode=FILE_READ) as chk:
            chk.load(f)
    return f


def export_final_state(fpath, f, fname='bathymetry', name='bathymetry', plexname='myplex'):
    """
    Export fields to be used in a subsequent simulation.

    :arg fpath: directory to save the data to.
    :arg f: field to be stored.
    :kwarg fname: file name to be used for the data file.
    :kwarg name: field name to be used in the data file.
    :kwarg plexname: file name to be used for the DMPlex data file.
    """
    if not os.path.exists(fpath):
        os.makedirs(fpath)
    print_output("Exporting fields for subsequent simulation")

    # Create checkpoint to HDF5
    with DumbCheckpoint(os.path.join(fpath, fname), mode=FILE_CREATE) as chk:
        chk.store(f, name=name)
    File(os.path.join(fpath, 'bathout.pvd')).write(f)  # TODO: Remove / make optional

    # Save mesh to DMPlex format
    save_mesh(f.function_space().mesh(), plexname, fpath)
