from thetis import *

import os


__all__ = ["initialise_fields", "export_final_state"]


def initialise_fields(mesh, input_dir, name='bathymetry'):
    """
    Initialise simulation with results from a previous simulation.

    :arg mesh: field will be defined in a P1 space on this mesh.
    :arg input_dir: directory to read the data from.
    :kwarg name: field name used in the data file.
    """
    fs = FunctionSpace(mesh, 'CG', 1)  # TODO: Have fs as input, rather than mesh
    with timed_stage('initialising {:s}'.format(name)):
        f = Function(fs, name=name)
        with DumbCheckpoint(os.path.join(input_dir, name), mode=FILE_READ) as chk:
            chk.load(f)
    return f


def export_final_state(input_dir, f, name='bathymetry', fname='bathout', plexname='myplex'):
    """
    Export fields to be used in a subsequent simulation.

    :arg input_dir: directory to save the data to.
    :arg f: field to be stored.
    :kwarg name: field name to be used in the data file.
    :kwarg fname: file name to be used for the data file.
    :kwarg plexname: file name to be used for the DMPlex data file.
    """
    if not os.path.exists(input_dir):
        os.makedirs(input_dir)
    print_output("Exporting fields for subsequent simulation")

    # Create checkpoint to HDF5
    with DumbCheckpoint(os.path.join(input_dir, name), mode=FILE_CREATE) as chk:
        chk.store(f, name=name)
    File(os.path.join(input_dir, filename + '.pvd')).write(f)

    # Save DMPlex
    mesh = f.function_space().mesh()
    try:
        plex = mesh._topology_dm
    except AttributeError:
        plex = mesh._plex  # Backwards compatability
    viewer = PETSc.Viewer().createHDF5(os.path.join(input_dir, plexname + '.h5'), 'w')
    viewer(plex)
