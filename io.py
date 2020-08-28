from thetis import *
from firedrake.petsc import PETSc

import os


__all__ = ["save_mesh", "load_mesh", "initialise_bathymetry", "initialise_hydrodynamics",
           "export_bathymetry"]


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


def initialise_bathymetry(mesh, fpath):
    """
    Initialise bathymetry field with results from a previous simulation.

    :arg mesh: field will be defined in finite element space on this mesh.
    :arg fpath: directory to read the data from.
    """
    fs = FunctionSpace(mesh, "CG", 1)  # TODO: Avoid hard-coding
    with timed_stage('initialising {:s}'.format(name)):
        f = Function(fs, name='bathymetry')
        with DumbCheckpoint(os.path.join(fpath, 'bathymetry'), mode=FILE_READ) as chk:
            chk.load(f)
    return f


def initialise_hydrodynamics(inputdir, outputdir=None, plexname='myplex'):
    """
    Initialise velocity and elevation with results from a previous simulation.
    """
    with timed_stage('mesh'):
        mesh = load_mesh(plexname, inputdir)

    # Velocity
    U = VectorFunctionSpace(mesh, "DG", 1)  # TODO: Pass Options class to avoid hard-coding
    with timed_stage('initialising velocity'):
        with DumbCheckpoint(os.path.join(inputdir, "velocity"), mode=FILE_READ) as chk:
            uv_init = Function(U, name="velocity")
            chk.load(uv_init)

    # Elevation
    H = FunctionSpace(mesh, "DG", 1)  # TODO: Pass Options class to avoid hard-coding
    with timed_stage('initialising elevation'):
        with DumbCheckpoint(os.path.join(inputdir, "elevation"), mode=FILE_READ) as chk:
            elev_init = Function(H, name="elevation")
            chk.load(elev_init)

    # Plot to .pvd
    if outputdir is not None:
        File(os.path.join(outputdir, "velocity_imported.pvd")).write(uv_init)
        File(os.path.join(outputdir, "elevation_imported.pvd")).write(elev_init)
    return elev_init, uv_init


def export_bathymetry(bathymetry, fpath, plexname=None, plot_pvd=False):
    """
    Export bathymetry field to be used in a subsequent simulation.

    :arg bathymetry: field to be stored.
    :arg fpath: directory to save the data to.
    :kwarg plexname: file name to be used for the DMPlex data file.
    """
    if not os.path.exists(fpath):
        os.makedirs(fpath)
    print_output("Exporting fields for subsequent simulation")

    # Create checkpoint to HDF5
    with DumbCheckpoint(os.path.join(fpath, 'bathymetry'), mode=FILE_CREATE) as chk:
        chk.store(bathymetry, name='bathymetry')
    if plot_pvd:
        File(os.path.join(fpath, 'bathout.pvd')).write(bathymetry)

    # Save mesh to DMPlex format
    if plexname is not None:
        save_mesh(bathymetry.function_space().mesh(), plexname, fpath)


def export_hydrodynamics(uv, elev, inputdir, outputdir=None, plexname='myplex'):
    """
    Export fields to be used in a subsequent simulation
    """
    if not os.path.exists(inputdir):
        os.makedirs(inputdir)
    print_output("Exporting fields for subsequent simulation")

    # Export velocity
    with DumbCheckpoint(os.path.join(inputdir, "velocity"), mode=FILE_CREATE) as chk:
        chk.store(uv, name="velocity")

    # Export elevation
    with th.DumbCheckpoint(os.path.join(inputdir, "elevation"), mode=FILE_CREATE) as chk:
        chk.store(elev, name="elevation")

    if outputdir is not None:

        # Plot to .pvd
        File(os.path.join(outputdir, 'velocityout.pvd')).write(uv)
        File(os.path.join(outputdir, 'elevationout.pvd')).write(elev)

        # Export mesh
        mesh = elev.function_space().mesh()
        assert mesh == uv.function_space().mesh()
        save_mesh(mesh, plexname, outputdir)
