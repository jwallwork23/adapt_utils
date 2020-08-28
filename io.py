from thetis import *
from firedrake.petsc import PETSc

import os

from adapt_utils.unsteady.options import CoupledOptions


__all__ = ["save_mesh", "load_mesh", "initialise_bathymetry", "initialise_hydrodynamics",
           "export_bathymetry", "export_hydrodynamics"]


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


def initialise_bathymetry(mesh, fpath, outputdir=None, op=CoupledOptions()):
    """
    Initialise bathymetry field with results from a previous simulation.

    :arg mesh: field will be defined in finite element space on this mesh.
    :arg fpath: directory to read the data from.
    """
    # TODO: Would be nice to have consistency: here mesh is an arg but below it is read from file
    fs = FunctionSpace(mesh, op.bathymetry_family.upper(), 1)
    with timed_stage('initialising bathymetry'):
        bathymetry = Function(fs, name='bathymetry')
        with DumbCheckpoint(os.path.join(fpath, 'bathymetry'), mode=FILE_READ) as chk:
            chk.load(bathymetry)

    # Plot to .pvd
    if outputdir is not None and op.plot_pvd:
        File(os.path.join(outputdir, "bathymetry_imported.pvd")).write(bathymetry)
    return bathymetry


def initialise_hydrodynamics(inputdir, outputdir=None, plexname='myplex', op=CoupledOptions()):
    """
    Initialise velocity and elevation with results from a previous simulation.

    :arg inputdir: directory to read the data from.
    :kwarg inputdir: directory to optionally plot the data in .pvd format.
    :kwarg plexname: file name used for the DMPlex data file.
    """
    with timed_stage('mesh'):
        mesh = load_mesh(plexname, inputdir)

    # Get finite element space
    if op.family == 'dg-dg':
        uv_element = ("DG", 1)
        elev_element = ("DG", 1)
    elif op.family == 'dg-cg':
        uv_element = ("DG", 1)
        elev_element = ("CG", 2)
    elif op.family == 'cg-cg':
        uv_element = ("CG", 2)
        elev_element = ("CG", 1)

    # Velocity
    U = VectorFunctionSpace(mesh, *uv_element)
    with timed_stage('initialising velocity'):
        with DumbCheckpoint(os.path.join(inputdir, "velocity"), mode=FILE_READ) as chk:
            uv_init = Function(U, name="velocity")
            chk.load(uv_init)

    # Elevation
    H = FunctionSpace(mesh, *elev_element)
    with timed_stage('initialising elevation'):
        with DumbCheckpoint(os.path.join(inputdir, "elevation"), mode=FILE_READ) as chk:
            elev_init = Function(H, name="elevation")
            chk.load(elev_init)

    # Plot to .pvd
    if outputdir is not None and op.plot_pvd:
        File(os.path.join(outputdir, "velocity_imported.pvd")).write(uv_init)
        File(os.path.join(outputdir, "elevation_imported.pvd")).write(elev_init)
    return elev_init, uv_init  # TODO: Consistent ordering


def export_bathymetry(bathymetry, fpath, plexname='myplex', op=CoupledOptions()):
    """
    Export bathymetry field to be used in a subsequent simulation.

    :arg bathymetry: field to be stored.
    :arg fpath: directory to save the data to.
    :kwarg plexname: file name to be used for the DMPlex data file.
    :kwarg op: Options parameter class.
    """
    if not os.path.exists(fpath):
        os.makedirs(fpath)
    print_output("Exporting fields for subsequent simulation")

    # Create checkpoint to HDF5
    with DumbCheckpoint(os.path.join(fpath, 'bathymetry'), mode=FILE_CREATE) as chk:
        chk.store(bathymetry, name='bathymetry')
    if op.plot_pvd:
        File(os.path.join(fpath, 'bathout.pvd')).write(bathymetry)

    # Save mesh to DMPlex format
    if plexname is not None:
        save_mesh(bathymetry.function_space().mesh(), plexname, fpath)


def export_hydrodynamics(uv, elev, fpath, plexname='myplex', op=CoupledOptions()):
    """
    Export velocity and elevation to be used in a subsequent simulation

    :arg uv: velocity field to be stored.
    :arg elev: elevation field to be stored.
    :arg fpath: directory to save the data to.
    :kwarg plexname: file name to be used for the DMPlex data file.
    :kwarg op: Options parameter class.
    """
    if not os.path.exists(fpath):
        os.makedirs(fpath)
    print_output("Exporting fields for subsequent simulation")

    # Export velocity
    with DumbCheckpoint(os.path.join(fpath, "velocity"), mode=FILE_CREATE) as chk:
        chk.store(uv, name="velocity")

    # Export elevation
    with DumbCheckpoint(os.path.join(fpath, "elevation"), mode=FILE_CREATE) as chk:
        chk.store(elev, name="elevation")

    # Plot to .pvd
    if op.plot_pvd:
        File(os.path.join(fpath, 'velocityout.pvd')).write(uv)
        File(os.path.join(fpath, 'elevationout.pvd')).write(elev)

    # Export mesh
    if plexname is not None:
        mesh = elev.function_space().mesh()
        assert mesh == uv.function_space().mesh()
        save_mesh(mesh, plexname, fpath)
