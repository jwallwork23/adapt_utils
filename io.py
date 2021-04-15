from thetis import *
from firedrake.petsc import PETSc

import datetime
import os

<<<<<<< HEAD
from adapt_utils.unsteady.options import CoupledOptions
=======
from adapt_utils.options import CoupledOptions
>>>>>>> dfe1c0b3a34dfef1765835b64b574a69fe60dd9a


__all__ = ["save_mesh", "load_mesh", "initialise_field", "export_field",
           "initialise_bathymetry", "export_bathymetry",
           "initialise_hydrodynamics", "export_hydrodynamics",
<<<<<<< HEAD
           "OuterLoopLogger", "TimeDependentAdaptationLogger", "readfile", "index_string"]
=======
           "OuterLoopLogger", "TimeDependentAdaptationLogger",
           "readfile", "index_string", "get_date"]
>>>>>>> dfe1c0b3a34dfef1765835b64b574a69fe60dd9a


def get_filename(fname, index_str):
    if index_str is None:
        return fname
    try:
        return '_'.join([fname, index_str])
    except TypeError:
        return '_'.join([fname, index_string(index_str)])


# --- Input

def load_mesh(fname, fpath='.', delete=False):
    """
    :arg fname: file name (without '.h5' extension).
    :kwarg fpath: directory where the file is stored.
    :kwarg delete: toggle deletion of the file.
    :return: mesh loaded from DMPlex format.
    """
    if COMM_WORLD.size > 1:
        raise IOError("Loading a mesh from HDF5 only works in serial.")
    newplex = PETSc.DMPlex().create()
    newplex.createFromFile(os.path.join(fpath, fname + '.h5'))

    # Optionally delete the HDF5 file
    if delete:
        os.remove(os.path.join(fpath, fname) + '.h5')

    return Mesh(newplex)


def initialise_field(fs, name, fname, fpath='.', outputdir=None, op=CoupledOptions(), **kwargs):
    """
    Initialise bathymetry field with results from a previous simulation.

    :arg fs: field will live in this finite element space.
    :arg name: name used internally for field.
    :arg fname: file name (without '.h5' extension).
    :kwarg fpath: directory to read the data from.
    :kwarg op: :class:`Options` parameter object.
    :kwarg index_str: optional five digit string.
    :kwarg delete: toggle deletion of the file.
    """
    delete = kwargs.get('delete', False)
    index_str = kwargs.get('index_str', None)
    fname = get_filename(fname, index_str)
    with timed_stage('initialising {:s}'.format(name)):
        f = Function(fs, name=name)
        with DumbCheckpoint(os.path.join(fpath, fname), mode=FILE_READ) as chk:
            chk.load(f)

    # Optionally delete the HDF5 file
    if delete:
        os.remove(os.path.join(fpath, fname) + '.h5')

    # Plot to PVD
    if outputdir is not None and op.plot_pvd:
        File(os.path.join(outputdir, '_'.join([name, 'imported.pvd']))).write(f)
    return f


def initialise_bathymetry(mesh, fpath, op=CoupledOptions(), **kwargs):
    """
    Initialise bathymetry field with results from a previous simulation.

    :arg mesh: field will be defined in finite element space on this mesh.
    :arg fpath: directory to read the data from.
    :kwarg op: :class:`Options` parameter object.
    """
    # TODO: Would be nice to have consistency: here mesh is an arg but below it is read from file
    fs = FunctionSpace(mesh, op.bathymetry_family.upper(), 1)
    return initialise_field(fs, 'bathymetry', 'bathymetry', fpath, **kwargs)


def initialise_hydrodynamics(inputdir='.', outputdir=None, op=CoupledOptions(), **kwargs):
    """
    Initialise velocity and elevation with results from a previous simulation.

    :kwarg inputdir: directory to read the data from.
    :kwarg outputdir: directory to optionally plot the data in .pvd format.
    :kwarg op: :class:`Options` parameter object.
    :kwarg delete: toggle deletion of the file.
    :kwarg plexname: file name used for the DMPlex data file.
    :kwarg variant: relates to distribution of quadrature nodes in an element.
    """
    plexname = kwargs.get('plexname', 'myplex')
    variant = kwargs.get('variant', 'equispaced')
    delete = kwargs.get('delete', False)
    index_str = kwargs.get('index_str', None)

    # Get finite element
    if op.family == 'dg-dg':
        uv_element = VectorElement("DG", triangle, 1)
        elev_element = FiniteElement("DG", triangle, 1, variant=variant)
    elif op.family == 'dg-cg':
        uv_element = VectorElement("DG", triangle, 1)
        elev_element = FiniteElement("CG", triangle, 2, variant=variant)
    elif op.family == 'cg-cg':
        uv_element = VectorElement("CG", triangle, 2)
        elev_element = FiniteElement("CG", triangle, 1, variant=variant)

    # Load mesh
    with timed_stage('mesh'):
        mesh = op.default_mesh if plexname is None else load_mesh(plexname, inputdir, delete=delete)

    # Load velocity
    name = "velocity"
    fname = get_filename(name, index_str)
    U = FunctionSpace(mesh, uv_element)
    with timed_stage('initialising {:s}'.format(name)):
        with DumbCheckpoint(os.path.join(inputdir, fname), mode=FILE_READ) as chk:
            uv_init = Function(U, name=name)
            chk.load(uv_init)

    # Optionally delete the velocity HDF5 file
    if delete:
        os.remove(os.path.join(inputdir, fname) + '.h5')

    # Load elevation
    name = "elevation"
    fname = get_filename(name, index_str)
    H = FunctionSpace(mesh, elev_element)
    with timed_stage('initialising {:s}'.format(name)):
        with DumbCheckpoint(os.path.join(inputdir, fname), mode=FILE_READ) as chk:
            elev_init = Function(H, name=name)
            chk.load(elev_init)

    # Optionally delete the elevation HDF5 file
    if delete:
        os.remove(os.path.join(inputdir, fname) + '.h5')

    # Plot to .pvd
    if outputdir is not None and op.plot_pvd:
        uv_proj = Function(VectorFunctionSpace(mesh, "CG", 1), name="Initial velocity")
        uv_proj.project(uv_init)
        File(os.path.join(outputdir, "velocity_imported.pvd")).write(uv_proj)
        elev_proj = Function(FunctionSpace(mesh, "CG", 1), name="Initial elevation")
        elev_proj.project(elev_init)
        File(os.path.join(outputdir, "elevation_imported.pvd")).write(elev_proj)

    return uv_init, elev_init


# --- Output

def save_mesh(mesh, fname, fpath='.'):
    """
    :arg mesh: mesh to be saved in DMPlex format.
    :kwarg fname: file name (without '.h5' extension).
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


def export_field(f, name, fname, fpath='.', plexname='myplex', op=CoupledOptions(), index_str=None):
    """
    Export some field to be used in a subsequent simulation.

    :arg f: field (Firedrake :class:`Function`) to be stored.
    :arg name: name used internally for field.
    :arg fname: filename to save the data to.
    :kwarg fpath: directory to save the data to.
    :kwarg plexname: file name to be used for the DMPlex data file.
    :kwarg op: :class:`Options` parameter object.
    :kwarg index_str: optional five digit string.
    """
    if not os.path.exists(fpath):
        os.makedirs(fpath)
    op.print_debug("I/O: Exporting {:s} for subsequent simulation".format(name))

    # Create checkpoint to HDF5
    fname = get_filename(fname, index_str)
    with DumbCheckpoint(os.path.join(fpath, fname), mode=FILE_CREATE) as chk:
        chk.store(f, name=name)

    # Plot to PVD
    if op.plot_pvd:
        File(os.path.join(fpath, '_'.join([name, 'out.pvd']))).write(f)

    # Save mesh to DMPlex format
    if plexname is not None:
        save_mesh(f.function_space().mesh(), plexname, fpath)


def export_bathymetry(bathymetry, fpath, **kwargs):
    """
    Export bathymetry field to be used in a subsequent simulation.

    :arg bathymetry: field to be stored.
    :arg fpath: directory to save the data to.
    :kwarg plexname: file name to be used for the DMPlex data file.
    :kwarg op: :class:`Options` parameter object.
    :kwarg index_str: optional five digit string.
    """
    export_field(bathymetry, 'bathymetry', 'bathymetry', fpath, **kwargs)


def export_hydrodynamics(uv, elev, fpath='.', plexname='myplex', op=CoupledOptions(), **kwargs):
    """
    Export velocity and elevation to be used in a subsequent simulation

    :arg uv: velocity field to be stored.
    :arg elev: elevation field to be stored.
    :kwarg fpath: directory to save the data to.
    :kwarg plexname: file name to be used for the DMPlex data file.
    :kwarg op: :class:`Options` parameter object.
    :kwarg index_str: optional five digit string.
    """
    index_str = kwargs.get('index_str', None)
    if not os.path.exists(fpath):
        os.makedirs(fpath)
    op.print_debug("I/O: Exporting fields for subsequent simulation")

    # Check consistency of meshes
    mesh = elev.function_space().mesh()
    assert mesh == uv.function_space().mesh()

    # Export velocity
    name = "velocity"
    fname = get_filename(name, index_str)
    with DumbCheckpoint(os.path.join(fpath, fname), mode=FILE_CREATE) as chk:
        chk.store(uv, name=name)

    # Export elevation
    name = "elevation"
    fname = get_filename(name, index_str)
    with DumbCheckpoint(os.path.join(fpath, fname), mode=FILE_CREATE) as chk:
        chk.store(elev, name=name)

    # Plot to .pvd
    if op.plot_pvd:
        uv_proj = Function(VectorFunctionSpace(mesh, "CG", 1), name="Initial velocity")
        uv_proj.project(uv)
        File(os.path.join(fpath, 'velocity_out.pvd')).write(uv_proj)
        elev_proj = Function(FunctionSpace(mesh, "CG", 1), name="Initial elevation")
        elev_proj.project(elev)
        File(os.path.join(fpath, 'elevation_out.pvd')).write(elev_proj)

    # Export mesh
    if plexname is not None:
        save_mesh(mesh, plexname, fpath)


# --- Logging

class OuterLoopLogger(object):
    """
    A simple logger for simulations which have an outer loop over meshes. This might be for
    convergence analysis on a hierarchy of fixed meshes, or on sequences of time-dependent adapted
    meshes.
    """
    def __init__(self, prob, verbose=True, **known):
        """
        :arg prob: :class:`AdaptiveProblem` solver object.
        :kwarg verbose: print during logging.
        :kwargs known: expanded dictionary of parameters.
        """
        self.prob = prob
        self.verbose = verbose
        self.divider = 80*'*' + '\n'
        self.msg = "    {:34s}: {:}\n"

        # Create a log string
        self.logstr = self.divider + 33*' ' + 'PARAMETERS\n' + self.divider

        # Log known parameters
        for key in known:
            self.logstr += self.msg.format(key, known[key])

        # Print parameters to screen
        if self.verbose:
            print_output(self.logstr + self.divider)

    def log_git_sha(self):
        """
        Add a line to the log string which records the sha of the adapt_utils git commit in use.
        """
        adapt_utils_home = os.environ.get('ADAPT_UTILS_HOME')
        with open(os.path.join(adapt_utils_home, '.git', 'logs', 'HEAD'), 'r') as gitlog:
            for line in gitlog:
                words = line.split()  # TODO: Just read last line
            self.logstr += self.msg.format('adapt_utils git commit', words[1])

    def create_log_dir(self, fpath):
        """
        :arg fpath: directory to save log file in.
        """
<<<<<<< HEAD
        today = datetime.date.today()
        date = '{:d}-{:d}-{:d}'.format(today.year, today.month, today.day)
=======
        date = get_date()
>>>>>>> dfe1c0b3a34dfef1765835b64b574a69fe60dd9a
        j = 0
        while True:
            self.di = os.path.join(fpath, '{:s}-run-{:d}'.format(date, j))
            if not os.path.exists(self.di):
                create_directory(self.di)
                break
            j += 1

    def write(self, fpath, save_meshes=False):
        """
        Write the log out to file.

        :arg fpath: directory to save log file in.
        :kwarg save_meshes: toggle whether to save meshes to file in DMPlex format.
        """
        if fpath is None:
            return
        self.create_log_dir(fpath)
        with open(os.path.join(self.di, 'log'), 'w') as logfile:
            logfile.write(self.logstr)
        if save_meshes:
            self.prob.store_meshes(fpath=self.di)
        if self.verbose:
            print_output("logdir: {:s}".format(self.di))

    # TODO: Allow logging during simulation
    def log(self, fname='log', fpath=None, save_meshes=False):
        """
        :args unknown: expanded list of unknown parsed arguments.
        :kwarg fname: filename for log file.
        :kwarg fpath: directory to save log file in. If `None`, the log is simply printed.
        :kwarg save_meshes: save meshes to file in the same directory.
        """
        self.log_git_sha()

        # Log element count and QoI from each outer iteration
        self.logstr += self.divider + 35*' ' + 'SUMMARY\n' + self.divider
        self.logstr += "{:8s}    {:7s}\n".format('Elements', 'QoI')
        for num_cells, qoi in zip(self.prob.num_cells, self.prob.qois):
            self.logstr += "{:8d}    {:7.4e}\n".format(num_cells, qoi)
        if self.verbose:
            print_output(self.logstr)

        # Write out
        self.write(fpath, save_meshes=save_meshes)


class TimeDependentAdaptationLogger(OuterLoopLogger):
    """
    A simple logger for simulations which use time-dependent mesh adaptation with an outer loop.
    Statistics on metrics and meshes are printed to screen, saved to a log file, or both.
    """
    # TODO: Allow logging during simulation
    def log(self, *unknown, fname='log', fpath=None, save_meshes=False):
        """
        :args unknown: expanded list of unknown parsed arguments.
        :kwarg fname: filename for log file.
        :kwarg fpath: directory to save log file in. If `None`, the log is simply printed.
        :kwarg save_meshes: save meshes to file in the same directory.
        """
        self.log_git_sha()

        # Log unknown parameters
        for i in range(len(unknown)//2):
            self.logstr += self.msg.format(unknown[2*i][1:], unknown[2*i+1])

        # Log mesh and metric stats from each outer iteration
        self.logstr += self.divider + 35*' ' + 'SUMMARY\n' + self.divider
        for n, (qoi, complexity) in enumerate(zip(self.prob.qois, self.prob.st_complexities)):
            self.logstr += "Mesh iteration {:2d}: qoi {:.4e}".format(n+1, qoi)
            if n > 0:
                self.logstr += " space-time complexity {:.4e}".format(complexity)
            self.logstr += "\n"

        # Log stats from last outer iteration
        self.logstr += self.divider + 30*' ' + 'FINAL ELEMENT COUNTS\n' + self.divider
        l = self.prob.op.end_time/self.prob.op.num_meshes
        for i, num_cells in enumerate(self.prob.num_cells[-1]):
            self.logstr += "Time window ({:7.1f},{:7.1f}]: {:7d}\n".format(i*l, (i+1)*l, num_cells)
        self.logstr += self.divider
        if self.verbose:
            print_output(self.logstr)

        # Write out
        self.write(fpath)


def readfile(filename, reverse=False):
    """
    Read a file line-by-line.

    :kwarg reverse: read the lines in reverse order.
    """
    with open(filename, 'r') as read_obj:
        lines = read_obj.readlines()
    lines = [line.strip() for line in lines]
    if reverse:
        lines = reversed(lines)
    return lines


def index_string(index, n=5):
    """
    :arg index: integer form of index.
    :return: n-digit string form of index.
    """
    return (n - len(str(index)))*'0' + str(index)
<<<<<<< HEAD
=======


def get_date():
    """Get the date in year-month-day format."""
    today = datetime.date.today()
    return '{:d}-{:d}-{:d}'.format(today.year, today.month, today.day)
>>>>>>> dfe1c0b3a34dfef1765835b64b574a69fe60dd9a
