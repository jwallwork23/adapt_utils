from thetis import *
from firedrake.petsc import PETSc

import datetime
import os


__all__ = ["save_mesh", "load_mesh", "initialise_fields", "export_final_state",
           "OuterLoopLogger", "TimeDependentAdaptationLogger"]


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
    save_mesh(f.function_space().mesh(), plexname, fpath)


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

        # Check we have a time-dependent adaptive run
        assert prob.op.approach != 'fixed_mesh'

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
        today = datetime.date.today()
        date = '{:d}-{:d}-{:d}'.format(today.year, today.month, today.day)
        j = 0
        while True:
            self.di = os.path.join(fpath, '{:s}-run-{:d}'.format(date, j))
            if not os.path.exists(self.di):
                create_directory(self.di)
                break
            j += 1

    def write(self, fpath):
        """
        Write the log out to file.

        :arg fpath: directory to save log file in.
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
        self.write(fpath)


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
