from __future__ import absolute_import

from thetis import *
from thetis.callback import CallbackManager

import numpy as np
import os

from ..adapt.metric import *
from ..io import load_mesh, save_mesh
from ..mesh import aspect_ratio, quality, remesh
from .ts import *  # NOTE: Overrides some of the Thetis time integrators


class AdaptiveProblemBase(object):
    """
    Solver object for adaptive mesh simulations with a number of meshes which is known a priori.
    In the steady state case, the number of meshes is clearly known to be one. In the unsteady
    case, it is likely that we seek to use more than one mesh.

    The philosophy here is to separate the PDE solution from the mesh adaptation, in the sense that
    the forward (and possibly adjoint) equations are solved over the whole time period before any
    mesh adaptation is performed. This means that the solver object is based upon a sequence of
    meshes, as opposed to a single mesh which is updated on-the-fly. Whilst this approach has
    increased memory requirements compared with the on-the-fly strategy, it is beneficial for
    goal-oriented mesh adaptation, where an outer loop is required.

    Whilst this is the case for metric-based mesh adaptation using Pragmatic, mesh movement is
    performed on-the-fly on each mesh in the sequence.
    """
    def __init__(self, op, meshes=None, nonlinear=True, **kwargs):
        """
        :arg op: :class:`Options` parameter object.
        :kwarg meshes: optionally pass a list of meshes to the constructor.
        :kwarg nonlinear: should the PDE(s) be linearised?
        :kwarg checkpointing: should checkpointing be used for continuous adjoint solves?
        :kwarg print_progress: verbose solver output if set to `True`.
        :kwarg manual: if set to `True`, meshes (and objects built upon them) are not set up.
        """
        msg = "{:s} initialisation begin\n".format(self.__class__.__name__)
        op.print_debug(op.indent + 80*'*' + '\n' + op.indent + msg + 80*'*' + '\n')
        self.di = create_directory(op.di)

        # Read args and kwargs
        self.op = op
        self.stabilisation = op.stabilisation
        self.approach = op.approach
        self.nonlinear = nonlinear
        self.checkpointing = kwargs.pop('checkpointing', False)
        self.print_progress = kwargs.pop('print_progress', True)
        self.manual = kwargs.pop('manual', False)

        # Timestepping export details
        self.num_timesteps = int(np.round(op.end_time/op.dt, 0))
        self.num_meshes = op.num_meshes
        try:
            assert self.num_timesteps % op.num_meshes == 0
        except AssertionError:
            msg = "Number of meshes {:d} should divide total number of timesteps {:d}."
            raise ValueError(msg.format(op.num_meshes, self.num_timesteps))
        self.dt_per_mesh = self.num_timesteps//op.num_meshes
        try:
            assert self.dt_per_mesh % op.dt_per_export == 0
        except AssertionError:
            msg = "Timesteps per export {:d} should divide timesteps per mesh iteration {:d}."
            raise ValueError(msg.format(op.dt_per_export, self.dt_per_mesh))
        self.export_per_mesh = self.dt_per_mesh//op.dt_per_export + 1
        physical_constants['g_grav'].assign(op.g)

        # Setup problem
        if not hasattr(self, 'num_cells'):
            self.num_cells = [[], ]
        if not hasattr(self, 'num_vertices'):
            self.num_vertices = [[], ]
        if not hasattr(self, 'max_ar'):
            self.max_ar = [[], ]
        self.meshes = [None for i in range(self.num_meshes)]
        self.have_intermediaries = False
        self.set_meshes(meshes)
        if not self.manual:
            self.setup_all(**kwargs)
        implemented_steppers = {
            'CrankNicolson': CrankNicolson,
            'SteadyState': SteadyState,
        }
        try:
            assert op.timestepper in implemented_steppers
        except AssertionError:
            raise NotImplementedError("Time integrator {:s} not implemented".format(op.timestepper))
        self.integrator = implemented_steppers[self.op.timestepper]
        if op.timestepper == 'SteadyState':
            assert op.end_time <= op.dt

        # Mesh movement
        self.mesh_movers = [None for i in range(self.num_meshes)]

        # Checkpointing
        self.checkpoint = []

        # Storage for diagnostics over mesh adaptation loop
        self.indicators = [{} for i in range(self.num_meshes)]
        self._have_indicated_error = False
        if not hasattr(self, 'estimators'):
            self.estimators = {}
        self.metrics = [None for P1_ten in self.P1_ten]
        if not hasattr(self, 'qois'):
            self.qois = []
        if not hasattr(self, 'st_complexities'):
            self.st_complexities = [np.nan]
        self.outer_iteration = 0

        # Various empty lists and dicts
        self.kernels = [None for i in range(self.num_meshes)]
        if not hasattr(self, 'fwd_solutions'):
            self.fwd_solutions = [None for i in range(self.num_meshes)]
        if not hasattr(self, 'adj_solutions'):
            self.adj_solutions = [None for i in range(self.num_meshes)]
        if not hasattr(self, 'fields'):
            self.fields = [AttrDict() for i in range(self.num_meshes)]

    def print(self, msg):  # TODO: Write to log file
        if self.print_progress:
            print_output(msg)

    def warning(self, msg):
        if os.environ.get('WARNINGS', '0') != '0':
            print_output(msg)

    def set_meshes(self, meshes, index=None):
        """
        Build a mesh associated with each mesh.

        NOTE: If a single mesh is passed to the constructor then it is symlinked into each slot
              rather than explicitly copied. This rears its head in :attr:`run_dwr`, where a the
              enriched meshes are built from a single mesh hierarchy.
        """
        from ..misc import integrate_boundary

        op = self.op
        if meshes is None:
            op.print_debug("SETUP: Setting default meshes...")
            self.meshes = [op.default_mesh for i in range(self.num_meshes)]
        elif index is not None:
            self.meshes[index] = meshes
        elif isinstance(meshes, str):
            self.load_meshes(fname=meshes)  # TODO: allow fpath
        elif not isinstance(meshes, list):
            self.meshes = [meshes for i in range(self.num_meshes)]
        elif meshes != self.meshes:
            op.print_debug("SETUP: Setting user-provided meshes...")
            assert len(meshes) == self.num_meshes
            self.meshes = meshes
        if self.num_cells != [[], ]:
            self.num_cells.append([])
            self.num_vertices.append([])
            self.max_ar.append([])

        if index is not None:
            mesh = self.meshes[index]
            self.dim = mesh.topological_dimension()
            mesh.boundary_len = integrate_boundary(mesh)
            num_cells, num_vertices = mesh.num_cells(), mesh.num_vertices()
            self.op.print_debug(msg.format(i, num_cells, num_vertices))
            return

        msg = "SETUP: Mesh {:d} has {:d} elements and {:d} vertices"
        self.dim = self.meshes[0].topological_dimension()
        for i, mesh in enumerate(self.meshes):
            assert self.dim == mesh.topological_dimension()

            # Endow mesh with its boundary "length"
            mesh.boundary_len = integrate_boundary(mesh)

            # Print diagnostics / store for later use over mesh adaptation loop
            num_cells, num_vertices = mesh.num_cells(), mesh.num_vertices()
            self.op.print_debug(msg.format(i, num_cells, num_vertices))
            self.num_cells[-1].append(num_cells)
            self.num_vertices[-1].append(num_vertices)
            if self.dim == 2:
                self.max_ar[-1].append(aspect_ratio(mesh).vector().gather().max())
            else:
                self.max_ar[-1].append(np.nan)

    def setup_all(self, restarted=False, **kwargs):
        """
        Setup everything which isn't explicitly associated with either the forward or adjoint
        problem.
        """
        self.set_finite_elements()
        self.create_function_spaces()
        self.create_solutions()
        self.set_fields(init=True, reinit=False)
        self.set_boundary_conditions()
        self.create_outfiles(restarted=restarted)
        self.create_intermediary_spaces(self.have_intermediaries)
        self.callbacks = [CallbackManager() for i in range(self.num_meshes)]
        self.equations = [AttrDict() for i in range(self.num_meshes)]
        self.error_estimators = [AttrDict() for i in range(self.num_meshes)]
        self.timesteppers = [AttrDict() for i in range(self.num_meshes)]

    def get_plexes(self):
        """
        :return: DMPlex associated with the ith mesh.
        """
        if hasattr(self, '_plexes') and self._plexes != []:
            return
        self._plexes = []
        for mesh in self.meshes:
            try:
                self._plexes.append(mesh.topology_dm)
            except AttributeError:
                try:
                    self._plexes.append(mesh._topology_dm)
                except AttributeError:
                    self._plexes.append(mesh._plex_)  # Backwards compatability

    @property
    def plexes(self):
        self.get_plexes()
        return self._plexes

    @property
    def plex(self):
        return self.plexes[0]

    def set_finite_elements(self):
        raise NotImplementedError("To be implemented in derived class")

    def create_intermediary_spaces(self, have_intermediaries=False):
        """
        Create a copy of each mesh and define function spaces and solution fields upon it.

        This functionality is used for mesh movement driven by solving Monge-Ampere type equations.
        """
        if self.have_intermediaries or self.op.approach not in ('monge_ampere', 'hybrid'):
            return
        self.op.print_debug("SETUP: Creating intermediary spaces...")
        mesh_copies = [Mesh(mesh.coordinates.copy(deepcopy=True)) for mesh in self.meshes]
        spaces = [FunctionSpace(mesh, self.finite_element) for mesh in mesh_copies]
        self.intermediary_meshes = mesh_copies
        self.intermediary_solutions = [Function(space) for space in spaces]
        self.have_intermediaries = True

    def create_function_spaces(self):
        """
        Build various useful finite element spaces.

        NOTE: The prognostic spaces should be created in the inherited class.
        """
        self.op.print_debug("SETUP: Creating function spaces...")
        self.P0 = [FunctionSpace(mesh, "DG", 0) for mesh in self.meshes]
        self.P0_vec = [VectorFunctionSpace(mesh, "DG", 0) for mesh in self.meshes]
        self.P0_ten = [TensorFunctionSpace(mesh, "DG", 0) for mesh in self.meshes]
        self.P1 = [FunctionSpace(mesh, "CG", 1) for mesh in self.meshes]
        self.P1_vec = [VectorFunctionSpace(mesh, "CG", 1) for mesh in self.meshes]
        self.P1_ten = [TensorFunctionSpace(mesh, "CG", 1) for mesh in self.meshes]
        self.P1DG = [FunctionSpace(mesh, "DG", 1) for mesh in self.meshes]
        self.P1DG_vec = [VectorFunctionSpace(mesh, "DG", 1) for mesh in self.meshes]

        # Store mesh orientations
        self.jacobian_signs = [
            interpolate(sign(JacobianDeterminant(P0.mesh())), P0) for P0 in self.P0
        ]

    def create_solutions(self):
        """
        Set up :class:`Function`s in prognostic spaces defined on each mesh, which will hold
        forward and adjoint solutions.
        """
        self.op.print_debug("SETUP: Creating solutions...")
        for i in range(self.num_meshes):
            self.create_solutions_step(i)

    def create_solutions_step(self, i):
        self.fwd_solutions[i] = Function(self.V[i], name='Forward solution')
        self.adj_solutions[i] = Function(self.V[i], name='Adjoint solution')

    def free_solutions_step(self, i):
        """Free the memory associated with forward and adjoint solution tuples on mesh i."""
        self.fwd_solutions[i] = None
        self.adj_solutions[i] = None

    def set_fields(self, init, reinit, **kwargs):
        """
        Set various fields *on each mesh*, including:

            * viscosity;
            * diffusivity;
            * the Coriolis parameter;
            * drag coefficients;
            * bed roughness.

        The bathymetry is defined via a modified version of the `DepthExpression` found Thetis.
        """
        self.op.print_debug("SETUP: Creating fields...")
        for i in range(self.num_meshes):
            self.set_fields_step(i, init, reinit, **kwargs)

    def set_fields_step(self, i):
        raise NotImplementedError("To be implemented in derived class")

    def free_fields_step(self, i):
        """Free the memory associated with fields on mesh i."""
        self.fields[i] = AttrDict()

    def set_boundary_conditions(self):
        """
        Set boundary conditions *for all models*.
        """
        self.op.print_debug("SETUP: Setting boundary conditions...")
        self.boundary_conditions = [
            self.op.set_boundary_conditions(self, i) for i in range(self.num_meshes)
        ]

    def create_outfiles(self, restarted=False):
        if not self.op.plot_pvd:
            return
        self.op.print_debug("SETUP: Creating output files...")
        if restarted:
            self.solution_file._topology = None
            self.adjoint_solution_file._topology = None
        else:
            self.solution_file = File(os.path.join(self.di, 'solution.pvd'))
            self.adjoint_solution_file = File(os.path.join(self.di, 'adjoint_solution.pvd'))

    def set_initial_condition(self):
        self.op.set_initial_condition(self)

    def set_terminal_condition(self):
        self.op.set_terminal_condition(self)

    def create_forward_equations(self):
        for i in range(self.num_meshes):
            self.create_forward_equations_step(i)

    def create_adjoint_equations(self):
        for i in range(self.num_meshes):
            self.create_adjoint_equations_step(i)

    def create_forward_equations_step(self, i):
        raise NotImplementedError("To be implemented in derived class")

    def create_adjoint_equations_step(self, i):
        raise NotImplementedError("To be implemented in derived class")

    def free_forward_equations_step(self, i):
        """Free the memory associated with forward equations defined on mesh i."""
        raise NotImplementedError("To be implemented in derived class")

    def free_adjoint_equations_step(self, i):
        """Free the memory associated with adjoint equations defined on mesh i."""
        raise NotImplementedError("To be implemented in derived class")

    def create_error_estimators(self):
        for i in range(self.num_meshes):
            self.create_error_estimators_step(i)

    def create_error_estimators_step(self, i):
        raise NotImplementedError("To be implemented in derived class")

    def free_error_estimators_step(self, i):
        """Free the memory associated with error estimators defined on mesh i."""
        raise NotImplementedError("To be implemented in derived class")

    def create_forward_timesteppers(self):
        for i in range(self.num_meshes):
            self.create_forward_timesteppers_step(i)

    def create_adjoint_timesteppers(self):
        for i in range(self.num_meshes):
            self.create_adjoint_timesteppers_step(i)

    def create_forward_timesteppers_step(self, i):
        raise NotImplementedError("To be implemented in derived class")

    def create_adjoint_timesteppers_step(self, i):
        raise NotImplementedError("To be implemented in derived class")

    def free_forward_timesteppers_step(self, i):
        """Free the memory associated with forward timesteppers defined on mesh i."""
        raise NotImplementedError("To be implemented in derived class")

    def free_adjoint_timesteppers_step(self, i):
        """Free the memory associated with adjoint timesteppers defined on mesh i."""
        raise NotImplementedError("To be implemented in derived class")

    def add_callbacks(self, i, **kwargs):
        """To be implemented in derived class"""
        pass

    def project(self, f, i, j):
        """Project field `f` from mesh `i` onto mesh `j`."""
        if f[i] is None or isinstance(f[i], Constant):
            return
        elif f[i].function_space() == f[j].function_space():
            f[j].assign(f[i])
        else:
            for fik, fjk in zip(f[i].split(), f[j].split()):
                fjk.project(fik)

    def project_forward_solution(self, i, j):
        """Project forward solution from mesh `i` to mesh `j`."""
        if self.fwd_solutions[i] is None:
            raise ValueError("Nothing to project.")
        elif self.fwd_solutions[j] is None:
            self.fwd_solutions[j] = Function(self.V[j], name="Forward solution")
        self.project(self.fwd_solutions, i, j)

    def project_adjoint_solution(self, i, j):
        """Project adjoint solution from mesh `i` to mesh `j`."""
        if self.adj_solutions[i] is None:
            raise ValueError("Nothing to project.")
        elif self.adj_solutions[j] is None:
            self.adj_solutions[j] = Function(self.V[j], name="Adjoint solution")
        self.project(self.adj_solutions, i, j)

    def project_fields(self, fields, i):
        """Project a dictionary of fields into corresponding spaces defined on mesh i."""
        for f in fields:
            if isinstance(fields[f], Function):
                self.fields[i].project(fields[f])

    def transfer_solution(self, i, adjoint=False):
        if adjoint:
            self.transfer_adjoint_solution()
        else:
            self.transfer_forward_solution()

    def transfer_forward_solution(self, i, **kwargs):
        if i == 0:
            self.set_initial_condition(**kwargs)
        else:
            self.project_forward_solution(i-1, i)

    def transfer_adjoint_solution(self, i, **kwargs):
        if i == self.num_meshes - 1:
            self.set_terminal_condition(**kwargs)
        else:
            self.project_adjoint_solution(i+1, i)

    def project_to_intermediary_mesh(self, i):
        """
        Project solution fields from mesh i into corresponding function spaces defined on
        intermediary mesh i.

        This function is designed for use under Monge-Ampere type mesh movement methods.
        """
        for f, f_int in zip(self.fwd_solutions[i].split(), self.intermediary_solutions[i].split()):
            f_int.project(f)

    def project_from_intermediary_mesh(self, i):
        """
        Project solution fields from mesh i into corresponding function spaces defined on
        intermediary mesh i.

        This function is designed for use under Monge-Ampere type mesh movement methods.
        """
        for f, f_int in zip(self.fwd_solutions[i].split(), self.intermediary_solutions[i].split()):
            f.project(f_int)

    def copy_data_from_intermediary_mesh(self, i):
        """
        Copy the data from intermediary solution field i into the corresponding solution field
        on the physical mesh.

        This function is designed for use under Monge-Ampere type mesh movement methods.
        """
        for f, f_int in zip(self.fwd_solutions[i].split(), self.intermediary_solutions[i].split()):
            f.assign(f_int)

    def save_meshes(self, fname='plex', fpath=None):
        """
        Save meshes to disk using DMPlex format in HDF5 files.

        :kwarg fname: filename of HDF5 files (with an '_<index>' to be appended).
        :kwarg fpath: directory in which to save the HDF5 files.
        """
        self.op.print_debug("I/O: Saving meshes to file...")
        fpath = fpath or self.di
        self.op.print_debug(self.op.indent + "I/O: Storing plex to {:s}...".format(fname))
        for i, mesh in enumerate(self.meshes):
            save_mesh(mesh, '{:s}_{:d}'.format(fname, i), fpath)

    def load_meshes(self, fname='plex', fpath=None):
        """
        Load meshes in DMPlex format in HDF5 files.

        :kwarg fname: filename of HDF5 files (with an '_<index>' to be appended).
        :kwarg fpath: filepath to where the HDF5 files are to be loaded from.
        """
        self.op.print_debug("I/O: Loading meshes from file...")
        fpath = fpath or self.di
        for i in range(self.num_meshes):
            self.op.print_debug(self.op.indent + "I/O: Loading plex from {:s}...".format(fname))
            self.meshes[i] = load_mesh('{:s}_{:d}'.format(fname, i), fpath)

    def setup_solver_forward_step(self, i, **kwargs):
        raise NotImplementedError("To be implemented in derived class")

    def setup_solver_adjoint_step(self, i, **kwargs):
        raise NotImplementedError("To be implemented in derived class")

    def free_solver_forward_step(self, i):
        self.op.print_debug("FREE: Removing forward timesteppers on mesh {:d}...".format(i))
        self.free_forward_timesteppers_step(i)
        self.op.print_debug("FREE: Removing forward equations on mesh {:d}...".format(i))
        self.free_forward_equations_step(i)

    def free_solver_adjoint_step(self, i):
        self.op.print_debug("FREE: Removing adjoint timesteppers on mesh {:d}...".format(i))
        self.free_adjoint_timesteppers_step(i)
        self.op.print_debug("FREE: Removing adjoint equations on mesh {:d}...".format(i))
        self.free_adjoint_equations_step(i)

    def solve(self, adjoint=False, **kwargs):
        """
        Solve the forward or adjoint problem (as specified by the `adjoint` boolean kwarg) on the
        full sequence of meshes.

        NOTE: The implementation contains a very simple checkpointing scheme, in the sense that
            the terminal solution computed on mesh `i` is stored in `self.fwd_solutions[i]` or
            `self.adj_solutions[i]`, as appropriate.
        """
        if adjoint:
            self.solve_adjoint(**kwargs)
        else:
            self.solve_forward(**kwargs)

    def solve_forward(self, reverse=False, keep=False, **kwargs):
        """
        Solve forward problem on the full sequence of meshes.
        """
        R = range(self.num_meshes-1, -1, -1) if reverse else range(self.num_meshes)
        for i in R:
            self.transfer_forward_solution(i)
            self.setup_solver_forward_step(i)
            self.solve_forward_step(i, **kwargs)
            if not keep:
                self.free_solver_forward_step(i)

    def solve_adjoint(self, reverse=True, keep=False, **kwargs):
        """
        Solve adjoint problem on the full sequence of meshes.
        """
        R = range(self.num_meshes-1, -1, -1) if reverse else range(self.num_meshes)
        for i in R:
            self.transfer_adjoint_solution(i)
            self.setup_solver_adjoint_step(i)
            self.solve_adjoint_step(i, **kwargs)
            if not keep:
                self.free_solver_adjoint_step(i)

    def quantity_of_interest(self):
        """
        Functional of interest which takes the PDE solution as input.
        """
        if hasattr(self.op, 'J'):
            return self.op.J
        else:
            raise NotImplementedError("Should be implemented in derived class.")

    def save_to_checkpoint(self, i, f, mode='memory', **kwargs):
        """
        Extremely simple checkpointing scheme with a simple stack.

        In the case of memory checkpointing, the field to be stored is deep copied and put on top of
        the stack. For disk checkpointing, it is saved to a HDF5 file using a file extension which
        is put on top of the stack, for identification purposes.

        :kwarg mode: toggle whether to save checkpoints in memory or on disk.
        :kwarg fpath: filepath to checkpoint stored on disk.
        """
        assert mode in ('memory', 'disk')
        if mode == 'memory':
            self.checkpoint.append(f.copy(deepcopy=True))
        else:
            fpath = kwargs.get('fpath', self.op.di)
            if i is None:
                raise ValueError("Please provide mesh number")
            chk = len(self.checkpoint)
            self.export_state(i, fpath, index_str=chk)
            self.checkpoint.append(chk)
        self.op.print_debug("CHECKPOINT SAVE: {:3d} currently stored".format(len(self.checkpoint)))

    def collect_from_checkpoint(self, i, mode='memory', delete=True, **kwargs):
        """
        Extremely simple checkpointing scheme which pops off the top of a stack.

        In the case of memory checkpointing, the field is just taken off the top of the stack. For
        disk checkpointing, it is loaded from a HDF5 file using the file extension which is popped
        off the top of the stack.

        :kwarg mode: toggle whether to save checkpoints in memory or on disk.
        :kwarg delete: toggle deletion of the checkpoint file.
        :kwarg fpath: filepath to checkpoint stored on disk.
        """
        assert mode in ('memory', 'disk')
        assert len(self.checkpoint) > 0
        if mode == 'memory':
            return self.checkpoint.pop(-1)
        else:
            fpath = kwargs.get('fpath', self.op.di)
            chk = self.checkpoint.pop(-1)
            self.load_state(i, fpath, index_str=chk, delete=delete)
        self.op.print_debug("CHECKPOINT LOAD: {:3d} currently stored".format(len(self.checkpoint)))

    def get_qoi_kernels(self, i):
        """
        Define kernels associated with the quantity of interest from the corresponding
        `set_qoi_kernel` method of the `Options` parameter class.
        """
        self.op.set_qoi_kernel(self, i)

    # --- Mesh movement

    def set_monitor_functions(self, monitors, bc=None, bbc=None):
        """
        Pass a monitor function to each mesh, thereby defining a `MeshMover` object which drives
        r-adaptation.

        This method is used under 'monge_ampere' and 'laplacian_smoothing' adaptation approaches.

        :arg monitors: a monitor function which takes one argument (the mesh) or a list thereof.
        :kwarg bc: boundary conditions to apply within the mesh movement algorithm.
        :kwarg bbc: boundary conditions to apply within EquationBC objects which appear in the mesh
            movement algorithm.
        """
        from adapt_utils.adapt.r import MeshMover

        # Sanitise input
        assert self.approach in ('monge_ampere', 'laplacian_smoothing')
        assert monitors is not None
        if callable(monitors):
            monitors = [monitors for mesh in self.meshes]

        # Create `MeshMover` objects which drive r-adaptation
        kwargs = {
            'method': self.approach,
            'bc': bc,
            'bbc': bbc,
            'op': self.op,
        }
        self.op.print_debug("MESH MOVEMENT: Creating MeshMover objects...")
        for i in range(self.num_meshes):
            assert monitors[i] is not None
            args = (Mesh(self.meshes[i].coordinates.copy(deepcopy=True)), monitors[i])
            self.mesh_movers[i] = MeshMover(*args, **kwargs)

    def move_mesh(self, i, init=False, reinit=False):
        """
        Move the mesh using an r-adaptive or hybrid method of choice.
        """
        if self.op.approach in ('lagrangian', 'hybrid'):
            return self.move_lagrangian_mesh(i)  # TODO: Make more robust (apply BCs etc.)
        elif self.op.approach == 'ale':
            raise NotImplementedError  # TODO
        elif self.mesh_movers[i] is not None:
            return self.move_mesh_monge_ampere(i, init, reinit)

    def move_lagrangian_mesh(self, i):
        """
        Move mesh i in a Lagrangian sense, using a prescribed mesh velocity.
        """
        op = self.op
        mesh = self.meshes[i]
        t = self.simulation_time
        dt = op.dt
        coords = mesh.coordinates

        assert hasattr(op, 'get_velocity')
        assert op.timestepper == 'CrankNicolson'
        theta = Constant(op.implicitness_theta)  # Crank-Nicolson implicitness
        dx = thetis.dx(degree=3)
        coord_space = coords.function_space()
        coords_old = Function(coord_space).assign(coords)
        test = TestFunction(coord_space)
        trial = coords

        # NOTE: We allow for the velocity to be a nonlinear function of the trial
        F = inner(test, trial)*dx
        F += -theta*dt*inner(test, op.get_velocity(trial, t + op.dt))*dx
        F += -inner(test, coords_old)*dx
        F += -(1 - theta)*dt*inner(test, op.get_velocity(coords_old, t))*dx

        params = {
            'snes_type': 'newtonls',
            'mat_type': 'aij',
            # 'ksp_type': 'gmres',
            'ksp_type': 'preonly',
            # 'pc_type': 'sor',
            'pc_type': 'lu',
            'pc_type_factor_mat_solver_type': 'mumps',
        }
        solve(F == 0, coords, solver_parameters=params)  # TODO: Assemble once

        # Check for inverted elements
        Q = quality(mesh, initial_signs=self.jacobian_signs[i]).dat.data
        restarted = False
        if op.approach == 'hybrid' and Q.min() < op.scaled_jacobian_tol:
            restarted = True
            adapt_field = op.adapt_field
            if adapt_field not in ('tracer', 'sediment', 'bathymetry'):
                adapt_field = 'shallow_water'
            self.project_to_intermediary_mesh(i)
            M = self.get_static_hessian_metric(adapt_field, i=i)
            space_normalise(M, op=op)
            enforce_element_constraints(M, op=op)
            if op.hybrid_mode == 'h':
                self.meshes[i] = adapt(self.meshes[i], M)
            self.meshes[i] = remesh(self.meshes[i])  # TODO: Account for tags
            self.set_meshes(self.meshes)
            self.setup_all(restarted=True)
            self.project_from_intermediary_mesh(i)  # TODO: project fields, too
        num_inverted = len(Q[Q < 0])
        if num_inverted > 0:
            import warnings
            warnings.warn("Mesh has {:d} inverted element(s)!".format(num_inverted))
        return restarted

    def move_mesh_monge_ampere(self, i, init, reinit):  # TODO: Annotation
        """
        Move the physical mesh using a monitor based approach driven by solutions of a Monge-Ampere
        type equation.

        Using the monitor function provided, new mesh coordinates are established. As an intermediate
        step, these are passed to a separate 'intermediary' mesh. Solution fields are projected into
        spaces defined on the intermediary mesh. Finally, the physical mesh coordinates are updated
        and the projected solution data are copied into the solution fields on the physical mesh.

        NOTE: If we want to take the adjoint through mesh movement then there is no need to
              know how the coordinate transform was derived, only what the result was. In any
              case, the current implementation involves a supermesh projection which is not
              yet annotated in pyadjoint.
        """
        # TODO: documentation on how Monge-Ampere is solved.
        if self.mesh_movers[i] is None:
            raise ValueError("No monitor function was provided. Use `set_monitor_functions`.")

        # Compute new physical mesh coordinates
        self.op.print_debug("MESH MOVEMENT: Establishing mesh transformation...")
        self.mesh_movers[i].adapt()

        # Update intermediary mesh coordinates
        self.op.print_debug("MESH MOVEMENT: Updating intermediary mesh coordinates...")
        self.intermediary_meshes[i].coordinates.assign(self.mesh_movers[i].x)

        # Project a copy of the current solution onto mesh defined on new coordinates
        self.op.print_debug("MESH MOVEMENT: Projecting solutions onto intermediary mesh...")
        self.project_to_intermediary_mesh(i)

        # Update physical mesh coordinates
        self.op.print_debug("MESH MOVEMENT: Updating physical mesh coordinates...")
        self.meshes[i].coordinates.assign(self.intermediary_meshes[i].coordinates)

        # Copy over projected solution data
        self.op.print_debug("MESH MOVEMENT: Transferring solution data from intermediary mesh...")
        self.copy_data_from_intermediary_mesh(i)

        # Re-interpolate fields
        self.op.print_debug("MESH MOVEMENT: Re-interpolating fields...")
        self.set_fields(init, reinit)
        return False

    # --- Error estimation

    def get_strong_residual(self, i, adjoint=False, **kwargs):
        ts = self.get_timestepper(i, self.op.adapt_field, adjoint=adjoint)
        strong_residual = ts.error_estimator.strong_residual
        return [project(res, self.P1[i]) for res in list(strong_residual)]  # Project into P1 space

    # --- Metric based

    def maximum_metric(self, i):
        """
        Get an isotropic metric with the maximum prescribed sizes.
        """
        I = Identity(self.dim)
        return interpolate(Constant(pow(self.op.h_max, -2))*I, self.P1_ten[i])

    def get_recovery(self, i, **kwargs):
        raise NotImplementedError("To be implemented in derived class")

    def plot_metrics(self, normalised=True, hessians=False):
        """
        Plot all :attr:`metrics`.
        """
        if not self.op.plot_pvd:
            return
        if hessians:
            assert hasattr(self, '_H_windows')
            fnames = ['metric_{:d}'.format(i) for i in range(len(self._H_windows))]
        else:
            fnames = ['metric']
        for fname in fnames:
            if not normalised:
                if not self.op.debug:
                    return
                fname += '_before_normalisation'
            fname += '.pvd'
            metric_file = File(os.path.join(self.di, fname))
            for i, M in enumerate(self.metrics):
                metric_file._topology = None
                metric_file.write(M)

    def space_time_normalise(self):
        """
        Space-time normalise :attr:`metrics` and perform logging and plotting.
        """
        self.plot_metrics(normalised=False)
        space_time_normalise(self.metrics, op=self.op)
        self.plot_metrics(normalised=True)
        self.log_complexities()

    def log_complexities(self):
        """
        Log static and space-time complexities of :attr:`metrics`.
        """
        complexities = [metric_complexity(M) for M in self.metrics]
        self.st_complexities.append(np.sum(complexities)*self.num_timesteps)
        self.print("\nRiemannian metrics\n==================")
        for i, c in enumerate(complexities):
            self.print("  metric {:2d}: static complexity {:13.4e}".format(i, c))
        self.print("         space-time complexity {:13.4e}".format(self.st_complexities[-1]))

    def adapt_meshes(self, save=False):
        """
        Adapt all :attr:`meshes` based on :attr:`metrics`.
        """
        self.print("\nStarting mesh adaptation for iteration {:d}...".format(self.outer_iteration+1))
        for i, M in enumerate(self.metrics):
            if self.num_meshes > 1:
                self.print("Adapting mesh {:d}/{:d}...".format(i+1, self.num_meshes))
            self.meshes[i] = adapt(self.meshes[i], M)
        self.metrics = [None for P1_ten in self.P1_ten]
        self.set_meshes(self.meshes)
        self.setup_all()

        # Logging
        adapt_field = self.op.adapt_field
        if self.op.adapt_field not in ('tracer', 'sediment', 'bathymetry'):
            adapt_field = 'shallow_water'
        self.log_entities(adapt_field=adapt_field)
        if save:
            self.save_meshes()

    def log_entities(self, adapt_field=None):
        """
        Log DOF, element and vertex counts of :attr:`meshes`.
        """
        base_space = self.get_function_space(adapt_field or self.op.adapt_field)
        self.dofs.append([np.sum(fs.dof_count) for fs in base_space])
        self.print("\nResulting meshes\n================")
        msg = "  mesh {:2d}: vertices {:7d} elements {:7d}   max. aspect ratio {:.1f}"
        if self.num_meshes == 1:
            msg = msg[13:]
        for i, (nv, nc, ar) in enumerate(zip(self.num_vertices[-1], self.num_cells[-1], self.max_ar[-1])):
            if self.num_meshes == 1:
                self.print(msg.format(nv, nc, ar))
            else:
                self.print(msg.format(i, nv, nc, ar))
        self.print("\n")

    def combine_over_windows(self, adapt_fields):
        """
        Given a number of Hessians on each mesh, combine according to `adapt_fields`.
        """
        if not hasattr(self, '_H_windows'):
            raise ValueError("Cannot combine Hessian metrics if they do not exist!")
        self.metrics = [Function(P1_ten, name="Hessian metric") for P1_ten in self.P1_ten]
        for i in range(self.num_meshes):
            H_window = [self._H_windows[j][i] for j in range(len(adapt_fields))]
            if 'int' in self.op.adapt_field:
                if 'avg' in self.op.adapt_field:
                    msg = "Simultaneous intersection and averaging not supported"
                    raise NotImplementedError(msg)
                self.metrics[i].assign(metric_intersection(*H_window))
            elif 'avg' in self.op.adapt_field:
                self.metrics[i].assign(metric_average(*H_window))
            elif len(adapt_fields) == 1:
                self.metrics[i].assign(H_window[0])
            else:
                raise ValueError("adapt_field '{:s}' not recognised".format(self.op.adapt_field))

    def run_hessian_based(self, **kwargs):
        """
        Adaptation loop for Hessian based approach.

        Field for adaptation is specified by `op.adapt_field`.

        Multiple fields can be combined using double-understrokes and either 'avg' for metric
        average or 'int' for metric intersection. We assume distributivity of intersection over
        averaging.

        For example, `adapt_field = 'elevation__avg__velocity_x__int__bathymetry'` would imply
        first intersecting the Hessians recovered from the x-component of velocity and bathymetry
        and then averaging the result with the Hessian recovered from the elevation.

        Convergence criteria:
          * Convergence of quantity of interest (relative tolerance `op.qoi_rtol`);
          * Convergence of mesh element count (relative tolerance `op.element_rtol`);
          * Maximum number of iterations reached (`op.max_adapt`).

        :kwarg save_mesh: save all adapted meshes to HDF5 after every adaptation step. They will
            exist in :attr:`di`, with the name 'plex_', followed by the integer specifying the place
            in the sequence.
        """
        op = self.op
        wq = Constant(1.0)  # Quadrature weight
        dt_per_mesh = self.dt_per_mesh
        assert self.approach == 'hessian'

        # Process parameters
        if op.adapt_field in ('all_avg', 'all_int'):
            c = op.adapt_field[-3:]
            op.adapt_field = "velocity_x__{:s}__velocity_y__{:s}__elevation".format(c, c)
        adapt_fields = ('__int__'.join(op.adapt_field.split('__avg__'))).split('__int__')

        # Loop until we hit the maximum number of iterations, max_adapt
        assert op.min_adapt < op.max_adapt
        # hessian_kwargs = dict(normalise=False, enforce_constraints=False)
        hessian_kwargs = dict(normalise=True, enforce_constraints=False, noscale=True)
        for n in range(op.max_adapt):
            self.outer_iteration = n
            fwd_solutions = self.get_solutions(op.adapt_field, adjoint=False)

            # Arrays to hold Hessians for each field on each window
            self._H_windows = [[Function(P1_ten) for P1_ten in self.P1_ten] for f in adapt_fields]

            # Solve forward, accumulating Hessians
            for i in range(self.num_meshes):
                update_forcings = op.get_update_forcings(self, i, adjoint=False)
                export_func = op.get_export_func(self, i)
                self.simulation_time = i*op.dt*self.dt_per_mesh
                self.transfer_forward_solution(i)
                self.setup_solver_forward_step(i)

                # Create double L2 projection operator which will be repeatedly used
                recoverer = self.get_recovery(i, **hessian_kwargs)

                def hessian(sol, adapt_field):
                    fields = {'adapt_field': adapt_field, 'fields': self.fields[i]}
                    return recoverer.construct_metric(sol, **fields, **hessian_kwargs)

                # Array to hold time-integrated Hessian UFL expression
                H_window = [0 for f in adapt_fields]

                def update_forcings_wrapper(t):
                    """
                    Time-combine Hessians according to :attr:`op.hessian_time_combination` and
                    the time integrator.
                    """
                    update_forcings(t)
                    if self.iteration % op.hessian_timestep_lag != 0:
                        return
                    first_ts = self.iteration == i*dt_per_mesh
                    final_ts = self.iteration == (i+1)*dt_per_mesh

                    # Get quadrature weights
                    if op.hessian_time_combination == 'integrate':
                        if op.timestepper == 'CrankNicolson':
                            w = 0.5 if first_ts or final_ts else 1.0
                        else:
                            raise NotImplementedError  # TODO: Other timesteppers
                        wq.assign(w*op.dt*op.hessian_timestep_lag)

                    # Combine as appropriate
                    for f, field in enumerate(adapt_fields):
                        H = hessian(fwd_solutions[i], field)
                        if field == 'bathymetry':  # TODO: account for non-fixed bathymetry
                            H_window[f] = H
                        elif op.hessian_time_combination == 'integrate':
                            H_window[f] += wq*H
                        else:
                            H_window[f] = H if H_window[f] == 0 else metric_intersection(H, H_window[f])

                def export_func_wrapper():
                    """
                    Extract time-combined Hessians.

                    NOTE: We only care about the final export in each mesh iteration.
                    """
                    t = self.simulation_time
                    export_func()
                    if np.isclose(t, (i+1)*op.dt*dt_per_mesh):
                        update_forcings_wrapper(t)
                        for j, H in enumerate(H_window):
                            self._H_windows[j][i].interpolate(H_window[j])

                # Solve step for current mesh iteration
                solve_kwargs = {
                    'export_func': export_func_wrapper,
                    'update_forcings': update_forcings_wrapper,
                    'plot_pvd': op.plot_pvd,
                    'export_initial': False,
                    'final_update': False,
                }
                self.solve_forward_step(i, **solve_kwargs)

                # Delete objects to free memory
                self.free_solver_forward_step(i)

            # Check convergence
            if (self.qoi_converged or self.maximum_adaptations_met) and self.minimum_adaptations_met:
                break

            # Normalise metrics
            for H_window in self._H_windows:
                space_time_normalise(H_window, op=op)

            # Combine metrics
            self.combine_over_windows(adapt_fields)
            self.plot_metrics(normalised=True)
            self.log_complexities()

            # Adapt meshes
            self.adapt_meshes(save=kwargs.get('save_mesh', False))

            # Check convergence
            if not self.minimum_adaptations_met:
                continue
            if self.elements_converged:
                break

    # --- Convergence checking

    def _check_qoi_convergence(self):
        n = self.outer_iteration
        try:
            qoi = self.quantity_of_interest()
        except NotImplementedError:
            return False
        self.print("\nQuantity of interest\n====================")
        self.print("  iteration {:d}: {:.4e}\n".format(n+1, qoi))
        self.qois.append(qoi)
        converged = False
        if len(self.qois) == 1:
            return False
        if np.abs(self.qois[-1] - self.qois[-2]) < self.op.qoi_rtol*self.qois[-2]:
            n = self.outer_iteration
            self.print("Converged quantity of interest after {:d} iterations!".format(n+1))
            converged = True
        return converged

    @property
    def qoi_converged(self):
        return self._check_qoi_convergence()

    def _check_estimator_convergence(self):
        n = self.outer_iteration
        approach = self.op.approach
        if 'dwr_adjoint' in approach:
            estimators = self.estimators['dwr_adjoint']
        elif approach == 'dwr_both' or 'dwr_avg' in approach or 'dwr_int' in approach:
            estimators = self.estimators['dwr_both']
        elif 'dwr' in approach:
            estimators = self.estimators['dwr']
        else:
            return False
        self.print("\nError estimator\n===============")
        self.print("  iteration {:d}: {:.4e}".format(n+1, estimators[-1]))
        converged = False
        if len(estimators) == 1:
            return converged
        if np.abs(estimators[-1] - estimators[-2]) <= self.op.estimator_rtol*estimators[-2]:
            n = self.outer_iteration
            self.print("Converged error estimator after {:d} iterations!".format(n+1))
            converged = True
        return converged

    @property
    def estimator_converged(self):
        return self._check_estimator_convergence()

    def _check_maximum_adaptations(self):
        n = self.outer_iteration
        chk = n >= self.op.max_adapt-1
        msg = "Maximum number of adaptations{:s} met ({:d}/{:d})."
        print_func = self.print if chk else self.op.print_debug
        print_func(msg.format('' if chk else ' not', n+1, self.op.max_adapt))
        return chk

    @property
    def maximum_adaptations_met(self):
        return self._check_maximum_adaptations()

    def _check_minimum_adaptations(self):
        n = self.outer_iteration
        chk = n >= self.op.min_adapt
        msg = "Minimum number of adaptations{:s} met ({:d}/{:d})."
        self.op.print_debug(msg.format('' if chk else ' not', n+1, self.op.max_adapt))
        return chk

    @property
    def minimum_adaptations_met(self):
        return self._check_minimum_adaptations()

    def _check_element_convergence(self):
        converged = True
        for i, (num_cells, num_cells_) in enumerate(zip(self.num_cells[-1], self.num_cells[-2])):
            if np.abs(num_cells - num_cells_) > self.op.element_rtol*num_cells_:
                converged = False
        if converged:
            n = self.outer_iteration
            self.print("Converged number of mesh elements after {:d} iterations!".format(n+1))
            self.solve_forward()  # Ensure that final outputs are from converged mesh
        return converged

    @property
    def elements_converged(self):
        return self._check_element_convergence()

    def clear_tape(self):
        pass
