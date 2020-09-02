from __future__ import absolute_import

from thetis import *

import numpy as np
import os

from ..adapt.adaptation import pragmatic_adapt
from ..adapt.metric import *
from ..io import save_mesh, load_mesh
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
        op.print_debug(op.indent + "{:s} initialisation begin".format(self.__class__.__name__))

        # Read args and kwargs
        self.op = op
        self.stabilisation = op.stabilisation
        self.approach = op.approach
        self.nonlinear = nonlinear
        self.checkpointing = kwargs.get('checkpointing', False)
        self.print_progress = kwargs.get('print_progress', True)
        self.manual = kwargs.get('manual', False)

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
        physical_constants['g_grav'].assign(op.g)

        # Setup problem
        op.print_debug(op.indent + "SETUP: Building meshes...")
        self.num_cells = [[], ]
        self.num_vertices = [[], ]
        self.set_meshes(meshes)
        if not self.manual:
            self.setup_all()
        implemented_steppers = {
            'CrankNicolson': CrankNicolson,
            'SteadyState': SteadyState,
            'PressureProjectionPicard': PressureProjectionPicard,
        }
        try:
            assert op.timestepper in implemented_steppers
        except AssertionError:
            raise NotImplementedError("Time integrator {:s} not implemented".format(op.timestepper))
        self.integrator = implemented_steppers[self.op.timestepper]
        if op.timestepper == 'SteadyState':
            assert op.end_time < op.dt

        # Mesh movement
        self.mesh_movers = [None for i in range(self.num_meshes)]

        # Checkpointing
        self.checkpoint = []

        # Storage for diagnostics over mesh adaptation loop
        self.indicators = [{} for i in range(self.num_meshes)]
        self.estimators = [{} for i in range(self.num_meshes)]
        self.qois = []
        self.st_complexities = [np.nan]
        self.outer_iteration = 0

        # Various empty lists and dicts
        self.callbacks = [None for i in range(self.num_meshes)]
        self.equations = [AttrDict() for i in range(self.num_meshes)]
        self.error_estimators = [AttrDict() for i in range(self.num_meshes)]
        self.kernels = [None for i in range(self.num_meshes)]
        self.timesteppers = [AttrDict() for i in range(self.num_meshes)]
        if not hasattr(self, 'fwd_solutions'):
            self.fwd_solutions = [None for i in range(self.num_meshes)]
        if not hasattr(self, 'adj_solutions'):
            self.adj_solutions = [None for i in range(self.num_meshes)]
        if not hasattr(self, 'fields'):
            self.fields = [AttrDict() for i in range(self.num_meshes)]

    def print(self, msg):
        if self.print_progress:
            print_output(msg)

    def setup_all(self):
        """
        Setup everything which isn't explicitly associated with either the forward or adjoint
        problem.
        """
        op = self.op
        op.print_debug(op.indent + "SETUP: Creating function spaces...")
        self.set_finite_elements()
        self.create_function_spaces()
        self.jacobian_signs = [interpolate(sign(JacobianDeterminant(mesh)), P0) for mesh, P0 in zip(self.meshes, self.P0)]
        op.print_debug(op.indent + "SETUP: Creating solutions...")
        self.create_solutions()
        op.print_debug(op.indent + "SETUP: Creating fields...")
        self.set_fields(init=True)
        op.print_debug(op.indent + "SETUP: Setting stabilisation parameters...")
        self.set_stabilisation()
        op.print_debug(op.indent + "SETUP: Setting boundary conditions...")
        self.set_boundary_conditions()
        op.print_debug(op.indent + "SETUP: Creating output files...")
        self.di = create_directory(op.di)
        self.create_outfiles()
        op.print_debug(op.indent + "SETUP: Creating intermediary spaces...")
        self.create_intermediary_spaces()

    def set_meshes(self, meshes):
        """
        Build a mesh associated with each mesh.

        NOTE: If a single mesh is passed to the constructor then it is symlinked into each slot
              rather than explicitly copied. This rears its head in :attr:`run_dwr`, where a the
              enriched meshes are built from a single mesh hierarchy.
        """
        self.meshes = meshes or [self.op.default_mesh for i in range(self.num_meshes)]
        self.mesh_velocities = [None for i in range(self.num_meshes)]
        msg = self.op.indent + "SETUP: Mesh {:d} has {:d} elements"
        if self.num_cells != [[], ]:
            self.num_cells.append([])
            self.num_vertices.append([])
        for i, mesh in enumerate(self.meshes):
            bnd_len = compute_boundary_length(mesh)
            mesh.boundary_len = bnd_len
            self.op.print_debug(msg.format(i, mesh.num_cells()))
            # if self.op.approach in ('lagrangian', 'ale', 'monge_ampere'):  # TODO
            #     coords = mesh.coordinates
            #     self.mesh_velocities[i] = Function(coords.function_space(), name="Mesh velocity")

            # Store diagnostics for later use over mesh adaptation loop
            self.num_cells[-1].append(mesh.num_cells())
            self.num_vertices[-1].append(mesh.num_vertices())

    def get_plex(self, i):
        """
        :return: DMPlex associated with the ith mesh.
        """
        try:
            return self.meshes[i]._topology_dm
        except AttributeError:
            return self.meshes[i]._plex  # Backwards compatability

    def set_finite_elements(self):
        raise NotImplementedError("To be implemented in derived class")

    def create_intermediary_spaces(self):
        """
        Create a copy of each mesh and define function spaces and solution fields upon it.

        This functionality is used for mesh movement driven by solving Monge-Ampere type equations.
        """
        if self.op.approach != 'monge_ampere':
            return
        mesh_copies = [Mesh(mesh.coordinates.copy(deepcopy=True)) for mesh in self.meshes]
        spaces = [FunctionSpace(mesh, self.finite_element) for mesh in mesh_copies]
        self.intermediary_meshes = mesh_copies
        self.intermediary_solutions = [Function(space) for space in spaces]

    def create_function_spaces(self):
        raise NotImplementedError("To be implemented in derived class")

    def create_solutions(self):
        """
        Set up :class:`Function`s in prognostic spaces defined on each mesh, which will hold
        forward and adjoint solutions.
        """
        for i in range(self.num_meshes):
            self.create_solutions_step(i)

    def create_solutions_step(self, i):
        self.fwd_solutions[i] = Function(self.V[i], name='Forward solution')
        self.adj_solutions[i] = Function(self.V[i], name='Adjoint solution')

    def free_solutions_step(self, i):
        """Free the memory associated with forward and adjoint solution tuples on mesh i."""
        self.fwd_solutions[i] = None
        self.adj_solutions[i] = None

    def set_fields(self, **kwargs):
        """
        Set various fields *on each mesh*, including:

            * viscosity;
            * diffusivity;
            * the Coriolis parameter;
            * drag coefficients;
            * bed roughness.

        The bathymetry is defined via a modified version of the `DepthExpression` found Thetis.
        """
        for i in range(self.num_meshes):
            self.set_fields_step(i, **kwargs)

    def set_fields_step(self, i):
        raise NotImplementedError("To be implemented in derived class")

    def free_fields_step(self, i):
        """Free the memory associated with fields on mesh i."""
        self.fields[i] = AttrDict()

    def set_stabilisation(self):
        for i in range(self.num_meshes):
            self.set_stabilisation_step(i)

    def set_stabilisation_step(self, i):
        raise NotImplementedError("To be implemented in derived class")

    def set_boundary_conditions(self):
        """Set boundary conditions *for all models*"""
        self.boundary_conditions = [self.op.set_boundary_conditions(self, i) for i in range(self.num_meshes)]

    def create_outfiles(self):
        raise NotImplementedError("To be implemented in derived class")

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

    def add_callbacks(self, i):
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
        self.project(self.fwd_solutions, i, j)

    def project_adjoint_solution(self, i, j):
        """Project adjoint solution from mesh `i` to mesh `j`."""
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

    def transfer_forward_solution(self, i):
        if i == 0:
            self.set_initial_condition()
        else:
            self.project_forward_solution(i-1, i)

    def transfer_adjoint_solution(self, i):
        if i == self.num_meshes - 1:
            self.set_terminal_condition()
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

    def copy_data_from_intermediary_mesh(self, i):
        """
        Copy the data from intermediary solution field i into the corresponding solution field
        on the physical mesh.

        This function is designed for use under Monge-Ampere type mesh movement methods.
        """
        for f, f_int in zip(self.fwd_solutions[i].split(), self.intermediary_solutions[i].split()):
            f.dat.data[:] = f_int.dat.data

    def save_meshes(self, fname='plex', fpath=None):
        """
        Save meshes to disk using DMPlex format in HDF5 files.

        :kwarg fname: filename of HDF5 files (with an '_<index>' to be appended).
        :kwarg fpath: directory in which to save the HDF5 files.
        """
        fpath = fpath or os.path.join(self.di, self.approach)
        for i, mesh in enumerate(self.meshes):
            save_mesh(mesh, '_'.join([fname, '{:d}.h5'.format(i)]), fpath)

    def load_meshes(self, fname='plex', fpath=None):
        """
        Load meshes in DMPlex format in HDF5 files.

        :kwarg fname: filename of HDF5 files (with an '_<index>' to be appended).
        :kwarg fpath: filepath to where the HDF5 files are to be loaded from.
        """
        fpath = fpath or os.path.join(self.di, self.approach)
        for i in range(self.num_meshes):
            self.meshes[i] = load_mesh('_'.join([fname, '{:d}.h5'.format(i)]), fpath)

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

    def solve_forward(self, reverse=False, **kwargs):
        """Solve forward problem on the full sequence of meshes."""
        R = range(self.num_meshes-1, -1, -1) if reverse else range(self.num_meshes)
        for i in R:
            self.transfer_forward_solution(i)
            self.setup_solver_forward(i)
            self.solve_forward_step(i, **kwargs)

    def solve_adjoint(self, reverse=True, **kwargs):
        """Solve adjoint problem on the full sequence of meshes."""
        R = range(self.num_meshes-1, -1, -1) if reverse else range(self.num_meshes)
        for i in R:
            self.transfer_adjoint_solution(i)
            self.setup_solver_adjoint(i)
            self.solve_adjoint_step(i, **kwargs)

    def quantity_of_interest(self):
        """Functional of interest which takes the PDE solution as input."""
        raise NotImplementedError("Should be implemented in derived class.")

    def save_to_checkpoint(self, f, mode='memory'):
        """Extremely simple checkpointing scheme with a simple stack of copied fields."""
        assert mode in ('memory', 'disk')
        if mode == 'memory':
            self.checkpoint.append(f.copy(deepcopy=True))
        else:
            raise NotImplementedError("Checkpointing to disk not yet implemented.")
            # TODO: add a string to the stack which provides the (auto generated) file name
        self.op.print_debug("CHECKPOINT SAVE: {:3d} currently stored".format(len(self.checkpoint)))

    def collect_from_checkpoint(self, mode='memory', **kwargs):
        """
        Extremely simple checkpointing scheme which pops off the top of a stack of copied fields.
        """
        assert mode in ('memory', 'disk')
        if mode == 'disk':
            # delete = kwargs.get('delete', True)
            raise NotImplementedError("Checkpointing to disk not yet implemented.")
            # TODO: pop file name from stack, load file, delete if requested
        self.op.print_debug("CHECKPOINT LOAD: {:3d} currently stored".format(len(self.checkpoint)))
        return self.checkpoint.pop(-1)

    def run(self, **kwargs):
        """
        Run simulation using mesh adaptation approach specified by `self.approach`.

        For metric-based approaches, a fixed point iteration loop is used.
        """
        run_scripts = {

            # Non-adaptive
            'fixed_mesh': self.solve_forward,

            # Metric-based, no adjoint
            'hessian': self.run_hessian_based,

            # Metric-based with adjoint
            'dwp': self.run_dwp,
            'dwr': self.run_dwr,
        }
        try:
            run_scripts[self.approach](**kwargs)
        except KeyError:
            raise ValueError("Approach '{:s}' not recognised".format(self.approach))

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

    def move_mesh(self, i):
        # TODO: documentation
        if self.op.approach in ('lagrangian', 'ale'):  # TODO: Make more robust (apply BCs etc.)
            self.move_mesh_ale(i)
        elif self.mesh_movers[i] is not None:
            self.move_mesh_monge_ampere(i)

    def move_mesh_ale(self, i):
        # TODO: documentation
        mesh = self.meshes[i]
        t = self.simulation_time
        dt = self.op.dt
        coords = mesh.coordinates

        # Crank-Nicolson  # TODO: Assemble once
        if hasattr(self.op, 'get_velocity'):
            coord_space = coords.function_space()
            coords_old = Function(coord_space).assign(coords)
            test = TestFunction(coord_space)

            F = inner(test, coords)*dx - inner(test, coords_old)*dx
            F -= 0.5*dt*inner(test, self.op.get_velocity(coords_old, t))*dx
            F -= 0.5*dt*inner(test, self.op.get_velocity(coords, t))*dx

            params = {
                'mat_type': 'aij',
                'snes_type': 'newtonls',
                'snes_rtol': 1.0e-03,
                # 'ksp_type': 'gmres',
                'ksp_type': 'preonly',
                # 'pc_type': 'sor',
                'pc_type': 'lu',
                'pc_type_factor_mat_solver_type': 'mumps',
            }
            solve(F == 0, coords, solver_parameters=params)

        # Forward Euler
        else:
            coords.interpolate(coords + dt*self.mesh_velocities[i])

        # Check for inverted elements
        orig_signs = self.jacobian_signs[i]
        r = interpolate(JacobianDeterminant(mesh)/orig_signs, self.P0[i]).vector().gather()
        num_inverted = len(r[r < 0])
        if num_inverted > 0:
            import warnings
            warnings.warn("WARNING: Mesh has {:d} inverted element(s)!".format(num_inverted))

    def move_mesh_monge_ampere(self, i):  # TODO: Annotation
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
        """  # TODO: documentation on how Monge-Ampere is solved.
        if self.mesh_movers[i] is None:
            raise ValueError("No monitor function was provided. Use `set_monitor_functions`.")

        # Compute new physical mesh coordinates
        self.op.print_debug("MESH MOVEMENT: Establishing mesh transformation...")
        self.mesh_movers[i].adapt()

        # Update intermediary mesh coordinates
        self.op.print_debug("MESH MOVEMENT: Updating intermediary mesh coordinates...")
        self.intermediary_meshes[i].coordinates.dat.data[:] = self.mesh_movers[i].x.dat.data

        # Project a copy of the current solution onto mesh defined on new coordinates
        self.op.print_debug("MESH MOVEMENT: Projecting solutions onto intermediary mesh...")
        self.project_to_intermediary_mesh(i)

        # Update physical mesh coordinates
        self.op.print_debug("MESH MOVEMENT: Updating physical mesh coordinates...")
        self.meshes[i].coordinates.dat.data[:] = self.intermediary_meshes[i].coordinates.dat.data

        # Copy over projected solution data
        self.op.print_debug("MESH MOVEMENT: Transferring solution data from intermediary mesh...")
        self.copy_data_from_intermediary_mesh(i)  # FIXME: Needs annotation

        # Re-interpolate fields
        self.op.print_debug("MESH MOVEMENT: Re-interpolating fields...")
        self.set_fields()

    # --- Metric based

    def get_hessian_recoverer(self, i, **kwargs):
        raise NotImplementedError("To be implemented in derived class")

    # TODO: Create and free objects as needed
    # TODO: kwargs currently unused
    def run_hessian_based(self, update_forcings=None, export_func=None, **kwargs):
        """
        Adaptation loop for Hessian based approach.

        Field for adaptation is specified by `op.adapt_field`.

        Multiple fields can be combined using double-understrokes and either 'avg' for metric
        average or 'int' for metric intersection. We assume distributivity of intersection over
        averaging.

        For example, `adapt_field = 'elevation__avg__velocity_x__int__bathymetry'` would imply
        first intersecting the Hessians recovered from the x-component of velocity and bathymetry
        and then averaging the result with the Hessian recovered from the elevation.

        Stopping criteria:
          * iteration count > self.op.num_adapt;
          * relative change in element count < self.op.element_rtol;
          * relative change in quantity of interest < self.op.qoi_rtol.
        """
        op = self.op
        dt_per_mesh = self.dt_per_mesh

        # Process parameters
        if op.adapt_field in ('all_avg', 'all_int'):
            c = op.adapt_field[-3:]
            op.adapt_field = "velocity_x__{:s}__velocity_y__{:s}__elevation".format(c, c)
        adapt_fields = ('__int__'.join(op.adapt_field.split('__avg__'))).split('__int__')
        if op.hessian_time_combination not in ('integrate', 'intersect'):
            msg = "Hessian time combination method '{:s}' not recognised."
            raise ValueError(msg.format(op.hessian_time_combination))

        # Loop until we hit the maximum number of iterations, num_adapt
        self.outer_iteration = 0
        while self.outer_iteration < op.num_adapt:
            export_func_wrapper = None
            update_forcings_wrapper = None
            if hasattr(self, 'hessian_func'):
                delattr(self, 'hessian_func')

            # Arrays to hold Hessians for each field on each window
            H_windows = [[Function(P1_ten) for P1_ten in self.P1_ten] for f in adapt_fields]

            # Loop over meshes
            for i in range(self.num_meshes):
                update_forcings = update_forcings or op.get_update_forcings(self, i, adjoint=False)
                export_func = export_func or op.get_export_func(self, i)

                # Transfer the solution from the previous mesh / apply initial condition
                self.transfer_forward_solution(i)

                # Setup solver on mesh i
                self.setup_solver_forward(i)

                if self.outer_iteration < op.num_adapt-1:

                    # Create double L2 projection operator which will be repeatedly used
                    kwargs = {
                        'enforce_constraints': False,
                        'normalise': False,
                        'noscale': True,
                    }
                    recoverer = self.get_hessian_recoverer(i, **kwargs)

                    def hessian(sol, adapt_field):
                        fields = {'adapt_field': adapt_field, 'fields': self.fields[i]}
                        return recoverer.get_hessian_metric(sol, **fields, **kwargs)

                    # Array to hold time-integrated Hessian UFL expression
                    H_window = [0 for f in adapt_fields]

                    # TODO: Other timesteppers
                    def update_forcings_wrapper(t):
                        """Time-integrate Hessian using Trapezium Rule."""
                        update_forcings(t)
                        iteration = int(self.simulation_time/op.dt)
                        if iteration % op.hessian_timestep_lag != 0:
                            iteration += 1
                            return
                        first_ts = iteration == i*dt_per_mesh
                        final_ts = iteration == (i+1)*dt_per_mesh
                        dt = op.dt*op.hessian_timestep_lag
                        for j, f in enumerate(adapt_fields):
                            H = hessian(self.fwd_solutions[i], f)
                            if f == 'bathymetry':
                                H_window[j] = H
                            elif op.hessian_time_combination == 'integrate':
                                H_window[j] += (0.5 if first_ts or final_ts else 1.0)*dt*H
                            else:
                                H_window[j] = H if first_ts else metric_intersection(H, H_window[j])

                    def export_func_wrapper():
                        """
                        Extract time-averaged Hessian.

                        NOTE: We only care about the final export in each mesh iteration
                        """
                        export_func()
                        if np.allclose(self.simulation_time, (i+1)*op.dt*dt_per_mesh):
                            for j, H in enumerate(H_window):
                                if op.hessian_time_combination == 'intersect':
                                    H_window[j] *= op.dt*dt_per_mesh
                                H_windows[j][i].interpolate(H_window[j])

                # Solve step for current mesh iteration
                kwargs = {
                    'export_func': export_func_wrapper,
                    'update_forcings': update_forcings_wrapper,
                    'plot_pvd': op.plot_pvd,
                }
                self.solve_forward_step(i, **kwargs)

                # Delete objects to free memory
                H_window = None
                recoverer = None
                export_func = None
                update_forcings = None

            # --- Convergence criteria

            # Check QoI convergence
            if op.qoi_rtol is not None:
                qoi = self.quantity_of_interest()
                self.print("Quantity of interest {:d}: {:.4e}".format(self.outer_iteration+1, qoi))
                self.qois.append(qoi)
                if len(self.qois) > 1:
                    if np.abs(self.qois[-1] - self.qois[-2]) < op.qoi_rtol*self.qois[-2]:
                        self.print("Converged quantity of interest!")
                        break

            # Check maximum number of iterations
            if self.outer_iteration == op.num_adapt-1:
                break

            # --- Time normalise metrics

            for j in range(len(adapt_fields)):
                space_time_normalise(H_windows[j], op=op)

            # Combine metrics (if appropriate)
            metrics = [Function(P1_ten, name="Hessian metric") for P1_ten in self.P1_ten]
            for i in range(self.num_meshes):
                H_window = [H_windows[j][i] for j in range(len(adapt_fields))]
                if 'int' in op.adapt_field:
                    if 'avg' in op.adapt_field:
                        raise NotImplementedError  # TODO: mixed case
                    metrics[i].assign(metric_intersection(*H_window))
                elif 'avg' in op.adapt_field:
                    metrics[i].assign(metric_average(*H_window))
                else:
                    if len(adapt_fields) != 1:
                        msg = "Field for adaptation '{:s}' not recognised"
                        raise ValueError(msg.format(op.adapt_field))
                    metrics[i].assign(H_window[0])
            H_window = None
            H_windows = [[None for P1_ten in self.P1_ten] for f in adapt_fields]

            # metric_file = File(os.path.join(self.di, 'metric.pvd'))
            complexities = []
            for i, M in enumerate(metrics):
                # metric_file.write(M)
                complexities.append(metric_complexity(M))
            self.st_complexities.append(sum(complexities)*op.end_time/op.dt)

            # --- Adapt meshes

            msg = "\nStarting mesh adaptation for iteration {:d}..."
            self.print(msg.format(self.outer_iteration+1))
            for i, M in enumerate(metrics):
                self.print("Adapting mesh {:d}/{:d}...".format(i+1, self.num_meshes))
                self.meshes[i] = pragmatic_adapt(self.meshes[i], M)
            metrics = [None for P1_ten in self.P1_ten]

            # ---  Setup for next run / logging

            self.set_meshes(self.meshes)
            self.setup_all()  # TODO: Manual mode
            self.dofs.append([np.array(V.dof_count).sum() for V in self.V])  # TODO: Manual mode

            self.print("\nResulting meshes")
            msg = "  {:2d}: complexity {:8.1f} vertices {:7d} elements {:7d}"
            num_vertices = self.num_vertices[self.outer_iteration+1]
            num_cells = self.num_cells[self.outer_iteration+1]
            for i, (c, nv, nc) in enumerate(zip(complexities, num_vertices, num_cells)):
                self.print(msg.format(i, c, nv, nc))
            self.print("  total:            {:8.1f}          {:7d}          {:7d}\n".format(
                self.st_complexities[-1], sum(num_vertices)*dt_per_mesh, sum(num_cells)*dt_per_mesh,
            ))

            # Check convergence of *all* element counts
            converged = True
            for i, num_cells_ in enumerate(self.num_cells[self.outer_iteration-1]):
                diff = np.abs(self.num_cells[self.outer_iteration][i] - num_cells_)
                if diff > op.element_rtol*num_cells_:
                    converged = False
            if converged:
                self.print("Converged number of mesh elements!")
                break

            # Increment
            self.outer_iteration += 1
