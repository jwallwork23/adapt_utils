from __future__ import absolute_import

from thetis import *

import os
import numpy as np

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
    def __init__(self, op, meshes=None, nonlinear=True, checkpointing=False, print_progress=True):
        op.print_debug(op.indent + "{:s} initialisation begin".format(self.__class__.__name__))

        # Read args and kwargs
        self.op = op
        self.stabilisation = op.stabilisation
        self.approach = op.approach
        self.nonlinear = nonlinear
        self.checkpointing = checkpointing
        self.print_progress = print_progress

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
        self.setup_all(meshes)
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
        self.num_cells = [[mesh.num_cells() for mesh in self.meshes], ]
        self.num_vertices = [[mesh.num_vertices() for mesh in self.meshes], ]
        self.dofs = [[np.array(V.dof_count).sum() for V in self.V], ]
        self.indicators = [{} for mesh in self.meshes]
        self.estimators = [{} for mesh in self.meshes]
        self.qois = []
        self.st_complexities = [np.nan]
        self.outer_iteration = 0

    def setup_all(self, meshes):
        """
        Setup everything which isn't explicitly associated with either the forward or adjoint
        problem.
        """
        from thetis.callback import CallbackManager

        op = self.op
        op.print_debug(op.indent + "SETUP: Building meshes...")
        self.set_meshes(meshes)
        op.print_debug(op.indent + "SETUP: Creating function spaces...")
        self.set_finite_elements()
        self.create_function_spaces()
        self.jacobian_signs = [interpolate(sign(JacobianDeterminant(mesh)), P0) for mesh, P0 in zip(self.meshes, self.P0)]
        op.print_debug(op.indent + "SETUP: Creating solutions...")
        self.create_solutions()
        op.print_debug(op.indent + "SETUP: Creating fields...")
        self.set_fields(init = True)
        op.print_debug(op.indent + "SETUP: Setting stabilisation parameters...")
        self.set_stabilisation()
        op.print_debug(op.indent + "SETUP: Setting boundary conditions...")
        self.set_boundary_conditions()
        op.print_debug(op.indent + "SETUP: Creating CallbackManagers...")
        self.callbacks = [CallbackManager() for mesh in self.meshes]
        op.print_debug(op.indent + "SETUP: Creating output files...")
        self.di = create_directory(op.di)
        self.create_outfiles()
        op.print_debug(op.indent + "SETUP: Creating intermediary spaces...")
        self.create_intermediary_spaces()

        # Various empty lists and dicts
        self.equations = [AttrDict() for mesh in self.meshes]
        self.error_estimators = [AttrDict() for mesh in self.meshes]
        self.timesteppers = [AttrDict() for mesh in self.meshes]
        self.kernels = [None for mesh in self.meshes]

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
        for i, mesh in enumerate(self.meshes):
            bnd_len = compute_boundary_length(mesh)
            mesh.boundary_len = bnd_len
            self.op.print_debug(msg.format(i, mesh.num_cells()))
            # if self.op.approach in ('lagrangian', 'ale', 'monge_ampere'):  # TODO
            #     coords = mesh.coordinates
            #     self.mesh_velocities[i] = Function(coords.function_space(), name="Mesh velocity")

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
        raise NotImplementedError("To be implemented in derived class")

    def set_fields(self):
        raise NotImplementedError("To be implemented in derived class")

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

    def create_equations(self, i, adjoint=False):
        if adjoint:
            self.create_adjoint_equations(i)
        else:
            self.create_forward_equations(i)

    def create_forward_equations(self, i):
        raise NotImplementedError("To be implemented in derived class")

    def create_adjoint_equations(self, i):
        raise NotImplementedError("To be implemented in derived class")

    def create_error_estimators(self, i):
        raise NotImplementedError("To be implemented in derived class")

    def create_timesteppers(self, i, adjoint=False):
        if adjoint:
            self.create_adjoint_timesteppers(i)
        else:
            self.create_forward_timesteppers(i)

    def create_forward_timesteppers(self, i):
        raise NotImplementedError("To be implemented in derived class")

    def create_adjoint_timesteppers(self, i):
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

    def store_plexes(self, di=None):
        """Save meshes to disk using DMPlex format."""
        from firedrake.petsc import PETSc

        di = di or os.path.join(self.di, self.approach)
        fname = os.path.join(di, 'plex_{:d}.h5')
        for i, mesh in enumerate(self.meshes):
            assert os.path.isdir(di)
            viewer = PETSc.Viewer().createHDF5(fname.format(i), 'w')
            try:
                viewer(mesh._topology_dm)
            except AttributeError:
                viewer(mesh._plex)  # backwards compatability

    def load_plexes(self, fname):
        """Load meshes in DMPlex format."""
        from firedrake.petsc import PETSc

        for i in range(self.num_meshes):
            newplex = PETSc.DMPlex().create()
            newplex.createFromFile('_'.join([fname, '{:d}.h5'.format(i)]))
            self.meshes[i] = Mesh(newplex)

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
        self.op.print_debug("MESH MOVEMENT: Done!")

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
        self.op.print_debug("MESH MOVEMENT: Done!")

        # Update intermediary mesh coordinates
        self.op.print_debug("MESH MOVEMENT: Updating intermediary mesh coordinates...")
        self.intermediary_meshes[i].coordinates.dat.data[:] = self.mesh_movers[i].x.dat.data
        self.op.print_debug("MESH MOVEMENT: Done!")

        # Project a copy of the current solution onto mesh defined on new coordinates
        self.op.print_debug("MESH MOVEMENT: Projecting solutions onto intermediary mesh...")
        self.project_to_intermediary_mesh(i)
        self.op.print_debug("MESH MOVEMENT: Done!")

        # Update physical mesh coordinates
        self.op.print_debug("MESH MOVEMENT: Updating physical mesh coordinates...")
        self.meshes[i].coordinates.dat.data[:] = self.intermediary_meshes[i].coordinates.dat.data
        self.op.print_debug("MESH MOVEMENT: Done!")

        # Copy over projected solution data
        self.op.print_debug("MESH MOVEMENT: Transferring solution data from intermediary mesh...")
        self.copy_data_from_intermediary_mesh(i)  # FIXME: Needs annotation
        self.op.print_debug("MESH MOVEMENT: Done!")

        # Re-interpolate fields
        self.op.print_debug("MESH MOVEMENT: Re-interpolating fields...")
        self.set_fields()
        self.op.print_debug("MESH MOVEMENT: Done!")
