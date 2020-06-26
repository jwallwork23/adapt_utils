from __future__ import absolute_import

from thetis import *
from thetis.callback import CallbackManager
from firedrake.petsc import PETSc

import os
import numpy as np

from adapt_utils.adapt.r import MeshMover
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
    def __init__(self, op, meshes=None, discrete_adjoint=True, nonlinear=True):
        op.print_debug(op.indent + "{:s} initialisation begin".format(self.__class__.__name__))

        # Read args and kwargs
        self.op = op
        self.stabilisation = op.stabilisation
        self.discrete_adjoint = discrete_adjoint
        self.approach = op.approach
        self.nonlinear = nonlinear

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
        implemented_steppers = {  # TODO: Other timesteppers
            'CrankNicolson': CrankNicolson,
            'SteadyState': SteadyState,
        }
        assert op.timestepper in implemented_steppers
        self.integrator = implemented_steppers[self.op.timestepper]
        if op.timestepper == 'SteadyState':
            assert op.end_time < op.dt

        # Mesh movement
        self.mesh_movers = [None for i in range(self.num_meshes)]

        # Outputs
        self.bathymetry_file = File(os.path.join(self.di, 'bathymetry.pvd'))
        self.indicator_file = File(os.path.join(self.di, 'indicator.pvd'))
        self.kernel_file = File(os.path.join(self.di, 'kernel.pvd'))

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
        op = self.op
        op.print_debug(op.indent + "SETUP: Building meshes...")
        self.set_meshes(meshes)
        op.print_debug(op.indent + "SETUP: Creating function spaces...")
        self.set_finite_elements()
        self.create_function_spaces()
        op.print_debug(op.indent + "SETUP: Creating solutions...")
        self.create_solutions()
        op.print_debug(op.indent + "SETUP: Creating fields...")
        self.set_fields()
        op.print_debug(op.indent + "SETUP: Setting stabilisation parameters...")
        self.set_stabilisation()
        op.print_debug(op.indent + "SETUP: Setting boundary conditions...")
        self.set_boundary_conditions()
        op.print_debug(op.indent + "SETUP: Creating CallbackManagers...")
        self.callbacks = [CallbackManager() for mesh in self.meshes]
        op.print_debug(op.indent + "SETUP: Creating output files...")
        self.di = create_directory(op.di)
        self.create_outfiles()
        self.equations = [AttrDict() for mesh in self.meshes]
        self.error_estimators = [AttrDict() for mesh in self.meshes]
        self.timesteppers = [AttrDict() for mesh in self.meshes]
        self.kernels = [None for mesh in self.meshes]

    def set_meshes(self, meshes):
        """
        Build a mesh associated with each mesh.

        NOTE: If a single mesh is passed to the constructor then it is symlinked into each slot
              rather than explicitly copied. This rears its head in :attr:`run_dwr`, where a the
              enriched meshes are build from a single mesh hierarchy.
        """
        self.meshes = meshes or [self.op.default_mesh for i in range(self.num_meshes)]
        msg = self.op.indent + "SETUP: Mesh {:d} has {:d} elements"
        for i, mesh in enumerate(self.meshes):
            bnd_len = compute_boundary_length(mesh)
            mesh.boundary_len = bnd_len
            self.op.print_debug(msg.format(i, mesh.num_cells()))

    def set_finite_elements(self):
        raise NotImplementedError("To be implemented in derived class")

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

    def set_final_condition(self):
        self.op.set_final_condition(self)

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
            self.set_final_condition()
        else:
            self.project_adjoint_solution(i+1, i)

    def store_plexes(self, di=None):
        """Save meshes to disk using DMPlex format."""
        di = di or os.path.join(self.di, self.approach)
        fname = os.path.join(di, 'plex_{:d}.h5')
        for i, mesh in enumerate(self.meshes):
            assert os.path.isdir(di)
            viewer = PETSc.Viewer().createHDF5(fname.format(i), 'w')
            viewer(mesh._plex)

    def load_plexes(self, fname):
        """Load meshes in DMPlex format."""
        for i in range(self.num_meshes):
            newplex = PETSc.DMPlex().create()
            newplex.createFromFile('_'.join([fname, '{:d}.h5'.format(i)]))
            self.meshes[i] = Mesh(newplex)

    def solve(self, adjoint=False, **kwargs):
        """
        Solve the forward or adjoint problem (as specified by the `adjoint` boolean kwarg) on the
        full sequence of meshes.

        NOTE: The implementation contains a very simple checkpointing scheme, in the sense that
            the final solution computed on mesh `i` is stored in `self.fwd_solutions[i]` or
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
        self.op.set_qoi_kernel(self, i)

    # --- Mesh movement

    def set_monitor_functions(self, monitors):
        assert self.approach in ('monge_ampere', 'laplacian_smoothing', 'ale')
        assert monitors is not None
        if callable(monitors):
            monitors = [monitors, ]
        assert len(monitors) == self.num_meshes
        kwargs = {
            'method': self.approach,
            'mesh_velocity': None,  # TODO
            'bc': None,  # TODO
            'bbc': None,  # TODO
            'op': self.op,
        }
        for i in range(self.num_meshes):
            assert monitors[i] is not None
            self.mesh_movers[i] = MeshMover(self.meshes[i], monitors[i], **kwargs)

    def move_mesh(self, i):
        if self.mesh_movers[i] is not None:
            self.mesh_movers[i].adapt()
