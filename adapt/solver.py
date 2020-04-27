from thetis import *

import numpy as np

from adapt_utils.adapt.adaptation import AdaptiveMesh


__all__ = ["AdaptiveProblem"]


class AdaptiveProblem():
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
    """

    # --- Setup

    def __init__(self, op, meshes, finite_element, discrete_adjoint=True, levels=0, hierarchies=None):
        op.print_debug(op.indent + "{:s} initialisation begin".format(self.__class__.__name__))

        # Read args and kwargs
        self.op = op
        self.finite_element = finite_element
        self.stabilisation = op.stabilisation
        self.discrete_adjoint = discrete_adjoint
        self.approach = op.approach
        self.levels = levels
        if levels > 0:
            raise NotImplementedError  # TODO

        # Setup problem
        self.num_meshes = op.num_meshes
        op.print_debug(op.indent + "SETUP: Building meshes...")
        self.set_meshes(meshes)
        op.print_debug(op.indent + "SETUP: Creating function spaces...")
        self.create_function_spaces()
        op.print_debug(op.indent + "SETUP: Creating solutions...")
        self.create_solutions()
        op.print_debug(op.indent + "SETUP: Creating fields...")
        self.set_fields()
        op.print_debug(op.indent + "SETUP: Setting stabilisation parameters...")
        self.set_stabilisation()
        op.print_debug(op.indent + "SETUP: Setting boundary conditions...")
        self.boundary_conditions = [op.set_boundary_conditions(V) for V in self.V]
        self.fwd_solvers = [None for mesh in self.meshes]  # To be populated
        self.adj_solvers = [None for mesh in self.meshes]  # To be populated

        # Outputs
        self.di = create_directory(self.op.di)
        self.solution_file = File(os.path.join(self.di, 'solution.pvd'))
        # self.solution_fpath_hdf5 = os.path.join(self.di, 'solution.hdf5')
        self.adjoint_solution_file = File(os.path.join(self.di, 'adjoint_solution.pvd'))
        # self.adjoint_solution_fpath_hdf5 = os.path.join(self.di, 'adjoint_solution.hdf5')
        self.indicator_file = File(os.path.join(self.di, 'indicator.pvd'))

        # Storage for diagnostics over mesh adaptation loop
        self.num_cells = [[mesh.num_cells() for mesh in self.meshes], ]
        self.num_vertices = [[mesh.num_vertices() for mesh in self.meshes], ]
        self.dofs = [[np.array(V.dof_count).sum() for V in self.V], ]
        self.indicators = [{} for mesh in self.meshes]
        self.estimators = [{} for mesh in self.meshes]
        self.qois = []


    def set_meshes(self, meshes):  # TODO: levels > 0
        """
        Build an class:`AdaptiveMesh` object associated with each mesh.
        """
        self.meshes = meshes or [self.op.default_mesh for i in range(self.num_meshes)]
        self.ams = [AdaptiveMesh(self.meshes[i], levels=self.levels) for i in range(self.num_meshes)]
        msg = self.op.indent + "SETUP: Mesh {:d} has {:d} elements"
        for i in range(self.num_meshes):
            self.op.print_debug(msg.format(i, self.meshes[i].num_cells()))

    def create_function_spaces(self):  # NOTE: Keep minimal
        """
        Build the finite element space, `V`, for the prognostic solution, along with various other
        useful spaces and test functions and trial functions based upon them.
        """
        fe = self.finite_element
        self.V = [FunctionSpace(mesh, fe) for mesh in self.meshes]
        self.P0 = [FunctionSpace(mesh, "DG", 0) for mesh in self.meshes]
        self.P1 = [FunctionSpace(mesh, "CG", 1) for mesh in self.meshes]
        self.P1_vec = [VectorFunctionSpace(mesh, "CG", 1) for mesh in self.meshes]
        # self.P1_ten = [TensorFunctionSpace(mesh, "CG", 1) for mesh in self.meshes]
        self.P1DG = [FunctionSpace(mesh, "DG", 1) for mesh in self.meshes]
        # self.P1DG_vec = [VectorFunctionSpace(mesh, "DG", 1) for mesh in self.meshes]
        # self.P2 = [FunctionSpace(mesh, "CG", 2) for mesh in self.meshes]
        # self.test = [TestFunction(V) for V in self.V]
        # self.tests = [TestFunctions(V) for V in self.V]
        # self.trial = [TrialFunction(V) for V in self.V]
        # self.trials = [TrialFunctions(V) for V in self.V]
        # self.p0test = [TestFunction(P0) for P0 in self.P0]
        # self.p0trial = [TrialFunction(P0) for P0 in self.P0]

    def create_solutions(self):
        """
        Set up `Function`s in the prognostic space to hold the forward and adjoint solutions.
        """
        self.fwd_solutions = [Function(V, name='Forward solution') for V in self.V]
        self.adj_solutions = [Function(V, name='Adjoint solution') for V in self.V]

    def set_fields(self):
        """Set velocity field, viscosity, etc (on each mesh)."""
        raise NotImplementedError("Should be implemented in derived class.")

    def set_stabilisation(self):
        """ Set stabilisation mode and parameter (on each mesh)."""
        raise NotImplementedError("Should be implemented in derived class.")

    def set_initial_condition(self):
        """Apply initial condition for forward solution on first mesh."""
        self.fwd_solutions[0].assign(self.op.set_start_condition(self.V[0], adjoint=False))

    def set_final_condition(self):
        """Apply final time condition for adjoint solution on final mesh."""
        self.adj_solutions[-1].assign(self.op.set_start_condition(self.V[-1], adjoint=True))

    # --- Helper functions

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

    # --- Solvers

    def setup_solver_forward(self, i, **kwargs):
        """Setup forward solver on mesh `i`."""
        raise NotImplementedError("Should be implemented in derived class.")

    def setup_solver_adjoint(self, i, **kwargs):
        """Setup adjoint solver on mesh `i`."""
        raise NotImplementedError("Should be implemented in derived class.")

    def solve_forward_step(self, i, **kwargs):
        """Solve forward PDE on mesh `i`."""
        raise NotImplementedError("Should be implemented in derived class.")

    def solve_adjoint_step(self, i, **kwargs):
        """Solve adjoint PDE on mesh `i`."""
        raise NotImplementedError("Should be implemented in derived class.")

    def solve_forward(self):
        """Solve forward problem on the full sequence of meshes."""
        for i in range(self.num_meshes):
            if i == 0:
                self.set_initial_condition()
            else:
                self.project_forward_solution(i-1, i)
            self.setup_solver_forward(i)
            self.solve_forward_step(i)

    def solve_adjoint(self):
        """Solve adjoint problem on the full sequence of meshes."""
        for i in range(self.num_meshes - 1, -1):
            if i == self.num_meshes - 1:
                self.set_final_condition()
            else:
                self.project_adjoint_solution(i+1, i)
            self.setup_solver_adjoint(i)
            self.solve_adjoint_step(i)

    def dwr_indication(self, adjoint=False):
        """
        Indicate errors in the quantity of interest by the Dual Weighted Residual method. This is
        inherently problem-dependent.
        """
        raise NotImplementedError("Should be implemented in derived class.")

    def quantity_of_interest(self):
        """Functional of interest which takes the PDE solution as input."""
        raise NotImplementedError("Should be implemented in derived class.")
