from thetis import *
from pyadjoint.tape import no_annotations

import numpy as np

from adapt_utils.case_studies.tohoku.options.options import TohokuInversionOptions


__all__ = ["TohokuBoxBasisOptions"]


class TohokuBoxBasisOptions(TohokuInversionOptions):
    """
    Initialise the free surface with an initial condition consisting of an array of rectangular
    indicator functions, each scaled by a control parameter. Setups of this type have been used by
    numerous authors in the tsunami modelling literature.

    The source region centre is predefined. In the 1D case the basis function is centred at the same
    point. In the case of multiple basis functions they are distributed linearly both perpendicular
    and parallel to the fault axis. Note that the support does not overlap, unlike with radial basis
    functions.

    The 1D case is useful for inversion experiments because the control parameter space is one
    dimensional, meaning it can be easily plotted.
    """
    def __init__(self, **kwargs):
        """
        :kwarg control_parameters: a list of values to use for the basis function coefficients.
        :kwarg centre_x: x-coordinate of centre of source region in UTM coordinates [m].
        :kwarg centre_y: y-coordinate of centre of source region in UTM coordinates [m].
        :kwarg nx: number of basis functions along strike direction (i.e. along the fault).
        :kwarg ny: number of basis functions perpendicular to the strike direction.
        :kwarg radius_x: radius of basis function along strike direction [m].
        :kwarg radius_y: radius of basis function perpendicular to the strike direction [m].
        :kwarg strike angle: angle of fault to north [radians].
        """
        super(TohokuBoxBasisOptions, self).__init__(**kwargs)
        self.nx = kwargs.get('nx', 13)
        self.ny = kwargs.get('ny', 10)
        N_b = self.nx*self.ny
        control_parameters = kwargs.get('control_parameters', 10.0*np.random.rand(N_b))  # arbitrary
        N_c = len(control_parameters)
        if N_c != N_b:
            raise ValueError("{:d} controls inconsistent with {:d} basis functions".format(N_c, N_b))

        # Parametrisation of source region
        self.centre_x = kwargs.get('centre_x', 0.7e+06)
        self.centre_y = kwargs.get('centre_y', 4.2e+06)
        self.radius_x = kwargs.get('radius_x', 0.5*560.0e+03/self.nx)
        self.radius_y = kwargs.get('radius_y', 0.5*240.0e+03/self.ny)

        # Parametrisation of source basis
        self.strike_angle = kwargs.get('strike_angle', 7*np.pi/12)
        self.assign_control_parameters(control_parameters)

    def assign_control_parameters(self, control_values):
        """
        Create a list of control parameters defined in the R space and assign values from the list
        of floats, `control_values`.
        """
        self.print_debug("INIT: Assigning control parameter values...")
        R = FunctionSpace(self.default_mesh, "R", 0)  # TODO: Don't use default mesh
        self.control_parameters = []
        for i, control_value in enumerate(control_values):
            control = Function(R, name="Control parameter {:d}".format(i))
            control.assign(control_value)
            self.control_parameters.append(control)

    @no_annotations
    def get_basis_functions(self, fs):
        """
        Assemble an array of piecewise constant indicator functions, rotated by specified angle.
        """
        from adapt_utils.misc import box, rotation_matrix

        # Gather parameters
        x0, y0 = self.centre_x, self.centre_y  # Centre of basis region
        nx, ny = self.nx, self.ny              # Number of basis functions along each axis
        N = nx*ny                              # Total number of basis functions
        rx, ry = self.radius_x, self.radius_y  # Radius of each basis function in each direction
        angle = self.strike_angle              # Angle by which to rotate axis / basis array

        # Setup array coordinates
        X = np.linspace((1 - nx)*rx, (nx - 1)*rx, nx)
        Y = np.linspace((1 - ny)*ry, (ny - 1)*ry, ny)
        self.print_debug("INIT: Assembling rotated array of indicator functions...")
        self.basis_functions = [Function(fs) for i in range(N)]
        R = rotation_matrix(-angle)
        self._array = []
        for j, y in enumerate(Y):
            for i, x in enumerate(X):
                psi, phi = self.basis_functions[i + j*nx].split()
                x_rot, y_rot = tuple(np.array([x0, y0]) + np.dot(R, np.array([x, y])))
                self._array.append([x_rot, y_rot])
                phi.interpolate(box([(x_rot, y_rot, rx, ry), ], fs.mesh(), rotation=angle))

    def set_initial_condition(self, prob):
        """
        Project from the piecewise constant basis into the prognostic space used within the tsunami
        propagation model.

        :arg prob: the :class:`AdaptiveProblem` object to which the initial condition is assigned.
        """
        if not hasattr(self, 'basis_functions'):
            self.get_basis_functions(prob.V[0])

        # Assemble initial surface
        self.print_debug("INIT: Assembling initial surface...")
        q = prob.fwd_solutions[0]
        q.assign(0.0)
        for coeff, bf in zip(self.control_parameters, self.basis_functions):
            q.assign(q + project(coeff*bf, prob.V[0]))

        # Subtract initial surface from the bathymetry field
        self.subtract_surface_from_bathymetry(prob)

    @no_annotations
    def project(self, prob, source):
        """
        Project a source field into the box basis using a simple L2 projection.

        Note that the approach relies on the fact that the supports of the basis functions do not
        overlap.
        """
        cache_dir = create_directory(os.path.join(os.path.dirname(__file__), '.cache'))
        fname = os.path.join(cache_dir, 'mass_matrix_box_{:d}.npy'.format(self.level))

        # Get basis functions
        if not hasattr(self, 'basis_functions'):
            self.get_basis_functions(prob.V[0])
        basis_functions = [bf.split()[1] for bf in self.basis_functions]

        # Assemble mass matrix
        if os.path.isfile(fname):
            self.print_debug("PROJECTION: Loading mass matrix from cache...")
            A = np.load(fname)
        else:
            self.print_debug("PROJECTION: Assembling mass matrix...")
            A = np.array([assemble(phi*phi*dx) for phi in basis_functions])
            self.print_debug("PROJECTION: Cacheing mass matrix...")
            np.save(fname, A)

        # Assemble RHS
        self.print_debug("PROJECTION: Assembling RHS...")
        b = np.array([assemble(phi*source*dx) for phi in basis_functions])

        # Project
        self.print_debug("PROJECTION: Solving linear system...")
        m = b/A

        # Assign values
        self.print_debug("PROJECTION: Assigning values...")
        for i, mi in enumerate(m):
            self.control_parameters[i].assign(mi)

    @no_annotations
    def interpolate(self, prob, source):
        """
        Interpolate a source field into the box basis using point evaluation.

        Note that the approach relies on the fact that the supports of the basis functions do not
        overlap.
        """
        if not hasattr(self, 'basis_functions'):
            self.get_basis_functions(prob.V[0])
        for i, xy in enumerate(self._array):
            self.control_parameters[i].assign(source.at(xy))
