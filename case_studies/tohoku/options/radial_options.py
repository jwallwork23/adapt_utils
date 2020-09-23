from thetis import *
from pyadjoint.tape import no_annotations

import numpy as np
import scipy

from adapt_utils.case_studies.tohoku.options.options import TohokuInversionOptions
from adapt_utils.norms import vecnorm


__all__ = ["TohokuRadialBasisOptions"]


class TohokuRadialBasisOptions(TohokuInversionOptions):
    """
    Initialise the free surface with an initial condition consisting of an array of radial basis
    functions, each scaled by a control parameter. The setup with a 13 x 10 array was presented in
    [Saito et al. 2011]. By default, the radial basis functions are Gaussians.

    The source region centre is predefined. In the 1D case the basis function is centred at the same
    point. In the case of multiple basis functions they are distributed linearly both perpendicular
    and parallel to the fault axis. Note that support of basis functions is overlapping, unlike the
    case where indicator functions are used.

    The 1D case is useful for inversion experiments because the control parameter space is one
    dimensional, meaning it can be easily plotted.


    [Saito et al.] T. Saito, Y. Ito, D. Inazu, R. Hino, "Tsunami source of the 2011 Tohoku‐Oki
                   earthquake, Japan: Inversion analysis based on dispersive tsunami simulations",
                   Geophysical Research Letters (2011), 38(7).
    """
    def __init__(self, **kwargs):
        """
        :kwarg control_parameters: a list of values to use for the basis function coefficients.
        :kwarg centre_x: x-coordinate of centre of source region in UTM coordinates [m].
        :kwarg centre_y: y-coordinate of centre of source region in UTM coordinates [m].
        :kwarg extent_x: extent of source region along strike direction (i.e. along the fault) [m].
        :kwarg extent_y: extent of source region perpendicular to strike direction [m].
        :kwarg nx: number of basis functions along strike direction.
        :kwarg ny: number of basis functions perpendicular to strike direction.
        :kwarg radius_x: radius of basis function along strike direction [m].
        :kwarg radius_y: radius of basis function perpendicular to strike direction [m].
        :kwarg strike_angle: angle of fault to north [radians].
        """
        super(TohokuRadialBasisOptions, self).__init__(**kwargs)
        self.nx = kwargs.get('nx', 13)
        self.ny = kwargs.get('ny', 10)
        N_b = self.nx*self.ny

        # Parametrisation of source region
        # ================================
        #  These parameters define the geometry of the basis array. Note that if any of the values
        #  vary from the defaults then the cache is marked as dirty and the projection mass matrix
        #  will be re-assembled.
        self.centre_x = self.extract(kwargs, 'centre_x', 0.7e+06)
        self.centre_y = self.extract(kwargs, 'centre_y', 4.2e+06)
        self.extent_x = self.extract(kwargs, 'extent_x', 560.0e+03)
        self.extent_y = self.extract(kwargs, 'extent_y', 240.0e+03)
        self.radius_x = self.extract(kwargs, 'radius_x', 96e+03 if self.nx == 1 else 48.0e+03)
        self.radius_y = self.extract(kwargs, 'radius_y', 48e+03 if self.ny == 1 else 24.0e+03)
        self.strike_angle = self.extract(kwargs, 'strike_angle', 7*np.pi/12)

        # Set control parameters (if provided)
        control_parameters = kwargs.get('control_parameters', 10.0*np.random.rand(N_b))
        if control_parameters is not None:
            if len(control_parameters) != N_b:
                msg = "{:d} controls inconsistent with {:d} basis functions"
                raise ValueError(msg.format(len(control_parameters), N_b))
            self.assign_control_parameters(control_parameters)

    def assign_control_parameters(self, control_values, mesh=None):
        """
        Create a list of control parameters defined in the R space and assign values from the list
        of floats, `control_values`.
        """
        self.print_debug("INIT: Assigning control parameter values...")
        R = FunctionSpace(mesh or self.default_mesh, "R", 0)
        self.control_parameters = []
        for i, control_value in enumerate(control_values):
            control = Function(R, name="Control parameter {:d}".format(i))
            control.assign(control_value)
            self.control_parameters.append(control)

    @no_annotations
    def get_basis_functions(self, fs):  # TODO: Implement radial basis functions other than Gaussians
        """
        Assemble an array of radial basis functions, rotated by specified angle.
        """
        from adapt_utils.misc import gaussian, rotation_matrix

        # Gather parameters
        x0, y0 = self.centre_x, self.centre_y  # Centre of basis region
        nx, ny = self.nx, self.ny              # Number of basis functions along each axis
        N = nx*ny                              # Total number of basis functions
        rx, ry = self.radius_x, self.radius_y  # Radius of each basis function in each direction
        dx, dy = self.extent_x, self.extent_y  # Extent of basis region in each direction
        angle = self.strike_angle              # Angle by which to rotate axis / basis array

        # Setup array coordinates
        X = np.array([0.0]) if nx == 1 else np.linspace(-0.5*dx, 0.5*dx, nx)
        Y = np.array([0.0]) if ny == 1 else np.linspace(-0.5*dy, 0.5*dy, ny)

        # Assemble array
        self.print_debug("INIT: Assembling rotated radial basis function array...")
        self.basis_functions = [Function(fs) for i in range(N)]
        R = rotation_matrix(-angle)
        self._array = []
        for j, y in enumerate(Y):
            for i, x in enumerate(X):
                psi, phi = self.basis_functions[i + j*nx].split()
                x_rot, y_rot = tuple(np.array([x0, y0]) + np.dot(R, np.array([x, y])))
                self._array.append([x_rot, y_rot])
                phi.interpolate(gaussian([(x_rot, y_rot, rx, ry)], fs.mesh(), rotation=angle))

    @property
    def basis_function(self):
        return self.basis_functions[0]

    @property
    def control_parameter(self):
        return self.control_parameters[0]

    def set_initial_condition(self, prob):
        """
        Project from the radial basis into the prognostic space used within the tsunami propagation
        model.

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
        r"""
        Project a source field into the radial basis. This involves solving an auxiliary linear system.
        We seek a vector of control parameters :math:`\mathbf m` satisfying

      ..math::
            \sum_i m_i\phi_i = f,

        where :math:`\boldsymbol\phi` is the vector (radial) basis functions and :math:`f` is the
        source field we seek to represent. For an L2 projection we achieve this by solving the
        problem weakly:

      ..math::
            \sum_j \sum_i \int_\Omega m_i\phi_i\phi_j \;\mathrm dx =
                \sum_j \int_\Omega f\phi_j \;\mathrm dx, \quad \forall j.

        For ease, we simply assemble the mass matrix and RHS vector and solve using NumPy's `solve`.
        """
        cache_dir = create_directory(os.path.join(os.path.dirname(__file__), '.cache'))
        fname = os.path.join(cache_dir, 'mass_matrix_radial_{:d}_{:d}x{:d}.npy')
        fname = fname.format(self.level, self.nx, self.ny)

        # Get basis functions
        if not hasattr(self, 'basis_functions'):
            self.get_basis_functions(prob.V[0])
        phi = [bf.split()[1] for bf in self.basis_functions]
        N = self.nx*self.ny

        # Assemble mass matrix
        if os.path.isfile(fname) and not self.dirty_cache:
            self.print_debug("PROJECTION: Loading mass matrix from cache...")
            A = np.load(fname)
        else:
            self.print_debug("PROJECTION: Assembling mass matrix...")
            A = np.zeros((N, N))
            for i in range(N):
                for j in range(i+1):
                    A[i, j] = assemble(phi[i]*phi[j]*dx)
            for i in range(N):
                for j in range(i+1, N):
                    A[i, j] = A[j, i]
            self.print_debug("PROJECTION: Cacheing mass matrix...")
            np.save(fname, A)

        # Assemble RHS
        self.print_debug("PROJECTION: Assembling RHS...")
        b = np.array([assemble(phi[i]*source*dx) for i in range(N)])

        # Project
        self.print_debug("PROJECTION: Solving linear system...")
        m = np.linalg.solve(A, b)

        # Assign values
        self.assign_control_parameters(m, prob.meshes[0])

    @no_annotations
    def interpolate(self, prob, source):
        r"""
        Interpolate a source field into the radial basis using point evaluation.

        This involves solving an auxiliary optimisation problem! The objective functional is

      ..math::
            J(\mathbf m) = \tilde J(\mathbf m) \cdot \tilde J(\mathbf m),

        where :math:`m` is the control vector, :math:`\boldsymbol\phi` is the vector (radial) basis
        functions, :math:`f` is the source field we seek to represent and

      ..math::
            \tilde J(\mathbf m)_i = \frac1N \int_\Omega \sum_i (m_i\phi_i - f_i) \;\mathrm dx.

        Here :math:`N` is the number of radial basis functions. Due to the linearity of the basis
        expansion, the objective functional may be rewritten as

      ..math::
            J(\mathbf m) = \frac1{N^2} (m\cdot\boldsymbol\Phi - F) \cdot (m\cdot\boldsymbol\Phi - F),

        where :math:`\boldsymbol\Phi` and :math:`F` correspond to integrated quantities. These
        quantities can be pre-computed, meaning we just have a linear algebraic optimisation problem.
        """
        if not hasattr(self, 'basis_functions'):
            self.get_basis_functions(prob.V[0])
        N = self.nx*self.ny

        # Get RHS
        f = np.array([source.at(xy) for xy in self._array])

        # Get vector of pointwise basis function values
        g = np.array([[bf.split()[1].at(xy) for bf in self.basis_functions] for xy in self._array])

        # Get initial guess by point evaluation (which will probably be an overestimate)
        m_init = f.copy()

        # Rescale to avoid precision loss
        rescaling = 1/N**2

        def J(m):
            j = rescaling*sum([(np.dot(g[i, :], m) - f[i])**2 for i in range(N)])
            self.print_debug("INTERPOLATION: functional = {:8.6e}".format(j))
            return j

        def dJdm(m):
            djdm = 2*rescaling*sum([(np.dot(g[i, :], m) - f[i])*g[i, :] for i in range(N)])
            self.print_debug("INTERPOLATION: gradient = {:8.6e}".format(vecnorm(djdm, order=np.Inf)))
            return djdm

        # Run BFGS optimisation
        self.print_debug("INTERPOLATION: Running optimisation to interpolate optimal solution...")
        opt_kwargs = {
            'maxiter': 10000,
            'gtol': 1.0e-08,
            'callback': lambda m: self.print_debug("INTERPOLATION: LINE SEARCH COMPLETE"),
            'fprime': dJdm,
        }
        m_opt = scipy.optimize.fmin_bfgs(J, m_init, **opt_kwargs)

        # Assign values
        self.assign_control_parameters(m_opt, prob.meshes[0])
