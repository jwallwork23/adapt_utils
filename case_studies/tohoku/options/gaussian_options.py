import thetis

import numpy as np
import scipy

from adapt_utils.case_studies.tohoku.options.options import TohokuOptions
from adapt_utils.norms import vecnorm


__all__ = ["TohokuGaussianBasisOptions"]


class TohokuGaussianBasisOptions(TohokuOptions):
    """
    Initialise the free surface with an initial condition consisting of an array of Gaussian basis
    functions, each scaled by a control parameter. The setup with a 13 x 10 array was presented in
    [Saito et al. 2011].

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
        super(TohokuGaussianBasisOptions, self).__init__(**kwargs)
        self.nx = kwargs.get('nx', 1)
        self.ny = kwargs.get('ny', 1)
        N_b = self.nx*self.ny
        control_parameters = kwargs.get('control_parameters', 10.0*np.random.rand(N_b))
        N_c = len(control_parameters)
        if N_c != N_b:
            raise ValueError("{:d} controls inconsistent with {:d} basis functions".format(N_c, N_b))

        # Parametrisation of source region
        self.centre_x = kwargs.get('centre_x', 0.7e+06)
        self.centre_y = kwargs.get('centre_y', 4.2e+06)
        self.extent_x = kwargs.get('extent_x', 560.0e+03)
        self.extent_y = kwargs.get('extent_y', 240.0e+03)

        # Parametrisation of source basis
        self.centre_x = kwargs.get('centre_x', 0.7e+06)
        self.centre_y = kwargs.get('centre_y', 4.2e+06)
        self.extent_x = kwargs.get('extent_x', 560.0e+03)
        self.extent_y = kwargs.get('extent_y', 240.0e+03)

        # Parametrisation of source basis
        self.radius_x = kwargs.get('radius_x', 96e+03 if self.nx == 1 else 48.0e+03)
        self.radius_y = kwargs.get('radius_y', 48e+03 if self.ny == 1 else 24.0e+03)
        self.print_debug("INIT: Assigning control parameter values...")
        R = thetis.FunctionSpace(self.default_mesh, "R", 0)
        self.control_parameters = []
        for i in range(N_c):
            control = thetis.Function(R, name="Control parameter {:d}".format(i))
            control.assign(control_parameters[i])
            self.control_parameters.append(control)
        self.print_debug("INIT: Done!")
        self.strike_angle = kwargs.get('strike_angle', 7*np.pi/12)

    def get_basis_functions(self, fs):
        """
        Assemble an array of (Gaussian) radial basis functions, rotated by specified angle.
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
        X = np.array([0.0, ]) if nx == 1 else np.linspace(-0.5*dx, 0.5*dx, nx)
        Y = np.array([0.0, ]) if ny == 1 else np.linspace(-0.5*dy, 0.5*dy, ny)

        # Assemble an array of Gaussian basis functions, rotated by specified angle
        self.print_debug("INIT: Assembling rotated array of Gaussians...")
        self.basis_functions = [thetis.Function(fs) for i in range(N)]
        R = rotation_matrix(-angle)
        self._array = []
        for j, y in enumerate(Y):
            for i, x in enumerate(X):
                psi, phi = self.basis_functions[i + j*nx].split()
                x_rot, y_rot = tuple(np.array([x0, y0]) + np.dot(R, np.array([x, y])))
                self._array.append([x_rot, y_rot])
                phi.interpolate(gaussian([(x_rot, y_rot, rx, ry), ], fs.mesh(), rotation=angle))
        self.print_debug("INIT: Done!")

    def set_initial_condition(self, prob, sum_pad=100):
        """
        :arg prob: the :class:`AdaptiveProblem` object to which the initial condition is assigned.
        :kwarg sum_pad: when summing terms to assemble the initial surface, the calculation is split
            up for large arrays in order to avoid the UFL recursion limit. That is, every `sum_pad`
            terms are summed separately.
        """
        if not hasattr(self, 'basis_functions'):
            self.get_basis_functions(prob.V[0])

        # Assemble initial surface
        self.print_debug("INIT: Assembling initial surface...")
        prob.fwd_solutions[0].assign(0.0)
        controls = self.control_parameters
        for n in range(0, self.nx*self.ny, sum_pad):
            expr = sum(m*g for m, g in zip(controls[n:n+sum_pad], self.basis_functions[n:n+sum_pad]))
            prob.fwd_solutions[0].assign(prob.fwd_solutions[0] + thetis.project(expr, prob.V[0]))
        self.print_debug("INIT: Done!")

        # Subtract initial surface from the bathymetry field
        self.subtract_surface_from_bathymetry(prob)

    def project(self, prob, source):
        r"""
        Project a source field into the box basis.

        This involves solving an auxiliary optimisation problem! The objective functional is

      ..math::
            J(\mathbf m) = \left( \int_\Omega \sum_i (m_i\phi_i - f) \;\mathrm dx\right)^2,

        where :math:`m` is the control vector, :math:`\boldsymbol\phi` is the vector (radial) basis
        functions and :math:`f` is the source field we seek to represent. Due to the linearity of
        the basis expansion, the objective functional may be rewritten as

      ..math::
            J(\mathbf m) = \left(m\cdot\boldsymbol\Phi - F\right)^2,

        where :math:`\boldsymbol\Phi` and :math:`F` correspond to integrated quantities. These
        quantities can be pre-computed, meaning we just have a linear algebraic optimisation problem.
        """
        if not hasattr(self, 'basis_functions'):
            self.get_basis_functions(prob.V[0])

        # Get RHS (mass of source)
        f = thetis.assemble(source*thetis.dx)

        # Get vector of basis function masses
        g = np.array([thetis.assemble(bf.split()[1]*thetis.dx) for bf in self.basis_functions])

        # Get initial guess by point evaluation (which will probably be an overestimate)
        m_init = np.array([source.at(xy) for xy in self._array])

        def J(m):
            j = (np.dot(g, m) - f)**2
            self.print_debug("INIT: functional = {:8.6e}".format(j))
            return j

        def dJdm(m):
            djdm = 2*(np.dot(g, m) - f)*m
            self.print_debug("INIT: gradient = {:8.6e}".format(vecnorm(djdm, order=np.Inf)))
            return djdm

        # Run BFGS optimisation
        self.print_debug("INIT: Running optimisation to project optimal solution...")
        opt_kwargs = {
            'maxiter': 100,
            'gtol': 1.0e-08,
            'callback': lambda m: self.print_debug("INIT: LINE SEARCH COMPLETE"),
            'fprime': dJdm,
        }
        m_opt = scipy.optimize.fmin_bfgs(J, m_init, **opt_kwargs)
        self.print_debug("INIT: Done!")

        # Assign values
        for i, mi in enumerate(m_opt):
            self.control_parameters[i].assign(mi)

    def interpolate(self, prob, source):
        """
        Interpolate a source field into the box basis using point evaluation.

        This involves solving an auxiliary optimisation problem!
        """
        # TODO: doc
        if not hasattr(self, 'basis_functions'):
            self.get_basis_functions(prob.V[0])
        raise NotImplementedError  # TODO
