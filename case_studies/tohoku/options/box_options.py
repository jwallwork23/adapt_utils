import thetis
import numpy as np
from adapt_utils.case_studies.tohoku.options.options import TohokuOptions


__all__ = ["TohokuBoxBasisOptions"]


class TohokuBoxBasisOptions(TohokuOptions):
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
        self.nx = kwargs.get('nx', 1)
        self.ny = kwargs.get('ny', 1)
        N_b = self.nx*self.ny
        control_parameters = kwargs.get('control_parameters', [0.0 for i in range(N_b)])
        N_c = len(control_parameters)
        if N_c != N_b:
            raise ValueError("{:d} controls inconsistent with {:d} basis functions".format(N_c, N_b))

        # Parametrisation of source region
        self.centre_x = kwargs.get('centre_x', 0.7e+06)
        self.centre_y = kwargs.get('centre_y', 4.2e+06)
        self.radius_x = kwargs.get('radius_x', 96e+03 if self.nx == 1 else 48.0e+03)
        self.radius_y = kwargs.get('radius_y', 48e+03 if self.ny == 1 else 24.0e+03)

        # Parametrisation of source basis
        self.print_debug("INIT: Assigning control parameter values...")
        R = thetis.FunctionSpace(self.default_mesh, "R", 0)
        self.control_parameters = []
        for i in range(N_c):
            control = thetis.Function(R, name="Control parameter {:d}".format(i))
            control.assign(control_parameters[i])
            self.control_parameters.append(control)
        self.print_debug("INIT: Done!")
        self.strike_angle = kwargs.get('strike_angle', 7*np.pi/12)

    def get_basis_functions(self):
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
        self.basis_functions = [thetis.Function(prob.V[0]) for i in range(N)]
        R = rotation_matrix(-angle)
        for j, y in enumerate(Y):
            for i, x in enumerate(X):
                psi, phi = self.basis_functions[i + j*nx].split()
                x_rot, y_rot = tuple(np.array([x0, y0]) + np.dot(R, np.array([x, y])))
                phi.interpolate(box([(x_rot, y_rot, rx, ry), ], prob.meshes[0], rotation=angle))
        self.print_debug("INIT: Done!")

    def set_initial_condition(self, prob, sum_pad=100):
        """
        :arg prob: the :class:`AdaptiveProblem` object to which the initial condition is assigned.
        :kwarg sum_pad: when summing terms to assemble the initial surface, the calculation is split
            up for large arrays in order to avoid the UFL recursion limit. That is, every `sum_pad`
            terms are summed separately.
        """
        if not hasattr(self, 'basis_functions'):
            self.get_basis_functions()

        # Assemble initial surface
        self.print_debug("INIT: Assembling initial surface...")
        prob.fwd_solutions[0].assign(0.0)
        controls = self.control_parameters
        for n in range(0, N, sum_pad):
            expr = sum(m*g for m, g in zip(controls[n:n+sum_pad], self.basis_functions[n:n+sum_pad]))
            prob.fwd_solutions[0].assign(prob.fwd_solutions[0] + thetis.project(expr, prob.V[0]))
        self.print_debug("INIT: Done!")

        # Subtract initial surface from the bathymetry field
        self.subtract_surface_from_bathymetry(prob)
