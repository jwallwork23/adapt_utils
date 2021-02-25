from thetis import *

import numpy as np

from adapt_utils.case_studies.tohoku.options.options import TohokuInversionOptions


__all__ = ["TohokuOkadaBasisOptions"]


class TohokuOkadaBasisOptions(TohokuInversionOptions):
    """
    Initialise the free surface with an initial condition generated using Okada functions.

    Note that, unlike in the basis comprised of an array of piecewise constant or radial basis
    functions, the relationship between the control parameters and the initial surface is nonlinear.
    In addition, zero is not a feasible initial guess for the Okada parameters, meaning some physical
    intuition is required in order to set up the problem.

    Control parameters comprise of a dictionary of lists containing the following parameters. The
    list index corresponds to the subfault within the main fault. By default, a 19 x 10 grid of
    25km x 20km subfaults is considered, using the work of [Shao et al. 2011].

      * 'depth'     - depth of the centroid of the subfault plane [m].
      * 'length'    - length of the subfault plane [m].
      * 'width'     - width of the subfault plane [m].
      * 'slip'      - average displacement [m].
      * 'strike'    - angle from north of subfault [radians].
      * 'dip'       - angle from horizontal [radians].
      * 'rake'      - slip of one subfault block compared to another [radians].
      * 'latitude'  - latitude of the centroid of the subfault [degrees].
      * 'longitude' - longitude of the centroid of the subfault [degrees].

    [Shao et al. 2011] G. Shao, X. Li, C. Ji & T. Maeda, "Focal mechanism and slip history of the
                       2011 Mw 9.1 off the Pacific coast of Tohoku Earthquake, constrained with
                       teleseismic body and surface waves", Earth, Planets and Space 63.7 (2011),
                       p.559--564.
    """

    # --- Initialisation

    def __init__(self, **kwargs):
        """
        :kwarg control_parameters: a dictionary of values to use for the basis function coefficients.
        :kwarg centre_x: x-coordinate of centre of source region in UTM coordinates.
        :kwarg centre_y: y-coordinate of centre of source region in UTM coordinates.
        :kwarg nx: number of sub-faults along strike direction.
        :kwarg ny: number of sub-faults perpendicular to the strike direction.
        :kwarg okada_grid_resolution: integer N giving rise to an N x N grid upon which to evaluate
            the Okada functions and hence the resulting fault surface.
        :kwarg okada_grid_length_lon: extent of the Okada grid in the longitudinal direction.
        :kwarg okada_grid_length_lat: extent of the Okada grid in the latitudinal direction.
        :kwarg okada_grid_lon_min: minimum longitude in the Okada grid.
        :kwarg okada_grid_lat_min: minimum latitude in the Okada grid.
        """
        super(TohokuOkadaBasisOptions, self).__init__(**kwargs)
        self.assign_control_parameters(kwargs.get('control_parameters'))
        self.coordinate_specification = kwargs.get('coordinate_specification', 'centroid')
        self.N = kwargs.get('okada_grid_resolution', None)
        self.lx = kwargs.get('okada_grid_length_lon', 10)
        self.ly = kwargs.get('okada_grid_length_lat', 10)
        self.xmin = kwargs.get('okada_grid_lon_min', 138)
        self.ymin = kwargs.get('okada_grid_lat_min', 32)
        self.xmax = kwargs.get('okada_grid_lon_max', self.xmin + self.lx)
        self.ymax = kwargs.get('okada_grid_lat_max', self.ymin + self.ly)
        self.subfaults = []
        self.get_subfaults()
        self.all_controls = (
            'latitude', 'longitude', 'depth',
            'slip', 'rake', 'strike', 'dip',
            'length', 'width',
        )

        # Numbers of sub-faults in each direction
        self.nx = kwargs.get('nx', 19)
        self.ny = kwargs.get('ny', 10)
        msg = "nx = {:d}, ny = {:d} inconsistent with {:d} subfaults"
        self.num_subfaults = len(self.subfaults)
        assert self.nx*self.ny == self.num_subfaults, msg.format(self.nx, self.ny, self.num_subfaults)

        # Active controls for automatic differentation
        self.active_controls = ('slip', 'rake', 'strike', 'dip')

    def download_okada_parameters(self):
        """
        Download the Okada parameters generated by [Shao et al. 2011] from the webpage
          `http://ji.faculty.geol.ucsb.edu/big_earthquakes/2011/03/0311_v3/result_c/static_out`.

        A 19 x 10 array of Okada functions was assumed, each of which with a length of 25km and a
        width of 20km. Note that the webpage also provides initial times, rise times and end times
        for each subfault, although we do not use these here.
        """
        # TODO: Use geoclaw's inbuilt functionality for reading from UCSB format
        import os
        import urllib.request

        # Read data from file or downloaded it
        fname = os.path.join(self.resource_dir, 'surf', 'okada_parameters.txt')
        url = "http://ji.faculty.geol.ucsb.edu/big_earthquakes/2011/03/0311_v3/result_c/static_out"
        if os.path.exists(fname):
            data = open(fname, 'r').readlines()
        else:
            with urllib.request.urlopen(url) as fp:
                data_bytes = fp.read()                # read webpage as bytes
                data_str = data_bytes.decode("utf8")  # convert to a string
                with open(fname, 'w+') as f:
                    f.write(data_str)                 # save for future use
                data = data_str.split('\n')

        # Create a dictionary to contain the parameters
        self.all_controls = ('latitude', 'longitude', 'depth', 'slip', 'rake', 'strike', 'dip')
        self.control_parameters = {}
        for control in self.all_controls:
            self.control_parameters[control] = []
        self.control_parameters['length'] = []
        self.control_parameters['width'] = []

        # Extract the data and enter it into the dictionary
        for i, line in enumerate(data):
            if i < 12:
                continue
            for word, control in zip(line.split(), self.all_controls):
                val = float(word)
                if control == 'slip':
                    val /= 100  # convert from cm to m
                if control == 'depth':
                    val *= 1000  # convert from km to m
                self.control_parameters[control].append(val)
            if line not in ('', '\n'):
                self.control_parameters['length'].append(25.0e+03)
                self.control_parameters['width'].append(20.0e+03)
        self.all_controls += ('length', 'width', )

    def assign_control_parameters(self, control_values):
        """
        For consistency with :class:`TohokuBoxBasisOptions` and :class:`TohokuRadialBasisOptions`.
        """
        self.control_parameters = control_values

    def get_subfaults(self, check_validity=False, reset=False):
        """
        Create GeoCLAW :class:`SubFault` objects from provided subfault parameters, as well as a
        :class`Fault` object.

        If control parameters were not provided then data are downloaded according to the
        `download_okada_parameters` method.
        """
        from adapt_utils.swe.tsunami.dtopotools import Fault, SubFault

        # Reset subfaults if requested
        if reset:
            self.subfaults = []

        # If no Okada parameters have been provided then download them from a default webpage
        if self.control_parameters is None or self.control_parameters == {}:
            self.download_okada_parameters()

        # Check consistency of the inputs
        num_subfaults = len(self.control_parameters['latitude'])
        msg = "Mismatching '{:s}' data: {:d} controls vs. {:d} subfaults"
        for var in self.control_parameters:
            num_ctrls = len(self.control_parameters[var])
            assert num_ctrls == num_subfaults, msg.format(var, num_ctrls, num_subfaults)

        # Check validity of the inputs
        if check_validity:
            for i in range(num_subfaults):
                for var in ('depth', 'slip'):
                    value = self.control_parameters[var][i]
                    assert value > 0, "{:s} is not allowed to be negative".format(var.capitalize())
                lon = self.control_parameters['longitude'][i]
                lat = self.control_parameters['latitude'][i]
                self.check_in_domain([lon, lat])

        # Create subfaults
        for i in range(num_subfaults):
            subfault = SubFault()
            subfault.coordinate_specification = self.coordinate_specification
            self.subfaults.append(subfault)

        # Create a lon-lat grid upon which to represent the source
        if self.N is not None:
            x = np.linspace(self.xmin, self.xmax, self.N)
            y = np.linspace(self.ymin, self.ymax, self.N)
            self.coords = (x, y)
        else:
            self.indices = []
            self.coords = []
            for i, xy in enumerate(self.lonlat_mesh.coordinates.dat.data):
                if (self.xmin < xy[0] < self.xmax) and (self.ymin < xy[1] < self.ymax):
                    self.indices.append(i)
                    self.coords.append(xy)
            if len(self.coords) == 0:
                raise ValueError("Source region does not lie in domain.")
            self.coords = (np.array(self.coords), )

        # Create fault
        self.fault = Fault(*self.coords, subfaults=self.subfaults)

    def interpolate_dislocation_field(self, prob, data=None):
        from scipy.interpolate import griddata

        surf = Function(prob.P1[0])
        if data is not None:
            surf.dat.data[self.indices] = data
        elif self.N is not None:  # Interpolate it using SciPy
            surf.dat.data[:] = griddata(
                (self.fault.dtopo.X, self.fault.dtopo.Y),
                self.fault.dtopo.dZ.reshape(self.fault.dtopo.X.shape),
                self.coords,
                method='linear',
                fill_value=0.0,
            )
        else:  # Just insert the data at the appropriate nodes, assuming zero elsewhere
            surf.dat.data[self.indices] = self.fault.dtopo.dZ.reshape(self.fault.dtopo.X.shape)
        return surf

    def set_initial_condition(self, prob, annotate_source=False, unroll_tape=False, **kwargs):
        """
        Set initial condition using the Okada parametrisation [Okada 85].

        Uses code from GeoCLAW found in `geoclaw/src/python/geoclaw/dtopotools.py`.

        [Okada 85] Yoshimitsu Okada, "Surface deformation due to shear and tensile faults in a
                   half-space", Bulletin of the Seismological Society of America, Vol. 75, No. 4,
                   pp.1135--1154, (1985).

        :arg prob: :class:`AdaptiveTsunamiProblem` solver object.
        :kwarg annotate_source: toggle annotation of the rupture process using pyadolc.
        :kwarg tag: non-negative integer label for tape.
        """
        # separate_faults = kwargs.get('separate_faults', True)
        separate_faults = kwargs.get('separate_faults', False)
        subtract_from_bathymetry = kwargs.pop('subtract_from_bathymetry', True)
        tag = kwargs.get('tag', 0)

        # Create fault topography...
        if unroll_tape:
            # ... by unrolling ADOL-C's tape
            import adolc
            F = adolc.zos_forward(tag, self.input_vector, keep=1)
            if separate_faults:
                F = np.sum(F.reshape(self.num_subfaults, len(self.indices)), axis=0)
            surf = self.interpolate_dislocation_field(prob, data=F)
        else:
            # ... by running the Okada model
            self.create_topography(annotate=annotate_source, **kwargs)
            surf = self.interpolate_dislocation_field(prob)

        # Assume zero initial velocity and interpolate into the elevation space
        u, eta = prob.fwd_solutions[0].split()
        u.assign(0.0)
        eta.interpolate(surf)

        # Subtract initial surface from the bathymetry field
        if subtract_from_bathymetry:
            self.subtract_surface_from_bathymetry(prob, surf=surf)
        return surf

    def get_basis_functions(self, prob=None):
        """
        Assemble a dictionary containing lists of Okada basis functions on each subfault.

        Each basis function associated with a subfault has active controls set to zero on all other
        subfaults and only one non-zero active control on the subfault itself, set to one. All passive
        controls retain the value that they hold before assembly.
        """
        from adapt_utils.unsteady.solver import AdaptiveProblem

        prob = prob or AdaptiveProblem(self)
        self._basis_functions = {}

        # Stash the control parameters and zero out all active ones
        tmp = self.control_parameters.copy()
        for control in self.active_controls:
            self.control_parameters[control] = np.zeros(self.num_subfaults)

        # Loop over active controls on each subfault and compute the associated basis functions
        msg = "INIT: Assembling Okada basis function array with active controls {:}..."
        print_output(msg.format(self.active_controls))
        msg = "INIT: Assembling '{:s}' basis function on subfault {:d}/{:d}..."
        for control in self.active_controls:
            self._basis_functions[control] = []
            for i, subfault in enumerate(self.subfaults):
                self.print_debug(msg.format(control, i, self.num_subfaults), mode='full')
                self.control_parameters[control][i] = 1
                self.set_initial_condition(prob, annotate_source=False, subtract_from_bathymetry=False)
                self._basis_functions[control].append(prob.fwd_solutions[0].copy(deepcopy=True))
                self.control_parameters[control][i] = 0
        self.control_parameters = tmp

    @property
    def basis_functions(self):
        recompute = False
        if not hasattr(self, '_basis_functions'):
            recompute = True
        else:
            # Check the active controls haven't changed
            for control in self.active_controls:
                if control not in self._basis_functions:
                    recompute = True
            for control in self._basis_functions:
                if control not in self.active_controls:
                    recompute = True
        if recompute:
            self.get_basis_functions()
        return self._basis_functions

    def create_topography(self, annotate=False, interpolate=False, **kwargs):
        """
        Compute the topography dislocation due to the earthquake using the Okada model. This
        implementation makes use of the :class:`Fault` and :class:`SubFault` objects from GeoClaw.

        If annotation is turned on, the rupture process is annotated using the Python wrapper
        `pyadolc` to the C++ operator overloading automatic differentation tool ADOL-C.

        :kwarg annotate: toggle annotation using pyadolc.
        :kwarg interpolate: see the :attr:`interpolate` method.
        :kwarg tag: label for tape.
        :kwarg separate_faults: set to `True` if we want forward mode derivatives.
        """
        msg = "INIT: Fault corresponds to an earthquake with moment magnitude {:4.1e}"
        if annotate:
            if interpolate:
                self._create_topography_active_interpolate(**kwargs)
            else:
                self._create_topography_active(**kwargs)
            if self.debug and self.debug_mode == 'full':
                try:
                    print(msg.format(self.fault.Mw().val))
                except Exception:
                    print(msg.format(self.fault.Mw()))
        else:
            self._create_topography_passive()
            self.print_debug(msg.format(self.fault.Mw()), mode='full')

    def _create_topography_passive(self):
        msg = "INIT: Subfault {:d}: shear modulus {:4.1e} Pa, seismic moment is {:4.1e}"
        for i, subfault in enumerate(self.subfaults):
            for control in self.all_controls:
                subfault.__setattr__(control, self.control_parameters[control][i])
            if self.debug and self.debug_mode == 'full':
                print(msg.format(i, subfault.mu, subfault.Mo()))
        self.fault.create_dtopography(verbose=self.debug and self.debug_mode == 'full', active=False)

    # --- Automatic differentiation

    def _create_topography_active(self, tag=0, separate_faults=True):
        import adolc

        # Sanitise kwargs
        assert isinstance(tag, int)
        assert tag >= 0
        for control in self.active_controls:
            assert control in self.all_controls, "Active control '{:s}' not recognised.".format(control)

        # Initialise tape
        adolc.trace_on(tag)

        # Read parameters and mark active variables as independent
        msg = "INIT: Subfault {:d}: shear modulus {:4.1e} Pa, seismic moment is {:4.1e}"
        for i, subfault in enumerate(self.subfaults):
            for control in self.all_controls:
                if control in self.active_controls:
                    subfault.__setattr__(control, adolc.adouble(self.control_parameters[control][i]))
                    adolc.independent(subfault.__getattribute__(control))
                else:
                    subfault.__setattr__(control, self.control_parameters[control][i])
            if self.debug and self.debug_mode == 'full':
                try:
                    print(msg.format(i, subfault.mu, subfault.Mo().val))
                except Exception:
                    print(msg.format(i, subfault.mu, subfault.Mo()))

        # Create the topography, thereby calling Okada
        self.print_debug("SETUP: Creating topography using Okada model...")
        self.fault.create_dtopography(verbose=self.debug and self.debug_mode == 'full', active=True)

        # Mark output as dependent
        if separate_faults:
            for subfault in self.subfaults:
                adolc.dependent(subfault.dtopo.dZ)
        else:
            adolc.dependent(self.fault.dtopo.dZ_a)
        adolc.trace_off()

    def _create_topography_active_interpolate(self, tag=0, separate_faults=False):
        import adolc

        # Sanitise kwargs
        assert isinstance(tag, int)
        assert tag >= 0
        for control in self.active_controls:
            assert control in self.all_controls

        # Initialise tape
        adolc.trace_on(tag)

        # Read parameters and mark active variables as independent
        msg = "INIT: Subfault {:d}: shear modulus {:4.1e} Pa, seismic moment is {:4.1e}"
        for i, subfault in enumerate(self.subfaults):
            for control in self.all_controls:
                if control in self.active_controls:
                    subfault.__setattr__(control, adolc.adouble(self.control_parameters[control][i]))
                    adolc.independent(subfault.__getattribute__(control))
                else:
                    subfault.__setattr__(control, self.control_parameters[control][i])
            if self.debug and self.debug_mode == 'full':
                try:
                    print(msg.format(i, subfault.mu, subfault.Mo().val))
                except Exception:
                    print(msg.format(i, subfault.mu, subfault.Mo()))

        # Create the topography, thereby calling Okada
        self.print_debug("SETUP: Creating topography using Okada model...")
        self.fault.create_dtopography(verbose=self.debug and self.debug_mode == 'full', active=True)

        # Compute quantity of interest
        self.J_subfaults = [0.0 for j in range(self.N)]
        data = self._data_to_interpolate
        for j in range(self.N):
            for i in range(self.N):
                self.J_subfaults[j] += (data[i, j] - self.fault.dtopo.dZ_a[i, j])**2
            self.J_subfaults[j] /= self.N**2
        self.J = sum(self.J_subfaults)

        # Mark dependence
        if separate_faults:
            for j in range(self.N):
                adolc.dependent(self.J_subfaults[j])
        else:
            adolc.dependent(self.J)
        adolc.trace_off()

    @property
    def input_vector(self):
        """
        Get a vector of the same length as the total number of controls and populate it with passive
        versions of each parameter. This provides a point at which we can compute derivative
        matrices.
        """
        return np.array([
            self.control_parameters[control][i]
            for i in range(self.num_subfaults)
            for control in self.active_controls
        ])

    def get_seed_matrices(self):
        """
        Whilst the Okada function on each subfault is a nonlinear function of the associated
        controls, the total dislocation is just the sum over all subfaults. As such, the
        derivatives with respect to each parameter type (e.g. slip) may be computed simultaneously.
        All we need to do is choose an appropriate 'seed matrix' to propagate through the forward
        mode of AD.

        In the default case we have four active controls and hence there are four seed matrices.
        """
        n = len(self.active_controls)
        self._seed_matrices = np.array([
            [1 if i % n == j else 0 for j in range(n)]
            for i in range(len(self.input_vector))
        ])

    @property
    def seed_matrices(self):
        if not hasattr(self, '_seed_matrices'):
            self.get_seed_matrices()
        return self._seed_matrices

    # --- Regularisation

    def get_regularisation_term(self, prob):
        raise NotImplementedError  # TODO

    # --- Projection and interpolation into Okada basis

    def project(self, prob, source, maxiter=2, rtol=1.0e-02):
        """
        Project a source field into the Okada basis.

        Whilst the Okada model (evaluated on each subfault) is nonlinear, the dislocation field across
        the whole fault is computed by simply summing all contributions.

        We restrict attention to the case where slip and rake are the only active control parameters.
        On any given fault, if the slip is zero then so is the rake. This means that we cannot assemble
        a monolithic system using basis functions from both slip and rake parameter spaces, as half of
        its rows would be zero.

        Instead, we take an iterative approach:

          1. set all active control parameters to zero;
          2. while not converged:
              (a) solve for slip, holding rake parameters fixed;
              (b) solve for rake, holding slip parameters fixed;
              (c) compute dislocation field for current control parameters;
              (d) check for convergence.

        Convergence is determined either by subsequent dislocation field approximations meeting a
        relative l2 error tolerance, or when the maximum number of iterations is met.
        """
        active_controls = self.active_controls
        try:
            assert 'slip' in active_controls
            assert 'rake' in active_controls
            for control in self.all_controls:
                if control in active_controls:
                    assert control in ('slip', 'rake')
        except AssertionError:
            raise NotImplementedError

        # Cacheing
        cache_dir = create_directory(os.path.join(os.path.dirname(__file__), '.cache'))
        fname = os.path.join(cache_dir, 'mass_matrix_okada_{:d}_{:d}x{:d}_slip.npy')

        # Set rake to zero initially
        N = len(self.subfaults)
        self.control_parameters['rake'] = np.zeros(N)

        # Solve for slip and then rake
        dirty_cache = self.dirty_cache
        errors = []
        for n in range(maxiter):
            for control in ('slip', 'rake'):

                # Get basis functions with slip solution above
                self.active_controls = (control, )
                self.get_basis_functions(prob)
                phi = [bf.split()[1] for bf in self.basis_functions[control]]
                assert np.array([np.linalg.norm(phi_i.dat.data) for phi_i in phi]).min() > 0

                # Assemble mass matrix
                if os.path.isfile(fname) and not dirty_cache:
                    self.print_debug("PROJECTION: Loading slip mass matrix from cache...")
                    A = np.load(fname)
                    dirty_cache = True  # We only actually cache the first slip mass matrix
                else:
                    self.print_debug("PROJECTION: Assembling {:s} mass matrix...".format(control))
                    A = np.zeros((N, N))
                    for i in range(N):
                        for j in range(i+1):
                            A[i, j] = assemble(phi[i]*phi[j]*dx)
                    for i in range(N):
                        for j in range(i+1, N):
                            A[i, j] = A[j, i]
                    if control == 'slip' and n == 0:
                        self.print_debug("PROJECTION: Cacheing slip mass matrix...")
                        np.save(fname, A)

                # Assemble RHS
                self.print_debug("PROJECTION: Assembling RHS for {:s} solve...".format(control))
                b = np.array([assemble(phi[i]*source*dx) for i in range(N)])

                # Solve
                self.print_debug("PROJECTION: Solving linear system for {:s}...".format(control))
                self.control_parameters[control] = np.linalg.solve(A, b)

                # Ensure that we always end with a slip solve
                if control == 'rake':
                    continue

                # Generate surface
                surf = self.set_initial_condition(prob, subtract_from_bathymetry=False)
                if self.debug:
                    import matplotlib.pyplot as plt
                    fig, axes = plt.subplots(figsize=(8, 8))
                    fig.colorbar(tricontourf(surf, cmap='coolwarm', levels=50, axes=axes), ax=axes)
                    axes.set_title("Iteration {:d}".format(n))
                    axes.axis(False)
                    plt.show()

                # Check for convergence
                err = errornorm(surf, source)/norm(source)
                errors.append(err)
                msg = "PROJECTION: Relative l2 error at iteration {:d} = {:.2f}%"
                self.print_debug(msg.format(n, 100*err))
                if err < rtol:
                    msg = "PROJECTION: Converged due to meeting relative tol after {:d} iterations!"
                    self.print_debug(msg.format(n+1))
                    self.print_debug("PROJECTION: relative errors: {:}".format(errors))
                    break
                if n == maxiter-1:
                    msg = "PROJECTION: Terminated after maximum iteration count, {:d}!"
                    self.print_debug(msg.format(maxiter))
                    self.print_debug("PROJECTION: relative errors: {:}".format(errors))
                    break

        # Subtract initial surface from bathymetry
        self.subtract_surface_from_bathymetry(prob, surf=surf)

        # Reset active control tuple
        self.active_controls = active_controls

        return surf

    def interpolate(self, prob, source, tag=0):
        r"""
        Interpolate a source field into the radial basis using point evaluation.

        This involves solving an auxiliary optimisation problem! The objective functional is

      ..math::
            J(\mathbf m) = \tilde J(\mathbf m) \cdot \tilde J(\mathbf m),

        where :math:`m` is the control vector, :math:`\boldsymbol\phi` is the vector (radial) basis
        functions, :math:`f` is the source field we seek to represent and

      ..math::
            \tilde J(\mathbf m)_i = \frac1N\int_\Omega\sum_i(f^{okada}(\mathbf m)_i-f_i)\;\mathrm dx,

        where :math:`f^{okada}(\mathbf m)_i` is the result of the Okada model with control parameters
        :math:`\mathbf m`, evaluated at the centre of basis function :math:`i`. and :math:`N` is the
        number of radial basis functions.

        Solving this optimisation problem using a gradient-based approach means that we need to be
        able to differentiate the Okada model. We do this using the PyADOLC Python wrapper for the
        C++ operator overloading AD tool, ADOL-C.
        """
        # from adapt_utils.norms import vecnorm

        # self._data_to_interpolate = source  # TODO: Probably needs discretising on Okada grid
        self.create_topography(annotate=True, interpolate=True, tag=tag)
        # self.get_seed_matrices()
        raise NotImplementedError  # TODO: Copy over from notebook
