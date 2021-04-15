from thetis import *
from thetis.configuration import *

# import h5py
import weakref
import numpy as np
import matplotlib.pyplot as plt

from adapt_utils.options import CoupledOptions


__all__ = ["BoydOptions"]


plt.rc('text', usetex=True)
plt.rc('font', family='serif')


# TODO: Update
# TODO: Use paralellised version
class BoydOptions(CoupledOptions):
    """
    Parameters for test case in [Boyd et al. 1996].
    """
    soliton_amplitude = PositiveFloat(0.395).tag(config=True)

    def __init__(self, mesh=None, periodic=True, n=1, order=0, compute_metrics=True, **kwargs):
        """
        :kwarg approach: mesh adaptation approach
        :kwarg periodic: toggle periodic boundary in x-direction
        :kwarg n: mesh resolution
        """
        super(BoydOptions, self).__init__(mesh=mesh, **kwargs)
        self.solve_swe = True
        self.solve_tracer = False
        self.periodic = periodic  # TODO: needed?
        self.n = n
        self.order = order
        self.plot_pvd = True

        # Physics
        self.g.assign(1.0)
        self.base_viscosity = 0.0
        self.modon_propagation_speed = -1.0/3.0
        if self.order != 0:
            self.modon_propagation_speed -= 0.395*self.soliton_amplitude**2
        else:
            raise ValueError("Only zeroth and first order asymptotic expansions supported.")

        # Initial mesh
        self.lx = 48
        self.ly = 24
        self.distribution_parameters = {
            'partition': True,
            'overlap_type': (DistributedMeshOverlapType.VERTEX, 10),
        }
        if mesh is None:
            args = (self.lx*n, self.ly*n, self.lx, self.ly)
            kwargs = {'distribution_parameters': self.distribution_parameters}
            if periodic:
                kwargs['direction'] = 'x'
            mesh_constructor = PeriodicRectangleMesh if periodic else RectangleMesh
            self.default_mesh = mesh_constructor(*args, **kwargs)
            x, y = SpatialCoordinate(self.default_mesh)
            self.default_mesh.coordinates.interpolate(as_vector([x - self.lx/2, y - self.ly/2]))
        # NOTE: This setup corresponds to 'Grid B' in [Huang et al 2008].

        # Time integration
        self.dt = 0.05
        self.end_time = 120.0
        # self.end_time = 30.0
        self.dt_per_export = 50
        self.timestepper = 'CrankNicolson'
        self.family = 'dg-dg'
        # self.family = 'dg-cg'

        # Adaptivity
        self.h_min = 1e-8
        self.h_max = 10.0

        # Read in user-specified kwargs
        self.update(kwargs)  # TODO: Redundant?

        # Plotting
        self.relative_errors = {'l1': [], 'l2': [], 'linf': []}
        self.exact_solution_file = File(os.path.join(self.di, 'exact.pvd'))

    def asymptotic_expansion_uv(self, U_2d, time=0.0):
        x, y = SpatialCoordinate(U_2d.mesh())

        t = Constant(time)
        B = Constant(self.soliton_amplitude)
        c = Constant(self.modon_propagation_speed)
        xi = x - c*t
        psi = exp(-0.5*y*y)
        phi = 0.771*(B/cosh(B*xi))**2
        dphidx = -2*B*phi*tanh(B*xi)
        C = -0.395*B*B

        # Zeroth order terms
        u_terms = phi*0.25*(-9 + 6*y*y)*psi
        v_terms = 2*y*dphidx*psi
        if self.order == 0:
            return interpolate(as_vector([u_terms, v_terms]), U_2d)

        # Unnormalised Hermite series coefficients for u
        u = np.zeros(28)
        u[0] = 1.7892760e+00
        u[2] = 0.1164146e+00
        u[4] = -0.3266961e-03
        u[6] = -0.1274022e-02
        u[8] = 0.4762876e-04
        u[10] = -0.1120652e-05
        u[12] = 0.1996333e-07
        u[14] = -0.2891698e-09
        u[16] = 0.3543594e-11
        u[18] = -0.3770130e-13
        u[20] = 0.3547600e-15
        u[22] = -0.2994113e-17
        u[24] = 0.2291658e-19
        u[26] = -0.1178252e-21

        # Unnormalised Hermite series coefficients for v
        v = np.zeros(28)
        v[3] = -0.6697824e-01
        v[5] = -0.2266569e-02
        v[7] = 0.9228703e-04
        v[9] = -0.1954691e-05
        v[11] = 0.2925271e-07
        v[13] = -0.3332983e-09
        v[15] = 0.2916586e-11
        v[17] = -0.1824357e-13
        v[19] = 0.4920951e-16
        v[21] = 0.6302640e-18
        v[23] = -0.1289167e-19
        v[25] = 0.1471189e-21

        # Hermite polynomials
        polynomials = [Constant(1.0), 2*y]
        for i in range(2, 28):
            polynomials.append(2*y*polynomials[i-1] - 2*(i-1)*polynomials[i-2])

        # First order terms
        u_terms += C*phi*0.5625*(3 + 2*y*y)*psi
        u_terms += phi*phi*psi*sum(u[i]*polynomials[i] for i in range(28))
        v_terms += dphidx*phi*psi*sum(v[i]*polynomials[i] for i in range(28))

        return interpolate(as_vector([u_terms, v_terms]), U_2d)  # TODO: account for periodicity

    def asymptotic_expansion_eta(self, H_2d, time=0.0):
        x, y = SpatialCoordinate(H_2d.mesh())

        # Variables for asymptotic expansion
        t = Constant(time)
        B = Constant(self.soliton_amplitude)
        c = Constant(self.modon_propagation_speed)
        xi = x - c*t
        psi = exp(-0.5*y*y)
        phi = 0.771*(B/cosh(B*xi))**2
        C = -0.395*B*B

        # Zeroth order terms
        eta_terms = phi*0.25*(3 + 6*y*y)*psi
        if self.order == 0:
            return interpolate(eta_terms, H_2d)

        # Unnormalised Hermite series coefficients
        eta = np.zeros(28)
        eta[0] = -3.0714300e+00
        eta[2] = -0.3508384e-01
        eta[4] = -0.1861060e-01
        eta[6] = -0.2496364e-03
        eta[8] = 0.1639537e-04
        eta[10] = -0.4410177e-06
        eta[12] = 0.8354759e-09
        eta[14] = -0.1254222e-09
        eta[16] = 0.1573519e-11
        eta[18] = -0.1702300e-13
        eta[20] = 0.1621976e-15
        eta[22] = -0.1382304e-17
        eta[24] = 0.1066277e-19
        eta[26] = -0.1178252e-21

        # Hermite polynomials
        polynomials = [Constant(1.0), 2*y]
        for i in range(2, 28):
            polynomials.append(2*y*polynomials[i-1] - 2*(i-1)*polynomials[i-2])

        # First order terms
        eta_terms += C*phi*0.5625*(-5 + 2*y*y)*psi
        eta_terms += phi*phi*psi*sum(eta[i]*polynomials[i] for i in range(28))

        return interpolate(eta_terms, H_2d)  # TODO: account for periodicity

    def set_bathymetry(self, fs):
        return Constant(1.0)

    def set_coriolis(self, fs):
        """
        Set beta plane approximation Coriolis parameter.

        :arg fs: `FunctionSpace` in which the solution should live.
        """
        return interpolate(SpatialCoordinate(fs.mesh())[1], fs)

    def set_boundary_conditions(self, prob, i):
        """
        Set no slip boundary conditions uv = 0 along North and South boundaries.
        """
        dirichlet = {'uv': Constant(as_vector([0., 0.]))}
        boundary_conditions = {'shallow_water': {}}
        for tag in prob.meshes[i].exterior_facets.unique_markers:
            boundary_conditions['shallow_water'][tag] = dirichlet
        return boundary_conditions

    def get_exact_solution(self, prob, i, t=0.0):
        """
        Evaluate asymptotic solution of chosen order.

        :arg prob: :class:`AdaptiveProblem` object
        :arg i: index of mesh on `prob`
        :kwarg t: current time.
        """
        msg = "Computing order {:d} asymptotic solution at time {:.2f}s on mesh with {:d} local elements..."
        self.print_debug(msg.format(self.order, t, prob.meshes[i].num_cells()))
        q = Function(prob.V[i], name="Order {:d} asymptotic solution".format(self.order))
        u, eta = q.split()
        u.assign(self.asymptotic_expansion_uv(prob.V[i].sub(0), t))
        eta.assign(self.asymptotic_expansion_eta(prob.V[i].sub(1), t))
        u.rename('Asymptotic velocity')
        eta.rename('Asymptotic elevation')
        return q

    def set_initial_condition(self, prob):
        """Set initial elevation and velocity using asymptotic solution."""
        prob.fwd_solutions[0].assign(self.get_exact_solution(prob, 0, t=0.0))

    def get_reference_mesh(self, n=50):
        """Set up a non-periodic, very fine mesh on the PDE domain."""
        nx, ny = self.lx*n, self.ly*n
        self.print_debug("Generating reference mesh with {:d} local elements...".format(2*nx*ny))
        reference_mesh = RectangleMesh(nx, ny, self.lx, self.ly,
                                       distribution_parameters=self.distribution_parameters)
        x, y = SpatialCoordinate(reference_mesh)
        reference_mesh.coordinates.interpolate(as_vector([x - self.lx/2, y - self.ly/2]))
        return reference_mesh

    def remove_periodicity(self, sol):
        """Project a field `sol` from a periodic space into an equivalent non-periodic space."""

        # Generate an identical non-periodic mesh
        nx = int(self.lx*n)
        ny = int(self.ly*n)
        nonperiodic_mesh = RectangleMesh(nx, ny, self.lx, self.ly,
                                         distribution_parameters=self.distribution_parameters)
        x, y = SpatialCoordinate(nonperiodic_mesh)
        nonperiodic_mesh.coordinates.interpolate(as_vector([x - self.lx/2, y - self.ly/2]))

        # Mark meshes as compatible
        nonperiodic_mesh._parallel_compatible = {weakref.ref(sol.function_space().mesh())}

        # Project into corresponding function space
        V = FunctionSpace(nonperiodic_mesh, sol.ufl_element())
        return project(sol, V)

    def get_peaks(self, sol_periodic, reference_mesh_resolution=50):
        """
        Given a numerical solution of the test case, compute the metrics as given in
        [Huang et al 2008]:
          * h± : relative peak height for Northern / Southern soliton
          * C± : relative mean phase speed for Northern / Southern soliton
          * RMS: root mean square error
        The solution is projected onto a finer space in order to get a better approximation.

        :arg sol_periodic: Numerical solution of PDE.
        """
        self.print_debug("Generating non-periodic counterpart of periodic function...")
        sol = self.remove_periodicity(sol_periodic)  # TODO: Needed?

        # Project solution into a reference space on a fine mesh
        reference_mesh = self.get_reference_mesh(n=reference_mesh_resolution)
        self.print_debug("Projecting solution onto fine mesh...")
        P1_ref = FunctionSpace(reference_mesh, "CG", 1)
        reference_mesh._parallel_compatible = {weakref.ref(mesh)}  # Mark meshes as compatible
        sol_ref = project(sol, P1_ref)
        xcoords = project(reference_mesh.coordinates[0], P1_ref)

        # Calculate RMS error
        self.print_debug("Calculating root mean square error...")
        sol_diff = sol_ref.vector().gather()
        sol_diff -= sol.vector().gather()
        sol_diff *= sol_diff
        rms = sqrt(sol_diff.sum()/sol_diff.size)

        # Get relative mean peak heights
        self.print_debug("Extracting relative mean peak height...")
        sol_ref.interpolate(sign(y)*sol_ref)  # Flip sign in sourthern hemisphere
        with sol_ref.dat.vec_ro as v:
            i_n, h_n = v.max()
            i_s, h_s = v.min()

            # Find ranks which own peaks
            ownership_range = v.getOwnershipRanges()
            for j in range(sol.function_space().mesh().comm.size):
                if i_n >= ownership_range[j] and i_n < ownership_range[j+1]:
                    rank_with_n_peak = j
                if i_s >= ownership_range[j] and i_s < ownership_range[j+1]:
                    rank_with_s_peak = j

        # Get mean phase speeds
        x_n, x_s = None, None
        with xcoords.dat.vec_ro as xdat:
            if mesh2d.comm.rank == rank_with_n_peak:
                x_n = xdat[i_n]
            if mesh2d.comm.rank == rank_with_s_peak:
                x_s = xdat[i_s]
        x_n = mesh2d.comm.bcast(x_n, root=rank_with_n_peak)
        x_s = mesh2d.comm.bcast(x_s, root=rank_with_s_peak)

        # Get relative versions of metrics using high resolution FVCOM data
        h_n /= 0.1567020
        h_s /= -0.1567020  # Flip sign back
        c_n = (48.0 - x_n)/47.18
        c_s = (48.0 - x_s)/47.18
        return h_n, h_s, c_n, c_s, rms

    # def get_export_func(self, prob, i):
    #     def export_func():
    #         if self.debug:
    #             t = prob.simulation_time

    #             # Get exact solution and plot
    #             exact = self.get_exact_solution(prob, i, t=t)
    #             exact_u, exact_eta = exact.split()
    #             self.exact_solution_file.write(exact_u, exact_eta)

    #             # Compute relative error
    #             approx = prob.fwd_solutions[i]
    #             u, eta = approx.split()
    #             l1_error = errornorm(approx, exact, norm_type='l1')/norm(exact, norm_type='l1')
    #             l2_error = errornorm(approx, exact, norm_type='l2')/norm(exact, norm_type='l2')
    #             linf_error = abs(1.0 - eta.vector().gather().max()/exact_eta.vector().gather().max())
    #             self.relative_errors['l1'].append(l1_error)
    #             self.relative_errors['l2'].append(l2_error)
    #             self.relative_errors['linf'].append(linf_error)
    #             msg = "DEBUG: t = {:6.2f} l1 error {:6.2f}% l2 error {:6.2f}% linf error {:6.2f}%"
    #             print_output(msg.format(t, 100*l1_error, 100*l2_error, 100*linf_error))

    #             # TODO: Compute exact solution and error on fine reference mesh
    #         return
    #     return export_func

    # def write_to_hdf5(self, filename=None):
    #     """Write relative error timeseries to HDF5."""
    #     try:
    #         assert len(self.relative_errors['l2']) > 0
    #     except AssertionError:
    #         raise ValueError("Nothing to write to HDF5!")
    #     fname = 'relative_errors'
    #     if filename is not None:
    #         fname = '_'.join([fname, filename])
    #     fname = os.path.join(self.di, fname) + '.hdf5'
    #     errorfile = h5py.File(fname, 'w')
    #     errorfile.create_dataset('l1_error', data=self.relative_errors['l1'])
    #     errorfile.create_dataset('l2_error', data=self.relative_errors['l2'])
    #     errorfile.create_dataset('linf_error', data=self.relative_errors['linf'])
    #     errorfile.close()

    # def read_from_hdf5(self, filename=None):
    #     """Read relative error timeseries from HDF5."""
    #     fname = os.path.join(self.di, filename or 'relative_errors.hdf5')
    #     try:
    #         assert os.path.exists(fname)
    #     except AssertionError:
    #         raise IOError("HDF file {:s} does not exist!".format(fname))
    #     errorfile = h5py.File(fname, 'r')
    #     self.relative_errors['l1'] = np.array(errorfile['l1_error'])
    #     self.relative_errors['l2'] = np.array(errorfile['l2_error'])
    #     self.relative_errors['linf'] = np.array(errorfile['linf_error'])
    #     errorfile.close()

    # def plot_errors(self):
    #     """Plot relative error timeseries."""

    #     for norm_type in ("l1", "l2", "linf"):
    #         plt.figure()
    #         fnames = [f for f in os.listdir(self.di) if f.endswith('.hdf5') and 'relative_errors' in f]
    #         fnames.sort()
    #         for fname in fnames:
    #             self.read_from_hdf5(filename=fname)
    #             try:
    #                 n = len(self.relative_errors[norm_type])
    #             except KeyError:  # TODO: temp
    #                 break
    #             try:
    #                 assert n > 0
    #             except AssertionError:
    #                 raise ValueError("Nothing to plot!")
    #             self.relative_errors[norm_type] = 100*np.array(self.relative_errors[norm_type])
    #             label = fname.split('/')[-1][:-5]
    #             words = label.split('_')
    #             kwargs = {
    #                 'label': ' '.join(words[2:]).capitalize(),
    #                 # 'linestyle': 'dashed',
    #                 'marker': 'x',
    #             }
    #             plt.plot(np.linspace(0, self.end_time, n), self.relative_errors[norm_type], **kwargs)
    #         plt.xlabel(r"Time [s]")
    #         if norm_type == "l1":
    #             plt.ylabel(r"Relative $\mathcal L_1$ error (\%)")
    #         elif norm_type == "l2":
    #             plt.ylabel(r"Relative $\mathcal L_2$ error (\%)")
    #         elif norm_type == "linf":
    #             plt.ylabel(r"Relative $\mathcal L_\infty$ error (\%)")
    #         plt.legend()
    #         plt.savefig(os.path.join(self.di, 'relative_errors_{:s}.png'.format(norm_type)))
