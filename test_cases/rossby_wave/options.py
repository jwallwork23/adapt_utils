from thetis import *
from thetis.configuration import *

import h5py
import weakref
import numpy as np
import matplotlib.pyplot as plt

from adapt_utils.swe.options import ShallowWaterOptions


__all__ = ["BoydOptions"]


plt.rc('text', usetex=True)
plt.rc('font', family='serif')


class BoydOptions(ShallowWaterOptions):
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
        self.periodic = periodic  # TODO: needed?
        self.n = n
        self.order = order
        self.plot_pvd = True

        # Physics
        self.g.assign(1.0)

        # Initial mesh
        self.lx = 48
        self.ly = 24
        self.distribution_parameters = {
            'partition': True,
            'overlap_type': (DistributedMeshOverlapType.VERTEX, 10),
        }
        if mesh is None:
            args = (self.lx*n, self.ly*n, self.lx, self.ly)
            kwargs = {'distribution_parameters' : self.distribution_parameters}
            if periodic:
                kwargs['direction'] = 'x'
            mesh_constructor = PeriodicRectangleMesh if periodic else RectangleMesh
            self.default_mesh = mesh_constructor(*args, **kwargs)
            x, y = SpatialCoordinate(self.default_mesh)
            self.default_mesh.coordinates.interpolate(as_vector([x - self.lx/2, y - self.ly/2]))
        # NOTE: This setup corresponds to 'Grid B' in [Huang et al 2008].

        # Time integration
        self.dt = 0.05
        # self.end_time = 120.0
        self.end_time = 30.0  # TODO: Temporary until periodicity is fixed in exact solution
        self.dt_per_export = 50
        self.dt_per_remesh = 50
        self.timestepper = 'CrankNicolson'
        self.family = 'dg-dg'
        # self.family = 'dg-cg'

        # Adaptivity
        self.h_min = 1e-8
        self.h_max = 10.0

        # Read in user-specified kwargs
        self.update(kwargs)

        # Unnormalised Hermite series coefficients
        u = np.zeros(28)
        v = np.zeros(28)
        eta = np.zeros(28)
        u[0]    =  1.7892760e+00
        u[2]    =  0.1164146e+00
        u[4]    = -0.3266961e-03
        u[6]    = -0.1274022e-02
        u[8]    =  0.4762876e-04
        u[10]   = -0.1120652e-05
        u[12]   =  0.1996333e-07
        u[14]   = -0.2891698e-09
        u[16]   =  0.3543594e-11
        u[18]   = -0.3770130e-13
        u[20]   =  0.3547600e-15
        u[22]   = -0.2994113e-17
        u[24]   =  0.2291658e-19
        u[26]   = -0.1178252e-21
        v[3]    = -0.6697824e-01
        v[5]    = -0.2266569e-02
        v[7]    =  0.9228703e-04
        v[9]    = -0.1954691e-05
        v[11]   =  0.2925271e-07
        v[13]   = -0.3332983e-09
        v[15]   =  0.2916586e-11
        v[17]   = -0.1824357e-13
        v[19]   =  0.4920951e-16
        v[21]   =  0.6302640e-18
        v[23]   = -0.1289167e-19
        v[25]   =  0.1471189e-21
        eta[0]  = -3.0714300e+00
        eta[2]  = -0.3508384e-01
        eta[4]  = -0.1861060e-01
        eta[6]  = -0.2496364e-03
        eta[8]  =  0.1639537e-04
        eta[10] = -0.4410177e-06
        eta[12] =  0.8354759e-09
        eta[14] = -0.1254222e-09
        eta[16] =  0.1573519e-11
        eta[18] = -0.1702300e-13
        eta[20] =  0.1621976e-15
        eta[22] = -0.1382304e-17
        eta[24] =  0.1066277e-19
        eta[26] = -0.1178252e-21
        hermite_coeffs = {'u': u, 'v': v, 'eta': eta}

        # Hermite polynomials
        x, y = SpatialCoordinate(self.default_mesh)
        polynomials = [Constant(1.0), 2*y]
        for i in range(2, 28):
            polynomials.append(2*y*polynomials[i-1] - 2*(i-1)*polynomials[i-2])
        self.Ψ = exp(-0.5*y*y)
        self.hermite_sum = {}
        for f in ('u', 'v', 'eta'):
            self.hermite_sum[f] = self.Ψ*sum(hermite_coeffs[f][i]*polynomials[i] for i in range(28))

        # Variables for asymptotic expansion
        self.t = Constant(0.0)
        B = self.soliton_amplitude
        modon_propagation_speed = -1.0/3.0
        if self.order == 1:
            modon_propagation_speed -= 0.395*B*B
        c = Constant(modon_propagation_speed)
        ξ = x - c*self.t
        self.φ = 0.771*(B/cosh(B*ξ))**2
        self.dφdx = -2*B*self.φ*tanh(B*ξ)
        # TODO: account for periodicity

        # Plotting
        self.relative_errors = {'l1': [], 'l2': [], 'linf': []}
        self.exact_solution_file = File(os.path.join(self.di, 'exact.pvd'))

    def set_bathymetry(self, fs):
        self.bathymetry = Constant(1.0, domain=fs.mesh())
        return self.bathymetry

    def set_viscosity(self, fs):
        self.viscosity = Constant(0.0, domain=fs.mesh())
        return self.viscosity

    def set_coriolis(self, fs):
        """
        Set beta plane approximation Coriolis parameter.

        :arg fs: `FunctionSpace` in which the solution should live.
        """
        self.coriolis = interpolate(SpatialCoordinate(fs.mesh())[1], fs)
        return self.coriolis

    def set_boundary_conditions(self, fs):
        """
        Set no slip boundary conditions uv = 0 along North and South boundaries.
        """
        dirichlet = {'uv': Constant(as_vector([0., 0.]), domain=fs.mesh())}
        boundary_conditions = {}
        for tag in fs.mesh().exterior_facets.unique_markers:
            boundary_conditions[tag] = dirichlet
        return boundary_conditions

    def set_qoi_kernel(self, fs):
        # raise NotImplementedError  # TODO: Kelvin wave?
        return

    def get_exact_solution(self, fs, t=0.0):
        """
        Evaluate asymptotic solution of chosen order.

        :arg fs: `FunctionSpace` in which the solution should live.
        :kwarg t: current time.
        """
        msg = "Computing order {:d} asymptotic solution at time {:.2f}s on mesh with {:d} local elements..."
        self.print_debug(msg.format(self.order, t, fs.mesh().num_cells()))
        self.t.assign(t)
        x, y = SpatialCoordinate(self.default_mesh)
        B = self.soliton_amplitude
        C = -0.395*B*B

        # Zero order terms
        self.terms = {}
        self.terms['u'] = self.φ*0.25*(-9 + 6*y*y)*self.Ψ
        self.terms['v'] = 2*y*self.dφdx*self.Ψ
        self.terms['eta'] = self.φ*0.25*(3 + 6*y*y)*self.Ψ

        # First order terms
        if self.order > 0:
            assert self.order == 1
            self.terms['u'] += C*self.φ*0.5625*(3 + 2*y*y)*self.Ψ
            self.terms['u'] += self.φ*self.φ*self.hermite_sum['u']
            self.terms['v'] += self.dφdx*self.φ*self.hermite_sum['v']
            self.terms['eta'] += C*self.φ*0.5625*(-5 + 2*y*y)*self.Ψ
            self.terms['eta'] += self.φ*self.φ*self.hermite_sum['eta']

        self.exact_solution = Function(fs, name="Order {:d} asymptotic solution".format(self.order))
        u, eta = self.exact_solution.split()
        u.interpolate(as_vector([self.terms['u'], self.terms['v']]))
        eta.interpolate(self.terms['eta'])
        u.rename('Asymptotic velocity')
        eta.rename('Asymptotic elevation')
        return self.exact_solution

    def set_initial_condition(self, fs):
        """
        Set initial elevation and velocity using asymptotic solution.

        :arg fs: `FunctionSpace` in which the initial condition should live.
        """
        self.get_exact_solution(fs, t=0.0)
        self.initial_value = self.exact_solution.copy(deepcopy=True)
        return self.initial_value

    def get_reference_mesh(self, n=50):
        """
        Set up a non-periodic, very fine mesh on the PDE domain.
        """
        nx = int(self.lx*n)
        ny = int(self.ly*n)
        self.print_debug("Generating reference mesh with {:d} local elements...".format(2*nx*ny))
        reference_mesh = RectangleMesh(nx, ny, self.lx, self.ly,
                                       distribution_parameters=self.distribution_parameters)
        x, y = SpatialCoordinate(reference_mesh)
        reference_mesh.coordinates.interpolate(as_vector([x - self.lx/2, y - self.ly/2]))
        return reference_mesh

    def remove_periodicity(self, sol):
        """
        :arg sol: Function to remove periodicity of.
        """

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
        Given a numerical solution of the test case, compute the metrics as given in [Huang et al 2008]:
          * h± : relative peak height for Northern / Southern soliton
          * C± : relative mean phase speed for Northern / Southern soliton
          * RMS: root mean square error
        The solution is projected onto a finer space in order to get a better approximation.

        :arg sol_periodic: Numerical solution of PDE.
        """
        self.print_debug("Generating non-periodic counterpart of periodic function...")
        sol = self.remove_periodicity(sol_periodic)

        # Split Northern and Southern halves of domain
        self.print_debug("Splitting Northern and Southern halves of domain...")
        mesh = sol.function_space().mesh()
        y = SpatialCoordinate(mesh)[1]
        upper = Function(sol.function_space())
        upper.interpolate(0.5*(sign(y)+1))
        lower = Function(sol.function_space())
        lower.interpolate(0.5*(-sign(y)+1))
        sol_upper = Function(sol.function_space()).assign(sol)
        sol_lower = Function(sol.function_space()).assign(sol)
        sol_upper *= upper
        sol_lower *= lower

        # Project solution into a reference space on a fine mesh
        reference_mesh = self.get_reference_mesh(n=reference_mesh_resolution)
        self.print_debug("Projecting solution onto fine mesh...")
        fs = FunctionSpace(reference_mesh, sol.ufl_element())
        reference_mesh._parallel_compatible = {weakref.ref(mesh)}  # Mark meshes as compatible
        sol_proj = Function(fs)
        sol_proj.project(sol)
        sol_upper_proj = Function(fs)
        sol_upper_proj.project(sol_upper)
        sol_lower_proj = Function(fs)
        sol_lower_proj.project(sol_lower)
        xcoords = Function(sol_proj.function_space())
        xcoords.interpolate(reference_mesh.coordinates[0])

        # Get relative mean peak height
        self.print_debug("Extracting relative mean peak height...")
        with sol_upper_proj.dat.vec_ro as vu:
            i_upper, self.h_upper = vu.max()
        with sol_lower_proj.dat.vec_ro as vl:
            i_lower, self.h_lower = vl.max()
        self.h_upper /= 0.1567020
        self.h_lower /= 0.1567020

        # Get relative mean phase speed
        if self.debug:
            print_output("Extracting relative mean phase speed...")
        xdat = xcoords.vector().gather()
        self.c_upper = (48 - xdat[i_upper])/47.18
        self.c_lower = (48 - xdat[i_lower])/47.18

        # Calculate RMS error
        self.print_debug("Calculating root mean square error...")
        init = self.remove_periodicity(self.initial_value.split()[1])
        init_proj = Function(fs)
        init_proj.project(init)
        sol_proj -= init_proj
        sol_proj *= sol_proj
        self.rms = sqrt(np.mean(sol_proj.vector().gather()))
        self.print_debug("Done!")

    def get_export_func(self, solver_obj):  # TODO: Could just write as a callback
        def export_func():
            if self.debug:
                t = solver_obj.simulation_time

                # Get exact solution and plot
                exact = self.get_exact_solution(solver_obj.function_spaces.V_2d, t=t)
                exact_u, exact_eta = exact.split()
                self.exact_solution_file.write(exact_u, exact_eta)

                # Compute relative error
                approx = solver_obj.fields.solution_2d
                u, eta = approx.split()
                l1_error = errornorm(approx, exact, norm_type='l1')/norm(exact, norm_type='l1')
                l2_error = errornorm(approx, exact, norm_type='l2')/norm(exact, norm_type='l2')
                linf_error = abs(1.0 - eta.vector().gather().max()/exact_eta.vector().gather().max())
                self.relative_errors['l1'].append(l1_error)
                self.relative_errors['l2'].append(l2_error)
                self.relative_errors['linf'].append(linf_error)
                msg = "DEBUG: t = {:6.2f} l1 error {:6.2f}% l2 error {:6.2f}% linf error {:6.2f}%"
                print_output(msg.format(t, 100*l1_error, 100*l2_error, 100*linf_error))

                # TODO: Compute exact solution and error on fine reference mesh
            return
        return export_func

    # def get_update_forcings(self, solver_obj):
    #     def update_forcings(t):
    #         return
    #     return update_forcings

    def write_to_hdf5(self, filename=None):
        """Write relative error timeseries to HDF5."""
        try:
            assert len(self.relative_errors['l2']) > 0
        except AssertionError:
            raise ValueError("Nothing to write to HDF5!")
        fname = 'relative_errors'
        if filename is not None:
            fname = '_'.join([fname, filename])
        fname = os.path.join(self.di, fname) + '.hdf5'
        errorfile = h5py.File(fname, 'w')
        errorfile.create_dataset('l1_error', data=self.relative_errors['l1'])
        errorfile.create_dataset('l2_error', data=self.relative_errors['l2'])
        errorfile.create_dataset('linf_error', data=self.relative_errors['linf'])
        errorfile.close()

    def read_from_hdf5(self, filename=None):
        """Read relative error timeseries from HDF5."""
        fname = os.path.join(self.di, filename or 'relative_errors.hdf5')
        try:
            assert os.path.exists(fname)
        except AssertionError:
            raise IOError("HDF file {:s} does not exist!".format(fname))
        errorfile = h5py.File(fname, 'r')
        self.relative_errors['l1'] = np.array(errorfile['l1_error'])
        self.relative_errors['l2'] = np.array(errorfile['l2_error'])
        self.relative_errors['linf'] = np.array(errorfile['linf_error'])
        errorfile.close()

    def plot_errors(self):
        """Plot relative error timeseries."""

        for norm_type in ("l1", "l2", "linf"):
            plt.figure()
            fnames = [f for f in os.listdir(self.di) if f.endswith('.hdf5') and 'relative_errors' in f]
            fnames.sort()
            for fname in fnames:
                self.read_from_hdf5(filename=fname)
                try:
                    n = len(self.relative_errors[norm_type])
                except KeyError:  # TODO: temp
                    break
                try:
                    assert n > 0
                except AssertionError:
                    raise ValueError("Nothing to plot!")
                self.relative_errors[norm_type] = 100*np.array(self.relative_errors[norm_type])
                label = fname.split('/')[-1][:-5]
                words = label.split('_')
                kwargs = {
                    'label': ' '.join(words[2:]).capitalize(),
                    # 'linestyle': 'dashed',
                    'marker': 'x',
                }
                plt.plot(np.linspace(0, self.end_time, n), self.relative_errors[norm_type], **kwargs)
            plt.xlabel(r"Time [s]")
            if norm_type == "l1":
                plt.ylabel(r"Relative $\mathcal L_1$ error (\%)")
            elif norm_type == "l2":
                plt.ylabel(r"Relative $\mathcal L_2$ error (\%)")
            elif norm_type == "linf":
                plt.ylabel(r"Relative $\mathcal L_\infty$ error (\%)")
            plt.legend()
            plt.savefig(os.path.join(self.di, 'relative_errors_{:s}.png'.format(norm_type)))
