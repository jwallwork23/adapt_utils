from thetis import *
from thetis.configuration import *

from adapt_utils.options import Options

import numpy as np


__all__ = ["BoydOptions"]


# TODO: ShallowWaterOptions superclass to take away some of the Thetis parameters
# TODO: More test cases to consider


class BoydOptions(Options):
    """
    Parameters for test case in [Boyd et al. 1996].
    """
    bathymetry = FiredrakeScalarExpression(Constant(1.)).tag(config=True)
    viscosity = FiredrakeScalarExpression(Constant(0.)).tag(config=True)
    drag_coefficient = FiredrakeScalarExpression(Constant(0.)).tag(config=True)
    soliton_amplitude = PositiveFloat(0.395).tag(config=True)
    g = PositiveFloat(1.).tag(config=True)

    def __init__(self, approach='fixed_mesh', periodic=True, n=1, order=0, compute_metrics=True):
        """
        :kwarg approach: mesh adaptation approach
        :kwarg periodic: toggle periodic boundary in x-direction
        :kwarg n: mesh resolution
        """
        super(BoydOptions, self).__init__(approach)
        self.approach = approach
        self.periodic = periodic
        self.n = n
        self.order = order

        # Initial mesh
        self.lx = 48
        self.ly = 24
        if periodic:
            self.default_mesh = PeriodicRectangleMesh(self.lx*n, self.ly*n, self.lx, self.ly, direction='x')
        else:
            self.default_mesh = RectangleMesh(self.lx*n, self.ly*n, self.lx, self.ly)
        x, y = SpatialCoordinate(self.default_mesh)
        self.default_mesh.coordinates.interpolate(as_vector([x - self.lx/2, y - self.ly/2]))
        self.x, self.y = SpatialCoordinate(self.default_mesh)
        # NOTE: This setup corresponds to 'Grid B' in [Huang et al 2008].

        # Solver
        self.family = 'dg-dg'
        self.symmetric_viscosity = False

        # Time integration
        self.dt = 0.05
        if compute_metrics:
            self.end_time = 120.
        else:
            self.start_time = 10.
            self.end_time = 20.
        self.dt_per_export = 10
        self.dt_per_remesh = 20
        self.timestepper = 'CrankNicolson'

        # Adaptivity
        self.h_min = 1e-8
        self.h_max = 10.

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
        self.hermite_coeffs = {'u': u, 'v': v, 'eta': eta}

    def set_bcs(self):
        """
        Set no slip boundary conditions uv = 0 along North and South boundaries.
        """
        self.boundary_conditions[1] = {'uv': Constant(as_vector([0., 0.]))}
        self.boundary_conditions[2] = {'uv': Constant(as_vector([0., 0.]))}
        if not self.periodic:
            self.boundary_conditions[3] = {'uv': Constant(as_vector([0., 0.]))}
            self.boundary_conditions[4] = {'uv': Constant(as_vector([0., 0.]))}
        return self.boundary_conditions

    def polynomials(self):
        """
        Get Hermite polynomials.
        """
        polys = [Constant(1.), 2*self.y]
        for i in range(2, 28):
            polys.append(2*self.y*polys[i-1] - 2*(i-1)*polys[i-2])
        return polys

    def xi(self, t=0.):
        """
        Get time-shifted x-coordinate.

        :kwarg t: current time.
        """
        c = -1/3  # Modon propagation speed
        if self.order == 1:
            c -= 0.395*self.soliton_amplitude*self.soliton_amplitude
        return self.x - c*t

    def phi(self, t=0.):
        """
        sech^2 term.

        :kwarg t: current time.
        """
        B = self.soliton_amplitude
        return 0.771*B*B*(1/(cosh(B*self.xi(t))**2))

    def dphidx(self, t=0.):
        """
        tanh * phi term.

        :kwarg t: current time. 
        """
        B = self.soliton_amplitude
        return -2*B*self.phi(t)*tanh(B*self.xi(t))

    def psi(self):
        """
        exp term.
        """
        return exp(-0.5*self.y*self.y)

    def zeroth_order_terms(self, t=0.):
        """
        Zeroth order asymptotic solution.

        :kwarg t: current time.
        """
        self.terms = {}
        self.terms['u'] = self.phi(t)*0.25*(-9 + 6*self.y*self.y)*self.psi()
        self.terms['v'] = 2*self.y*self.dphidx(t)*self.psi()
        self.terms['eta'] = self.phi(t)*0.25*(3 + 6*self.y*self.y)*self.psi()

    def hermite_sum(self, field):
        """
        Sum the terms of the Hermite expansion for a given field.
        """
        h = sum(self.hermite_coeffs[field][i]*self.polynomials()[i] for i in range(28))
        return self.psi()*h

    def first_order_terms(self, t=0.):
        """
        First order asymptotic solution.

        :kwarg t: current time.
        """
        C = -0.395*self.soliton_amplitude*self.soliton_amplitude
        phi = self.phi(t)
        self.zeroth_order_terms(t)

        # Expansion for u
        self.terms['u'] += C*phi*0.5625*(3 + 2*self.y*self.y)*self.psi()
        self.terms['u'] += phi*phi*self.hermite_sum('u')

        # Expansion for v
        self.terms['v'] += self.dphidx(t)*phi*self.hermite_sum('v')

        # Expansion for eta
        self.terms['eta'] += C*phi*0.5625*(-5 + 2*self.y*self.y)*self.psi()
        self.terms['eta'] += phi*phi*self.hermite_sum('eta')

    def set_coriolis(self, fs):
        """
        Set beta plane approximation Coriolis parameter.

        :arg fs: `FunctionSpace` in which the solution should live.
        """
        x, y = SpatialCoordinate(fs.mesh())
        self.coriolis = Function(fs)
        self.coriolis.interpolate(y)
        return self.coriolis

    def get_exact_solution(self, fs, t=0.):
        """
        Evaluate asymptotic solution of chosen order.

        :arg fs: `FunctionSpace` in which the solution should live.
        :kwarg t: current time.
        """
        assert self.order in (0, 1)
        if self.order == 0:
            self.zeroth_order_terms(t)
        else:
            self.first_order_terms(t)
        self.exact_solution = Function(fs)
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
        self.get_exact_solution(fs, t=0.)
        self.initial_value = self.exact_solution.copy()
        return self.initial_value

    def get_reference_mesh(self, n=50):
        """
        Set up a non-periodic, very fine mesh on the PDE domain.
        """
        reference_mesh = RectangleMesh(self.lx*n, self.ly*n, self.lx, self.ly)
        x, y = SpatialCoordinate(reference_mesh)
        reference_mesh.coordinates.interpolate(as_vector([x - self.lx/2, y - self.ly/2]))
        return reference_mesh

    def remove_periodicity(self, sol):
        """
        :arg sol: Function to remove periodicity of.
        """
        # Generate an identical non-periodic mesh
        nonperiodic_mesh = RectangleMesh(self.lx*self.n, self.ly*self.n, self.lx, self.ly)
        x, y = SpatialCoordinate(nonperiodic_mesh)
        nonperiodic_mesh.coordinates.interpolate(as_vector([x - self.lx/2, y - self.ly/2]))

        # Project into corresponding function space
        V = FunctionSpace(nonperiodic_mesh, sol.ufl_element())
        sol_np = Function(V)
        sol_np.project(sol)

        return sol_np

    def get_peaks(self, sol_periodic, reference_mesh_resolution=50):
        """
        Given a numerical solution of the test case, compute the metrics as given in [Huang et al 2008]:
          * h± : relative peak height for Northern / Southern soliton
          * C± : relative mean phase speed for Northern / Southern soliton
          * RMS: root mean square error
        The solution is projected onto a finer space in order to get a better approximation.

        :arg sol_periodic: Numerical solution of PDE.
        """
        sol = self.remove_periodicity(sol_periodic)

        # Split Northern and Southern halves of domain
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
        fs = FunctionSpace(reference_mesh, sol.ufl_element())
        sol_proj = Function(fs)
        sol_proj.project(sol)
        sol_upper_proj = Function(fs)
        sol_upper_proj.project(sol_upper)
        sol_lower_proj = Function(fs)
        sol_lower_proj.project(sol_lower)
        xcoords = Function(sol_proj.function_space())
        xcoords.interpolate(reference_mesh.coordinates[0])

        # Get relative mean peak height
        with sol_upper_proj.dat.vec_ro as vu:
            i_upper, self.h_upper = vu.max()
        with sol_lower_proj.dat.vec_ro as vl:
            i_lower, self.h_lower = vl.max()
        self.h_upper /= 0.1567020
        self.h_lower /= 0.1567020

        # Get relative mean phase speed
        xdat = xcoords.vector().gather()
        self.c_upper = (48 - xdat[i_upper])/47.18
        self.c_lower = (48 - xdat[i_lower])/47.18

        # Calculate RMS error
        init = self.remove_periodicity(self.initial_value.split()[1])
        init_proj = Function(fs)
        init_proj.project(init)
        sol_proj -= init_proj
        sol_proj *= sol_proj
        self.rms = sqrt(np.mean(sol_proj.vector().gather()))
