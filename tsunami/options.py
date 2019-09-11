from thetis import *

import numpy as np


__all__ = ["BoydOptions"]

# TODO: More setups to consider

class BoydOptions(Options):
    """
    Parameters for test case in [Boyd et al. 1996].
    """

    # Domain
    nx = PositiveInteger(0, help="Mesh resolution in x- and y-directions.").tag(config=True)

    # Physical
    viscosity = FiredrakeScalarExpression(Constant(0.), help="(Scalar) viscosity field for tracer problem.").tag(config=True)
    soliton_amplitude = PositiveFloat(0.395).tag(config=True)

    def __init__(self, approach='fixed_mesh', periodic=True):
        super(BoydOptions, self).__init__(approach)

        # Initial mesh
        lx = 48
        ly = 24
        n = 2**self.nx
        if periodic:
            self.default_mesh = PeriodicRectangleMesh(lx*n, ly*n, lx, ly, direction='x')
        else:
            self.default_mesh = RectangleMesh(lx*n, ly*n, lx, ly)
        x, y = SpatialCoordinate(self.default_mesh)
        self.default_mesh.coordinates.interpolate(as_vector([x - lx/2, y - ly/2]))
        self.x, self.y = SpatialCoordinate(self.default_mesh)

        # Boundary conditions  # TODO: this doesn't seem right
        self.boundary_conditions[1] = {'uv': Constant(0.)}
        self.boundary_conditions[2] = {'uv': Constant(0.)}
        self.boundary_conditions[3] = {'uv': Constant(0.)}
        self.boundary_conditions[4] = {'uv': Constant(0.)}

        # Physical
        self.base_viscosity = 0.
        self.g = 1.

        # Time integration
        self.dt = 0.05
        self.start_time = 30.
        self.end_time = 120.
        self.dt_per_export = 10
        self.dt_per_remesh = 20

        # Adaptivity
        self.h_min = 1e-3
        self.h_max = 10.


        # Hermite series coefficients
        u = np.zeros(28)
        v = np.zeros(28)
        eta = np.zeros(28)
        u[0] = 1.789276
        u[2] = 0.1164146
        u[4] = -0.3266961e-3
        u[6] = -0.1274022e-2
        u[8] = 0.4762876e-4
        u[10] = -0.1120652e-5
        u[12] = 0.1996333e-7
        u[14] = -0.2891698e-9
        u[16] = 0.3543594e-11
        u[18] = -0.3770130e-13
        u[20] = 0.3547600e-15
        u[22] = -0.2994113e-17
        u[24] = 0.2291658e-19
        u[26] = -0.1178252e-21
        v[3] = -0.6697824e-1
        v[5] = -0.2266569e-2
        v[7] = 0.9228703e-4
        v[9] = -0.1954691e-5
        v[11] = 0.2925271e-7
        v[13] = -0.3332983e-9
        v[15] = 0.2916586e-11
        v[17] = -0.1824357e-13
        v[19] = 0.4920951e-16
        v[21] = 0.6302640e-18
        v[23] = -0.1289167e-19
        v[25] = 0.1471189e-21
        eta[0] = -3.071430
        eta[2] = -0.3508384e-1
        eta[4] = -0.1861060e-1
        eta[6] = -0.2496364e-3
        eta[8] = 0.1639537e-4
        eta[10] = -0.4410177e-6
        eta[12] = 0.8354759e-9
        eta[14] = -0.1254222e-9
        eta[16] = 0.1573519e-11
        eta[18] = -0.1702300e-13
        eta[20] = 0.1621976e-15
        eta[22] = -0.1382304e-17
        eta[24] = 0.1066277e-19
        eta[26] = -0.1178252e-21
        self.hermite_coeffs = {'u': u, 'v': v, 'eta': eta}

    def polynomials(self):
        """
        Get Hermite polynomials
        """
        polys = [Constant(1.), 2*self.y]
        for i in range(2, 28):
            polys.append(2*self.y*polys[i-1] - 2*(i-1)*polys[i-2])
        return polys

    def xi(self, t=0.):
        """
        :arg t: current time.
        :return: time shifted x-coordinate.
        """
        c = -1/3
        if self.order == 1:
            c -= 0.395*self.soliton_amplitude*self.soliton_amplitude
        return self.x - c*t

    def phi(self, t=0.):
        """
        :arg t: current time.
        :return: sech^2 term.
        """
        B = self.soliton_amplitude
        A = 0.771*B*B
        return A*(1/(cosh(B*self.xi(t))**2))

    def dphidx(self, t=0.):
        """
        :arg t: current time. 
        :return: tanh * phi term.
        """
        B = self.soliton_amplitude
        return -2*B*self.phi(t)*tanh(B*self.xi(t))

    def psi(self):
        """
        :arg t: current time. 
        :return: exp term.
        """
        return exp(-0.5*self.y*self.y)

    def zeroth_order_terms(self, t=0.):
        """
        :arg t: current time.
        :return: zeroth order asymptotic solution for test problem of Boyd.
        """
        self.terms = {}
        self.terms['u'] = self.phi(t)*0.25*(-9 + 6*self.y*self.y)*self.psi()
        self.terms['v'] = 2 * self.y*self.dphidx(t)*self.psi()
        self.terms['eta'] = self.phi(t)*0.25*(3 + 6*self.y*self.y)*self.psi()

    def first_order_terms(self, t=0.):
        """
        :arg t: current time.
        :return: first order asymptotic solution for test problem of Boyd.
        """
        C = - 0.395*self.soliton_amplitude*self.soliton_amplitude
        phi = self.phi(t)
        coeffs = self.hermite_coeffs
        polys = self.polynomials()
        self.zeroth_order_terms(t)
        # NOTE: The last psi in the following is not included
        self.terms['u'] += C*phi*0.5625*(3 + 2*self.y*self.y)*self.psi()
        self.terms['u'] += phi*phi*self.psi()*sum(coeffs['u'][i]*polys[i] for i in range(28))
        self.terms['v'] += self.dphidx(t)*phi*self.psi()*sum(coeffs['v'][i]*polys[i] for i in range(28))
        self.terms['eta'] += C*phi*0.5625 * (-5 + 2*self.y*self.y)*self.psi()
        self.terms['eta'] += phi*phi*self.psi()*sum(coeffs['eta'][i]*polys[i] for i in range(28))

    def set_viscosity(self, fs):
        self.viscosity = Constant(self.base_viscosity)
        return self.viscosity

    def set_bathymetry(self, fs):
        self.bathymetry = Constant(1.)
        return self.bathymetry

    def set_coriolis(self, fs, plane='beta'):  # TODO: f-plane, beta-plane and sin approximations
        x, y = SpatialCoordinate(fs.mesh())
        self.coriolis = Function(fs)
        self.coriolis.interpolate(y)
        return self.coriolis

    def exact_solution(self, fs, t, order=0):
        assert order in (0, 1)
        if order == 0:
            self.zeroth_order_terms()
        else:
            self.first_order_terms()
        self.solution = Function(fs)
        u, eta = self.solution.split()
        u.interpolate(as_vector([self.terms['u'], self.terms['v']]))
        eta.interpolate(self.terms['eta'])
        u.rename('Asymptotic velocity')
        eta.rename('Asymptotic elevation')
        return self.solution

    def set_initial_condition(self, fs):
        self.initial_value = self.exact_solution(t=0.)
        return self.initial_value
