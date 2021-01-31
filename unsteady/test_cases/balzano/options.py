from thetis import *
from thetis.configuration import *

from adapt_utils.swe.utils import heaviside_approx
from adapt_utils.options import CoupledOptions

import numpy as np
import os


__all__ = ["BalzanoOptions"]


class BalzanoOptions(CoupledOptions):
    """
    Parameters for test case described in [1].

    [1] A. Balzano, "Evaluation of methods for numerical simulation of wetting and drying in
        shallow water flow models." Coastal Engineering 34.1-2 (1998): 83-107.
    """
    def __init__(self, n=1, bathymetry_type=1, plot_timeseries=False, **kwargs):
        self.timestepper = 'CrankNicolson'
        super(BalzanoOptions, self).__init__(**kwargs)
        self.plot_timeseries = plot_timeseries
        self.basin_x = 13800.0  # Length of wet region
        self.default_mesh = RectangleMesh(17*n, n, 1.5*self.basin_x, 1200.0)
        self.plot_pvd = True
        self.num_hours = kwargs.get('num_hours', 24)

        # Three possible bathymetries
        try:
            assert bathymetry_type in (1, 2, 3)
        except AssertionError:
            raise ValueError("`bathymetry_type` should be chosen from (1, 2, 3).")
        self.bathymetry_type = bathymetry_type
        self.di = os.path.join(self.di, 'bathymetry{:d}'.format(self.bathymetry_type))

        # Physical
        self.base_viscosity = 1e-6
        self.base_diffusivity = 0.15
        self.wetting_and_drying = True
        self.wetting_and_drying_alpha = Constant(0.43)
        friction = kwargs.get('friction', 'manning')
        try:
            assert friction in ('nikuradse', 'manning')
        except AssertionError:
            raise ValueError("Friction parametrisation '{:s}' not recognised.".format(friction))
        self.friction = friction
        self.average_size = 200e-6  # Average sediment size
        self.friction_coeff = 0.025
        self.ksp = None
        # Stabilisation
        self.stabilisation = None

        # Boundary conditions
        h_amp = 0.5  # Ocean boundary forcing amplitude
        h_T = self.num_hours/2*3600  # Ocean boundary forcing period
        self.elev_func = lambda t: h_amp*(-cos(2*pi*(t-(6*3600))/h_T)+1)
        self.elev_in = Constant(self.elev_func(0.0))

        # Time integration
        self.dt = 600.0
        self.end_time = self.num_hours*3600.0
        # self.dt_per_export = 6
        self.dt_per_export = 1
        self.implicitness_theta = 0.5

        # Timeseries
        self.wd_obs = []
        self.trange = np.linspace(0.0, self.end_time, self.num_hours+1)
        tol = 1e-8  # FIXME: Point evaluation hack
        self.xrange = np.linspace(tol, 1.5*self.basin_x-tol, 20)

        # Outputs  (NOTE: self.di has changed)
        self.eta_tilde_file = File(os.path.join(self.di, 'eta_tilde.pvd'))

    def set_quadratic_drag_coefficient(self, fs):
        if self.friction == 'nikuradse':
            return interpolate(self.get_cfactor(), fs)

    def get_cfactor(self, depth):
        try:
            assert hasattr(self, 'depth')
        except AssertionError:
            raise ValueError("Depth is undefined.")
        ksp = Constant(3*self.average_size)
        hc = conditional(depth > 0.001, depth, 0.001)
        aux = max_value(11.036*hc/ksp, 1.001)
        return 2*(0.4**2)/(ln(aux)**2)

    def set_bathymetry(self, fs):
        max_depth = 5.0
        x, y = SpatialCoordinate(fs.mesh())
        L = self.basin_x
        ξ = lambda X: L - X  # Coordinate transformation
        b1 = ξ(x)/L*max_depth
        expr = b1
        if self.bathymetry_type == 2:
            expr = conditional(le(abs(ξ(x) - 4000.0), 1000.0),
                               conditional(ge(ξ(x), 4000.0),
                                           (3000.0 + 2*(ξ(x) - 4000.0))/L*max_depth,
                                           3000.0/L*max_depth),
                               b1)
        elif self.bathymetry_type == 3:
            expr = conditional(le(abs(ξ(x) - 4000.0), 1000.0),
                               conditional(ge(ξ(x), 4000.0),
                                           (2000.0 + 3*(ξ(x) - 4000.0))/L*max_depth,
                                           (3000.0 - (ξ(x) - 3000.0))/L*max_depth),
                               b1)
        return interpolate(expr, fs)

    def set_boundary_conditions(self, prob, i):
        inflow_tag = 1
        outflow_tag = 2
        bottom_wall_tag = 3
        top_wall_tag = 4
        boundary_conditions = {
            'shallow_water': {
                inflow_tag: {'elev': self.elev_in},
                outflow_tag: {'un': Constant(0.0)},
                bottom_wall_tag: {'un': Constant(0.0)},
                top_wall_tag: {'un': Constant(0.0)},
            }
        }
        return boundary_conditions

    def update_boundary_conditions(self, t=0.0):
        self.elev_in.assign(self.elev_func(t) if 6*3600 <= t <= 18*3600 else 0.0)

    def set_initial_condition(self, prob):
        u, eta = prob.fwd_solutions[0].split()
        u.interpolate(as_vector([1.0e-7, 0.0]))
        eta.assign(0.0)

    def get_update_forcings(self, prob, i, adjoint=False):
        u, eta = prob.fwd_solutions[i].split()
        bathymetry_displacement = prob.equations[i].shallow_water.depth.wd_bathymetry_displacement
        depth = prob.depth[i]

        def update_forcings(t):
            self.update_boundary_conditions(t=t)

            # Update bathymetry and friction
            if self.friction == 'nikuradse':
                if self.wetting_and_drying:
                    depth.project(eta + bathymetry_displacement(eta) + prob.bathymetry[i])
                prob.fields[i].quadratic_drag_coefficient.interpolate(self.get_cfactor(depth))

        return update_forcings

    def get_export_func(self, prob, i):
        eta_tilde = Function(prob.P1DG[i], name="Modified elevation")
        self.eta_tilde_file._topology = None
        if self.plot_timeseries:
            u, eta = prob.fwd_solutions[i].split()
            b = prob.bathymetry[i]
            wd = Function(prob.P1DG[i], name="Heaviside approximation")

        def export_func():
            eta_tilde.project(self.get_eta_tilde(prob, i))
            self.eta_tilde_file.write(eta_tilde)

            if self.plot_timeseries:

                # Store modified bathymetry timeseries
                wd.project(heaviside_approx(-eta-b, self.wetting_and_drying_alpha))
                self.wd_obs.append([wd.at([x, 0]) for x in self.xrange])

        return export_func
