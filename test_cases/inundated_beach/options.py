from thetis import *
from thetis.configuration import *

from adapt_utils.swe.tsunami.options import TsunamiOptions, heaviside_approx

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc

rc('text', usetex=True)


__all__ = ["BalzanoOptions"]


class BalzanoOptions(TsunamiOptions):
    """
    Parameters for test case described in [1].

    [1] A. Balzano, "Evaluation of methods for numerical simulation of wetting and drying in
        shallow water flow models." Coastal Engineering 34.1-2 (1998): 83-107.
    """

    def __init__(self, friction='manning', plot_timeseries=False, n=1, bathymetry_type=1, **kwargs):
        try:
            assert bathymetry_type in (1, 2, 3)
        except AssertionError:
            raise ValueError("`bathymetry_type` should be chosen from (1, 2, 3).")
        self.bathymetry_type = bathymetry_type
        self.plot_timeseries = plot_timeseries
        self.basin_x = 13800.0  # Length of wet region
        self.default_mesh = RectangleMesh(17*n, n, 1.5*self.basin_x, 1200.0)
        super(BalzanoOptions, self).__init__(**kwargs)
        self.plot_pvd = True
        self.num_hours = 24

        # Physical
        self.base_viscosity = 1e-6
        self.base_diffusivity = 0.15
        self.wetting_and_drying = True
        self.wetting_and_drying_alpha = Constant(0.43)
        try:
            assert friction in ('nikuradse', 'manning')
        except AssertionError:
            raise ValueError("Friction parametrisation '{:s}' not recognised.".format(friction))
        self.friction = friction
        self.average_size = 200e-6  # Average sediment size
        self.friction_coeff = 0.025

        # Stabilisation
        self.stabilisation = 'no'

        # Boundary conditions
        h_amp = 0.5  # Ocean boundary forcing amplitude
        h_T = self.num_hours/2*3600  # Ocean boundary forcing period
        self.elev_func = lambda t: h_amp*(-cos(2*pi*(t-(6*3600))/h_T)+1)

        # Time integration
        self.dt = 600.0
        self.end_time = self.num_hours*3600.0
        self.dt_per_export = 6
        self.dt_per_remesh = 6
        self.timestepper = 'CrankNicolson'
        # self.implicitness_theta = 0.5  # TODO

        # Adaptivity
        self.h_min = 1e-8
        self.h_max = 10.

        # Goal-Oriented
        self.qoi_mode = 'inundation_volume'

        P1DG = FunctionSpace(self.default_mesh, "DG", 1)  # FIXME
        self.get_initial_depth(VectorFunctionSpace(self.default_mesh, "CG", 2)*P1DG)  # FIXME

        # Timeseries
        self.wd_obs = []
        self.trange = np.linspace(0.0, self.end_time, self.num_hours+1)
        tol = 1e-8  # FIXME: Point evaluation hack
        self.xrange = np.linspace(tol, 1.5*self.basin_x-tol, 20)

    def set_quadratic_drag_coefficient(self, fs):
        if self.friction == 'nikuradse':
            self.quadratic_drag_coefficient = interpolate(self.get_cfactor(), fs)
        return self.quadratic_drag_coefficient

    def get_cfactor(self):
        try:
            assert hasattr(self, 'depth')
        except AssertionError:
            raise ValueError("Depth is undefined.")
        ksp = Constant(3*self.average_size)
        hc = conditional(self.depth > 0.001, self.depth, 0.001)
        aux = max_value(11.036*hc/ksp, 1.001)
        return 2*(0.4**2)/(ln(aux)**2)

    def set_manning_drag_coefficient(self, fs):
        if self.friction == 'manning':
            self.manning_drag_coefficient = Constant(self.friction_coeff or 0.02)
        return self.manning_drag_coefficient

    def set_bathymetry(self, fs, **kwargs):
        max_depth = 5.0
        x, y = SpatialCoordinate(fs.mesh())
        self.bathymetry = Function(fs, name="Bathymetry")
        L = self.basin_x
        ξ = lambda X: L - X  # Coordinate transformation
        b1 = ξ(x)/L*max_depth
        if self.bathymetry_type == 1:
            self.bathymetry.interpolate(b1)
        elif self.bathymetry_type == 2:
            self.bathymetry.interpolate(
                conditional(le(abs(ξ(x) - 4000.0), 1000.0),
                            conditional(ge(ξ(x), 4000.0),
                                        (3000.0 + 2*(ξ(x) - 4000.0))/L*max_depth,
                                        3000.0/L*max_depth),
                            b1))
        else:
            raise NotImplementedError  # TODO
        return self.bathymetry

    def set_viscosity(self, fs):
        self.viscosity = Function(fs)
        self.viscosity.assign(self.base_viscosity)
        return self.viscosity

    def set_coriolis(self, fs):
        return

    def set_boundary_conditions(self, fs):
        if not hasattr(self, 'elev_in'):
            self.set_boundary_surface()
        self.elev_in.assign(self.elev_func(0.0))
        inflow_tag = 1
        outflow_tag = 2
        bottom_wall_tag = 3
        top_wall_tag = 4
        boundary_conditions = {}
        boundary_conditions[inflow_tag] = {'elev': self.elev_in}
        boundary_conditions[outflow_tag] = {'un': Constant(0.0)}
        boundary_conditions[bottom_wall_tag] = {'un': Constant(0.0)}
        boundary_conditions[top_wall_tag] = {'un': Constant(0.0)}
        return boundary_conditions

    def update_boundary_conditions(self, t=0.0):
        self.elev_in.assign(self.elev_func(t) if 6*3600 <= t <= 18*3600 else 0.0)

    def set_initial_condition(self, fs):
        self.initial_value = Function(fs, name="Initial condition")
        u, eta = self.initial_value.split()
        u.interpolate(as_vector([1.0e-7, 0.0]))
        eta.assign(0.0)
        return self.initial_value

    def get_update_forcings(self, solver_obj):
        eta = solver_obj.fields.elev_2d
        bathymetry_displacement = solver_obj.eq_sw.bathymetry_displacement_mass_term.wd_bathymetry_displacement

        def update_forcings(t):
            self.update_boundary_conditions(t=t)

            # Update bathymetry and friction
            if self.friction == 'nikuradse':
                if self.wetting_and_drying:
                    self.depth.project(eta + bathymetry_displacement(eta) + self.bathymetry)
                self.quadratic_drag_coefficient.interpolate(self.get_cfactor())

        return update_forcings

    def get_export_func(self, solver_obj):
        bathymetry_displacement = solver_obj.eq_sw.bathymetry_displacement_mass_term.wd_bathymetry_displacement
        eta = solver_obj.fields.elev_2d
        b = solver_obj.fields.bathymetry_2d
        def export_func():
            self.eta_tilde.project(eta + bathymetry_displacement(eta))
            self.eta_tilde_file.write(self.eta_tilde)

            if self.plot_timeseries:

                # Store modified bathymetry timeseries
                P1DG = solver_obj.function_spaces.P1DG_2d
                wd = project(heaviside_approx(-eta-b, self.wetting_and_drying_alpha), P1DG)
                self.wd_obs.append([wd.at([x, 0]) for x in self.xrange])
        return export_func
