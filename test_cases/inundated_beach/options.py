from thetis import *
from thetis.configuration import *

from adapt_utils.swe.options import ShallowWaterOptions

import numpy as np


__all__ = ["BalzanoOptions"]


class BalzanoOptions(ShallowWaterOptions):
    """
    Parameters for test case in [...].
    """ # TODO: cite

    def __init__(self, approach='fixed_mesh', friction='manning'):
        super(BalzanoOptions, self).__init__(approach)
        self.plot_pvd = True

        # Initial mesh
        try:
            assert os.path.exists('strip.msh')  # TODO: abspath
        except AssertionError:
            raise ValueError("Mesh does not exist or cannot be found. Please build it.")
        self.default_mesh = Mesh('strip.msh')

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
        h_amp = 0.5    # Ocean boundary forcing amplitude
        h_T = 12*3600  # Ocean boundary forcing period
        self.elev_func = lambda t: h_amp*(-cos(2*pi*(t-(6*3600))/h_T)+1)

        # Time integration
        self.dt = 600.0
        self.end_time = 24*3600.0
        self.dt_per_export = 6
        self.dt_per_remesh = 20
        self.timestepper = 'CrankNicolson'
        # self.implicitness_theta = 0.5  # TODO

        # Adaptivity
        self.h_min = 1e-8
        self.h_max = 10.

        # Outputs
        self.wd_bath_file = File(os.path.join(self.di, 'moving_bath.pvd'))
        P1DG = FunctionSpace(self.default_mesh, "DG", 1)  # FIXME
        self.moving_bath = Function(P1DG, name='Moving bathymetry')
        self.eta_tilde_file = File(os.path.join(self.di, 'eta_tilde.pvd'))
        self.eta_tilde = Function(P1DG, name='Modified elevation')

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

    def set_bathymetry(self, fs):
        max_depth = 5.0
        basin_x = 13800.0
        x, y = SpatialCoordinate(fs.mesh())
        self.bathymetry = Function(fs, name="Bathymetry")
        self.bathymetry.interpolate((1.0 - x/basin_x)*max_depth)
        return self.bathymetry

    def set_viscosity(self, fs):
        self.viscosity = Function(fs)
        self.viscosity.assign(self.base_viscosity)
        return self.viscosity

    def set_boundary_conditions(self, fs):
        if not hasattr(self, 'elev_in'):
            self.set_boundary_surface()
        self.elev_in.assign(self.elev_func(0.0))
        inflow_tag = 1
        wall_tag = 2
        outflow_tag = 3
        boundary_conditions = {}
        boundary_conditions[inflow_tag] = {'elev': self.elev_in}
        boundary_conditions[wall_tag] = {'un': Constant(0.0)}
        return boundary_conditions

    def update_boundary_conditions(self, t=0.0):
        self.elev_in.assign(self.elev_func(t) if 6*3600 <= t <= 18*3600 else 0.0)

    def set_initial_condition(self, fs):
        """
        Set initial elevation and velocity using asymptotic solution.

        :arg fs: `FunctionSpace` in which the initial condition should live.
        """
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
        def export_func():
            self.moving_bath.project(self.bathymetry + bathymetry_displacement(eta))
            self.wd_bath_file.write(self.moving_bath)
            self.eta_tilde.project(eta + bathymetry_displacement(eta))
            self.eta_tilde_file.write(self.eta_tilde)
        return export_func

    def set_qoi_kernel(self, solver_obj):  # TODO: Could play around with this a bit
        self.kernel = Function(solver_obj.function_spaces.V_2d)
        k_eta = self.kernel.split()[1]
        dry = conditional(ge(self.bathymetry, 0), 0, 1)
        bathymetry_displacement = solver_obj.eq_sw.bathymetry_displacement_mass_term.wd_bathymetry_displacement
        eta = solver_obj.fields.elev_2d
        self.eta_tilde.project(eta + bathymetry_displacement(eta))
        k_eta.interpolate(dry*(self.eta_tilde-self.bathymetry)/eta)
        return self.kernel
