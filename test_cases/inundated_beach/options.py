from thetis import *
from thetis.configuration import *

from adapt_utils.swe.options import ShallowWaterOptions

import numpy as np


__all__ = ["BalzanoOptions"]


class BalzanoOptions(ShallowWaterOptions):
    """
    Parameters for test case in [...].
    """ # TODO: cite

    def __init__(self, approach='fixed_mesh', friction='nikuradse'):
        super(BoydOptions, self).__init__(approach)
        self.plot_pvd = True

        # Initial mesh
        try:
            assert os.path.exists('strip.msh')  # TODO: abspath
        except AssertionError:
            raise ValueError("Mesh does not exist or cannot be found. Please build it.")
        self.default_mesh = Mesh('strip.msh')

        # TODO
        self.base_viscosity = 1e-6
        self.wetting_and_drying = True
        self.wetting_and_drying_alpha = 0.43
        self.stabilisation = 'no'

        # Time integration
        self.dt = 600.0
        self.end_time = 24*3600.0
        self.dt_per_export = np.round(self.end_time/40, 0)
        self.dt_per_remesh = 20
        self.timestepper = 'CrankNicolson'

        # Adaptivity
        self.h_min = 1e-8
        self.h_max = 10.

    def set_drag_coefficient(self, fs):
        if friction == 'nikuradse':
            raise NotImplementedError  # TODO
            # self.drag_coefficient = Constant(cfactor)
        elif friction == 'manning':
            raise NotImplementedError  # TODO
            # friction_coeff = friction_coeff or 0.02
            # self.drag_coefficient = Constant(friction_coeff)
        else:
            raise NotImplementedError("Friction parametrisation '{:s}' not recognised.".format(friction))

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
        h_amp = 0.5    # Ocean boundary forcing amplitude
        h_T = 12*3600  # Ocean boundary forcing period
        ocean_elev_func = lambda t: h_amp*(-cos(2*pi*(t-(6*3600))/h_T)+1)
        # TODO: More to do
        boundary_conditions = {}
        return boundary_conditions

    def set_initial_condition(self, fs):
        """
        Set initial elevation and velocity using asymptotic solution.

        :arg fs: `FunctionSpace` in which the initial condition should live.
        """
        self.initial_value = Function(fs, name="Initial condition")
        u, eta = self.initial_value.split()
        u.assign(0.0)
        eta.assign(as_vector([1.0e-7, 0.0]))
        return self.initial_value
