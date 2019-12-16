from thetis import *
from thetis.configuration import *

import numpy as np

from adapt_utils.swe.options import ShallowWaterOptions


__all__ = ["SteadyTurbineOptions", "UnsteadyTurbineOptions"]


# Default: Newton with line search; solve linear system exactly with LU factorisation
default_params = {
    'mat_type': 'aij',
    'snes_type': 'newtonls',
    'snes_rtol': 1e-3,
    'snes_atol': 1e-16,
    'snes_max_it': 100,
    'snes_linesearch_type': 'bt',
    'snes_monitor': None,
    'ksp_type': 'preonly',
    'pc_type': 'lu',
    'pc_factor_mat_solver_type': 'mumps',
}
keys = {key for key in default_params if not 'snes' in key}
default_adjoint_params = {}
for key in keys:
    default_adjoint_params[key] = default_params[key]


class SteadyTurbineOptions(ShallowWaterOptions):
    """
    Base class holding parameters for steady state tidal turbine problems.
    """

    # Solver parameters
    params = PETScSolverParameters(default_params).tag(config=True)
    adjoint_params = PETScSolverParameters(default_adjoint_params).tag(config=True)

    def __init__(self, approach='fixed_mesh', num_iterations=1):
        super(SteadyTurbineOptions, self).__init__(approach)
        self.timestepper = 'SteadyState'
        self.dt = 20.
        self.end_time = num_iterations*self.dt - 0.2
        self.bathymetry = Constant(40.0)
        self.viscosity = Constant(self.base_viscosity)
        self.lax_friedrichs = True
        self.drag_coefficient = Constant(0.0025)

        # Adaptivity
        self.h_min = 1e-5
        self.h_max = 500.0

    def set_viscosity(self, fs):
        self.viscosity.assign(self.base_viscosity)
        return self.viscosity

    def set_inflow(self, fs):
        self.inflow = Function(fs).interpolate(as_vector([3., 0.]))
        return self.inflow

    def thrust_coefficient_correction(self):
        """
        Correction to account for the fact that the thrust coefficient is based on an upstream
        velocity whereas we are using a depth averaged at-the-turbine velocity (see Kramer and
        Piggott 2016, eq. (15))
        """
        D = self.turbine_diameter
        A_T = pi*(D/2)**2
        correction = 4/(1+sqrt(1-A_T/(40.*D)))**2
        self.thrust_coefficient *= correction
        # NOTE: We're not yet correcting power output here, so that will be overestimated

    def set_bcs(self, fs):
        pass


# TODO: bring below up to date
class UnsteadyTurbineOptions(SteadyTurbineOptions):
    def __init__(self, approach='fixed_mesh'):
        super(UnsteadyTurbineOptions, self).__init__(approach)

        # Solver
        # self.params = {'ksp_type': 'gmres',
        #                'pc_type': 'fieldsplit',
        #                'pc_fieldsplit_type': 'multiplicative',
        #                'snes_type': 'newtonls',
        #                #'snes_rtol': 1e-5,
        #                'snes_monitor': None,}
        # self.params = {'ksp_type': 'gmres',
        #                'pc_type': 'lu',
        #                'pc_factor_mat_solver_type': 'mumps',
        #                'snes_type': 'newtonls',
        #                'snes_monitor': None,}

        # Time period and discretisation
        self.dt = 3
        self.timestepper = 'CrankNicolson'
        self.T_tide = 1.24*3600
        self.T_ramp = 1*self.T_tide
        self.end_time = self.T_ramp+2*self.T_tide
        self.dt_per_export = 10
        self.dt_per_remesh = 10  # FIXME: solver seems to go out of sync if this != dt_per_export

        # Boundary forcing
        self.hmax = 0.5
        self.omega = 2*pi/self.T_tide

        # Turbines
        self.base_viscosity = 3.
        self.thrust_coefficient = 7.6

    def set_boundary_surface(self, fs):
        self.elev_in = Function(fs)
        self.elev_out = Function(fs)

class Unsteady2TurbineOptions(UnsteadyTurbineOptions):
    """Parameters for the unsteady 2 turbine problem"""

    # Turbine parameters
    turbine_diameter = PositiveFloat(18.).tag(config=True)
    thrust_coefficient = NonNegativeFloat(7.6).tag(config=True)
    base_viscosity = NonNegativeFloat(3., help="Fluid viscosity (assumed constant).").tag(config=True)
    depth = PositiveFloat(40., help="Water depth (assumes flat bathymetry).").tag(config=True)

    def __init__(self, approach='fixed_mesh'):
        super(Unsteady2TurbineOptions, self).__init__(approach)
        self.default_mesh = RectangleMesh(100, 20, 1000., 200.)

        # Tidal farm
        D = self.turbine_diameter
        self.region_of_interest = [(325, 100, D/2), (675, 100, D/2)]
        self.thrust_coefficient_correction()

    def set_initial_condition(self, fs):
        x, y = SpatialCoordinate(fs.mesh())
        q_init = Function(fs)
        self.uv_init, self.elev_init = q_init.split()
        self.uv_init.interpolate(as_vector([1e-8, 0.]))
        self.elev_init.interpolate(-1/1000*(x-500))  # linear from -1 to 1
        return q_init

    def set_bcs(self):  # TODO: standardise with above
        left_tag = 1
        right_tag = 2
        top_bottom_tag = 3
        freeslip_bc = {'un': Constant(0.)}
        self.boundary_conditions = {
          left_tag: {'elev': self.elev_in},
          right_tag: {'elev': self.elev_out},
          top_bottom_tag: freeslip_bc,
        }


class Unsteady15TurbineOptions(UnsteadyTurbineOptions):
    """Parameters for the unsteady 15 turbine problem"""

    # Turbine parameters
    turbine_diameter = PositiveFloat(20.).tag(config=True)
    thrust_coefficient = NonNegativeFloat(7.6).tag(config=True)
    base_viscosity = NonNegativeFloat(3., help="Fluid viscosity (assumed constant).").tag(config=True)
    depth = PositiveFloat(50., help="Water depth (assumes flat bathymetry).").tag(config=True)

    def __init__(self, approach='fixed_mesh'):
        super(Unsteady15TurbineOptions, self).__init__(approach)
        self.default_mesh = RectangleMesh(150, 50, 3000., 1000.)    # FIXME: wrong ids
        self.default_mesh.coordinates.dat.data[:] -= [1500., 500.]  # FIXME: not parallel
        self.h_max = 100

        # Tidal farm
        D = self.turbine_diameter
        delta_x = 10*D
        delta_y = 7.5*D
        for i in [-2, -1, 0, 1, 2]:
            for j in [-1, 0, 1]:
                self.region_of_interest.append((i*delta_x, j*delta_y, D/2))
        self.thrust_coefficient_correction()

    def set_initial_condition(self, fs):
        x, y = SpatialCoordinate(fs.mesh())
        q_init = Function(fs)
        self.uv_init, self.elev_init = q_init.split()
        self.uv_init.interpolate(as_vector([1e-8, 0.]))
        self.elev_init.interpolate(-1/3000*x)  # linear from -1 to 1
        return q_init

    def set_bcs(self):  # TODO: standardise with above
        bottom_tag = 1
        right_tag = 2
        top_tag = 3
        left_tag = 4
        freeslip_bc = {'un': Constant(0.)}
        self.boundary_conditions = {
          left_tag: {'elev': self.elev_in},
          right_tag: {'elev': self.elev_out},
          top_tag: freeslip_bc,
          bottom_tag: freeslip_bc,
        }
