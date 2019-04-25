from thetis import *
from thetis.configuration import *
import math

import numpy as np

from adapt_utils.options import Options


__all__ = ["TwoTurbineOptions"]


class TwoTurbineOptions(Options):
    name = 'Parameters for the 2 turbine problem'
    mode = 'Turbine'

    # solver parameters
    dt = PositiveFloat(20.).tag(config=True)
    end_time = PositiveFloat(18.).tag(config=True)

    # adaptivity parameters
    target_vertices = PositiveFloat(1000, help="Target number of vertices").tag(config=True)
    adapt_field = Unicode('fluid_speed', help="Adaptation field of interest, from {'fluid_speed', 'elevation', 'both'}.").tag(config=True)
    h_min = PositiveFloat(1e-4, help="Minimum element size").tag(config=True)
    h_max = PositiveFloat(10., help="Maximum element size").tag(config=True)
    rescaling = PositiveFloat(0.85, help="Scaling parameter for target number of vertices.").tag(config=True)

    # physical parameters
    viscosity = NonNegativeFloat(1.).tag(config=True)
    symmetric_viscosity = Bool(False, help="Symmetrise viscosity term").tag(config=True)
    drag_coefficient = NonNegativeFloat(0.0025).tag(config=True)

    # turbine parameters
    turbine_diameter = PositiveFloat(18.).tag(config=True)
    thrust_coefficient = NonNegativeFloat(0.8).tag(config=True)
    region_of_interest = List(default_value=[(50., 100., 9.), (400., 100., 9.)]).tag(config=True)

    def __init__(self, approach='fixed_mesh', adapt_field='fluid_speed'):
        super(TwoTurbineOptions, self).__init__(approach)
        self.adapt_field = adapt_field
        try:
            assert self.adapt_field in ('fluid_speed', 'elevation', 'both')
        except:
            raise ValueError('Field for adaptation {:s} not recognised.'.format(self.adapt_field))
        self.default_mesh = RectangleMesh(100, 20, 1000., 200.)
        self.bathymetry = Constant(40.)

        # Correction to account for the fact that the thrust coefficient is based on an upstream
        # velocity whereas we are using a depth averaged at-the-turbine velocity (see Kramer and
        # Piggott 2016, eq. (15))
        D = self.turbine_diameter
        A_T = math.pi*(D/2)**2
        correction = 4/(1+math.sqrt(1-A_T/(40.*D)))**2
        self.thrust_coefficient *= correction
        # NOTE, that we're not yet correcting power output here, so that will be overestimated

    def set_bcs(self):
        left_tag = 1
        right_tag = 2
        top_bottom_tag = 3
        freeslip_bc = {'un': Constant(0.)}
        self.boundary_conditions = {
          left_tag: {'uv': self.inflow},
          right_tag: {'elev': Constant(0.)},
          top_bottom_tag: freeslip_bc,
        }

    def set_inflow(self, fs):
        self.inflow = Function(fs).interpolate(as_vector([3., 0.]))
        return self.inflow


class UnsteadyTwoTurbineOptions(TwoTurbineOptions):
    def __init__(self, approach='fixed_mesh', adapt_field='fluid_speed'):
        super(UnsteadyTwoTurbineOptions, self).__init__(approach)
        self.dt = 3
        #self.T_tide = 1.24*3600 
        self.T_tide = 1.24*60 
        self.T_ramp = 1*T_tide
        self.end_time = T_ramp+2*T_tide
        self.dt_per_export = 10
        self.dt_per_remesh = 20
        self.t_const = Constant(0.)
        self.hmax = Constant(0.5)
        self.omega = Constant(2*math.pi/self.T_tide)

    def update_forcings(self, t):
        self.t_const.assign(t)
        self.elev_func_in.assign(self.hmax*math.cos(self.omega*(t_const-self.T_ramp)))
        self.elev_func_out.assign(self.hmax*math.cos(self.omega*(t_const-self.T_ramp)+math.pi))
