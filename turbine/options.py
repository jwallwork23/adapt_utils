from thetis import *
from thetis.configuration import *

import numpy as np

from adapt_utils.options import Options


__all__ = ["TurbineOptions"]


class TurbineOptions(Options):
    name = 'Parameters for the 2 turbine problem'
    mode = 'Turbine'

    # solver parameters
    dt = PositiveFloat(20.).tag(config=True)

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
        super(TurbineOptions, self).__init__(approach)
        self.adapt_field = adapt_field
        try:
            assert self.adapt_field in ('fluid_speed', 'elevation', 'both')
        except:
            raise ValueError('Field for adaptation {:s} not recognised.'.format(self.adapt_field))
