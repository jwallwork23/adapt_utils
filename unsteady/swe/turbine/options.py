# from thetis import *
# from thetis.configuration import *

from adapt_utils.unsteady.options import CoupledOptions


# TODO
class UnsteadyTurbineOptions(SteadyTurbineOptions):
    def __init__(self, **kwargs):
        super(UnsteadyTurbineOptions, self).__init__(**kwargs)
        self.timestepper = 'CrankNicolson'
