# from thetis import *
# from thetis.configuration import *

from adapt_utils.unsteady.options import CoupledOptions


__all__ = ["TurbineOptions"]


# TODO
class TurbineOptions(CoupledOptions):
    def __init__(self, **kwargs):
        super(TurbineOptions, self).__init__(**kwargs)
        self.timestepper = 'CrankNicolson'
