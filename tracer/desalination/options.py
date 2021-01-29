from thetis import *
from thetis.configuration import *

from adapt_utils.options import CoupledOptions


__all__ = ["DesalinationOutfallOptions"]


class DesalinationOutfallOptions(CoupledOptions):
    """
    Parameters for general desalination plant outfall scenarios. The simulation has two phases:

    * Spin-up: hydrodynamics only, driven by a tidal forcing;
    * Run:     hydrodynamics + salinity (interpreted as a passive tracer).
    """

    # Salinity specification
    background_salinity = PositiveFloat(39, help="""
        Background salinity, with units g/L.
        """).tag(config=False)

    # Tidal forcing
    M2_tide_period = PositiveFloat(12.4*3600).tag(config=False)

    def __init__(self, spun=False, **kwargs):
        super(DesalinationOutfallOptions, self).__init__(**kwargs)
        self.solve_swe = True
        self.solve_tracer = spun
        self.spun = spun

        self.h_min = 1.0e-03
        self.h_max = 1.0e+03
