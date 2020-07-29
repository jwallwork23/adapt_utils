"""
****************************************************************************************************
THIS FILE HAS LARGELY BEEN COPIED FROM GEOCLAW FOR THE PURPOSES OF AUTOMATIC DIFFERENTIATION.

See GeoClaw dtopotools Module `$CLAW/geoclaw/src/python/geoclaw/dtopotools.py` for original version.
****************************************************************************************************
"""
import adolc

import numpy
from time import clock

import clawpack.geoclaw.dtopotools
from clawpack.geoclaw.dtopotools import *
from clawpack.geoclaw.data import DEG2RAD, LAT2METER


class Fault(clawpack.geoclaw.dtopotools.Fault):
    """Subclass of the CLAWPACK Fault class allowing automatic differentation using pyadolc."""
    def __init__(self, x, y, times=[1.0], **kwargs):
        super(Fault, self).__init__(**kwargs)
        self.dtopo = DTopography()
        self.dtopo.x = x
        self.dtopo.y = y
        self.dtopo.X, self.dtopo.Y = numpy.meshgrid(self.dtopo.x, self.dtopo.y)
        if len(times) != 1:
            raise NotImplementedError
        self.dtopo.times = times

    def create_dtopography(self, active=True, verbose=False):
        """Annotated version of topography method."""

        if self.rupture_type != 'static':
            raise NotImplementedError("Only static ruptures currently considered.")

        num_subfaults = len(self.subfaults)
        tic = clock()
        msg = "created topography for subfault {:d}/{:d} ({:.1f} seconds)"
        dz = numpy.zeros(self.dtopo.X.shape)
        if active:
            dz = adolc.adouble(dz)
        for k, subfault in enumerate(self.subfaults):
            subfault.okada(self.dtopo.x, self.dtopo.y)
            dz += subfault.dtopo.dZ[0, :, :]
            if k % 10 == 0:
                print(msg.format(k+1, num_subfaults, clock() - tic))
                tic = clock()
        self.dtopo.dZ_a = dz
        self.dtopo.dZ = numpy.array([dzi.val for dzi in numpy.ravel(dz)]).reshape((1, ) +  dz.shape)

        return self.dtopo
