"""
****************************************************************************************************
THIS FILE HAS LARGELY BEEN COPIED FROM GEOCLAW FOR THE PURPOSES OF AUTOMATIC DIFFERENTIATION.

See GeoClaw dtopotools Module `$CLAW/geoclaw/src/python/geoclaw/dtopotools.py` for original version.
****************************************************************************************************
"""
try:
    import adolc
    have_adolc = True
except ImportError:
    print("WARNING: pyadolc could not be imported")
    have_adolc = False

import numpy
from time import clock

import clawpack.geoclaw.dtopotools
from clawpack.geoclaw.dtopotools import *


class Fault(clawpack.geoclaw.dtopotools.Fault):
    """
    Subclass of the CLAWPACK `Fault` class allowing automatic differentation using pyadolc.
    """
    def __init__(self, *args, times=[1.0], **kwargs):
        super(Fault, self).__init__(**kwargs)
        self.dtopo = DTopography()
        if len(args) == 1:
            coordinates = args[0]
            self.dtopo.X, self.dtopo.Y = coordinates[:][0], coordinates[:][1]
        elif len(args) == 2:
            self.dtopo.x, self.dtopo.y = args
            self.dtopo.X, self.dtopo.Y = numpy.meshgrid(self.dtopo.x, self.dtopo.y)
        else:
            raise ValueError
        for subfault in self.subfaults:
            subfault.X, subfault.Y = self.dtopo.X, self.dtopo.Y
        if len(times) != 1:
            raise NotImplementedError
        self.dtopo.times = times

    def create_dtopography(self, active=True, verbose=False):
        """
        Annotated version of topography method.
        """
        if active and not have_adolc:
            raise ValueError("Cannot annotate the rupture process without an appropriate AD tool.")
        if self.rupture_type != 'static':
            raise NotImplementedError("Only static ruptures currently considered.")

        num_subfaults = len(self.subfaults)
        tic = clock()
        msg = "created topography for subfault {:d}/{:d} ({:.1f} seconds)"
        dz = numpy.zeros(self.dtopo.X.shape)
        if active:
            dz = adolc.adouble(dz)
        for k, subfault in enumerate(self.subfaults):
            subfault.okada()
            dz += subfault.dtopo.dZ[0, :, :]
            if k % 10 == 0 and verbose:
                print(msg.format(k+1, num_subfaults, clock() - tic))
                tic = clock()
        if active:
            self.dtopo.dZ_a = dz
            self.dtopo.dZ = numpy.array([dzi.val for dzi in numpy.ravel(dz)]).reshape((1, ) + dz.shape)
        else:
            self.dtopo.dZ = dz

        return self.dtopo


class SubFault(clawpack.geoclaw.dtopotools.SubFault):
    """
    Subclass of the CLAWPACK `SubFault` class allowing non-rectangular meshes.
    """
    def okada(self):
        if self.coordinate_specification == 'triangular':
            raise NotImplementedError

        # Okada model assumes x,y are at bottom center:
        x_bottom = self.centers[2][0]
        y_bottom = self.centers[2][1]
        depth_bottom = self.centers[2][2]

        length = self.length
        width = self.width
        slip = self.slip

        halfL = 0.5*length
        w = width

        # convert angles to radians:
        ang_dip = DEG2RAD * self.dip
        ang_rake = DEG2RAD * self.rake
        ang_strike = DEG2RAD * self.strike

        # Use upper case convention for 2d
        if hasattr(self, 'x') and hasattr(self, 'y'):
            X, Y = self.X, self.Y
        elif hasattr(self, 'X') and hasattr(self, 'Y'):
            X, Y = self.X, self.Y
        else:
            raise ValueError

        # Convert distance from (X,Y) to (x_bottom,y_bottom) from degrees to
        # meters:
        xx = LAT2METER * numpy.cos(DEG2RAD * Y) * (X - x_bottom)
        yy = LAT2METER * (Y - y_bottom)

        # Convert to distance along strike (x1) and dip (x2):
        x1 = xx*numpy.sin(ang_strike) + yy*numpy.cos(ang_strike)
        x2 = xx*numpy.cos(ang_strike) - yy*numpy.sin(ang_strike)

        # In Okada's paper, x2 is distance up the fault plane, not down dip:
        x2 = -x2

        p = x2*numpy.cos(ang_dip) + depth_bottom*numpy.sin(ang_dip)
        q = x2*numpy.sin(ang_dip) - depth_bottom*numpy.cos(ang_dip)

        f1 = self._strike_slip(x1 + halfL, p, ang_dip, q)
        f2 = self._strike_slip(x1 + halfL, p - w, ang_dip, q)
        f3 = self._strike_slip(x1 - halfL, p, ang_dip, q)
        f4 = self._strike_slip(x1 - halfL, p - w, ang_dip, q)

        g1 = self._dip_slip(x1 + halfL, p, ang_dip, q)
        g2 = self._dip_slip(x1 + halfL, p - w, ang_dip, q)
        g3 = self._dip_slip(x1 - halfL, p, ang_dip, q)
        g4 = self._dip_slip(x1 - halfL, p - w, ang_dip, q)

        # Displacement in direction of strike and dip:
        ds = slip * numpy.cos(ang_rake)
        dd = slip * numpy.sin(ang_rake)

        us = (f1 - f2 - f3 + f4) * ds
        ud = (g1 - g2 - g3 + g4) * dd

        dz = us + ud

        dtopo = DTopography()
        dtopo.X = X
        dtopo.Y = Y
        dtopo.dZ = numpy.array(dz, ndmin=3)
        dtopo.times = [0.0, ]
        self.dtopo = dtopo
