"""
Extension of GeoClaw's :class:`Fault` to allow automatic differentiation using ADOL-C. See
GeoClaw dtopotools Module `$CLAW/geoclaw/src/python/geoclaw/dtopotools.py` for original version.

**********************************************************************************************
*  NOTE: This file is based on GeoClaw (http://www.clawpack.org/geoclaw) and contains some   *
*        copied code.                                                                        *
**********************************************************************************************
"""
from __future__ import absolute_import
try:
    import adolc
    have_adolc = True
except ImportError:
    import warnings
    warnings.warn("PyADOL-C could not be imported")
    have_adolc = False

import numpy as np
from time import perf_counter

try:
    from clawpack.geoclaw.dtopotools import *
except ImportError:
    from dtopotools import *

ClawFault = Fault
ClawSubFault = SubFault


class Fault(ClawFault):
    """
    Subclass of the CLAWPACK `Fault` class allowing automatic differentation using pyadolc.
    """
    def __init__(self, *args, times=[1.0], **kwargs):
        super(Fault, self).__init__(**kwargs)
        self.dtopo = DTopography()
        if len(args) == 1:
            coordinates = args[0]
            self.dtopo.X, self.dtopo.Y = coordinates[:, 0], coordinates[:, 1]
        elif len(args) == 2:
            self.dtopo.x, self.dtopo.y = args
            self.dtopo.X, self.dtopo.Y = np.meshgrid(self.dtopo.x, self.dtopo.y)
        else:
            raise ValueError
        for subfault in self.subfaults:
            subfault.X, subfault.Y = self.dtopo.X, self.dtopo.Y
        if len(times) != 1:
            raise NotImplementedError
        self.dtopo.times = times

    def create_dtopography(self, active=True, **kwargs):
        """
        Modified version of :attr:`create_dtopography` which accounts for automatic differentation (AD).
        There are two cases:
          1. Passive mode, where no AD is applied;
          2. Active mode, where operations performed during the source model are annotated to tape;
        """
        if self.rupture_type != 'static':
            raise NotImplementedError("Only static ruptures currently considered.")
        if active:
            if not have_adolc:
                raise ValueError("Cannot annotate the rupture process without an appropriate AD tool.")
            self._create_dtopography_active(**kwargs)
        else:
            self._create_dtopography_passive(**kwargs)

    def _create_dtopography_passive(self, verbose=False):
        num_subfaults = len(self.subfaults)
        tic = perf_counter()
        msg = "created topography for subfault {:d}/{:d} ({:.1f} seconds)"
        dz = np.zeros(self.dtopo.X.shape)
        for k, subfault in enumerate(self.subfaults):
            subfault.okada()
            dz += subfault.dtopo.dZ[0, :, :].reshape(dz.shape)
            if k % 10 == 0 and verbose:
                print(msg.format(k+1, num_subfaults, perf_counter() - tic))
                tic = perf_counter()
        self.dtopo.dZ = dz
        return self.dtopo

    def _create_dtopography_active(self, verbose=False):
        num_subfaults = len(self.subfaults)
        tic = perf_counter()
        msg = "created topography for subfault {:d}/{:d} ({:.1f} seconds)"
        dz = np.zeros(self.dtopo.X.shape)
        dz = adolc.adouble(dz)
        for k, subfault in enumerate(self.subfaults):
            subfault.okada()
            dz += subfault.dtopo.dZ[0, :, :].reshape(dz.shape)
            if k % 10 == 0 and verbose:
                print(msg.format(k+1, num_subfaults, perf_counter() - tic))
                tic = perf_counter()
        self.dtopo.dZ_a = dz
        self.dtopo.dZ = np.array([dzi.val for dzi in np.ravel(dz)]).reshape((1, ) + dz.shape)
        return self.dtopo


class SubFault(ClawSubFault):
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
            X, Y = self.x, self.y
        elif hasattr(self, 'X') and hasattr(self, 'Y'):
            X, Y = self.X, self.Y
        else:
            raise ValueError

        # Convert distance from (X,Y) to (x_bottom,y_bottom) from degrees to
        # meters:
        xx = LAT2METER * np.cos(DEG2RAD * Y) * (X - x_bottom)
        yy = LAT2METER * (Y - y_bottom)

        # Convert to distance along strike (x1) and dip (x2):
        x1 = xx*np.sin(ang_strike) + yy*np.cos(ang_strike)
        x2 = xx*np.cos(ang_strike) - yy*np.sin(ang_strike)

        # In Okada's paper, x2 is distance up the fault plane, not down dip:
        x2 = -x2

        p = x2*np.cos(ang_dip) + depth_bottom*np.sin(ang_dip)
        q = x2*np.sin(ang_dip) - depth_bottom*np.cos(ang_dip)

        f1 = self._strike_slip(x1 + halfL, p, ang_dip, q)
        f2 = self._strike_slip(x1 + halfL, p - w, ang_dip, q)
        f3 = self._strike_slip(x1 - halfL, p, ang_dip, q)
        f4 = self._strike_slip(x1 - halfL, p - w, ang_dip, q)

        g1 = self._dip_slip(x1 + halfL, p, ang_dip, q)
        g2 = self._dip_slip(x1 + halfL, p - w, ang_dip, q)
        g3 = self._dip_slip(x1 - halfL, p, ang_dip, q)
        g4 = self._dip_slip(x1 - halfL, p - w, ang_dip, q)

        # Displacement in direction of strike and dip:
        ds = slip * np.cos(ang_rake)
        dd = slip * np.sin(ang_rake)

        us = (f1 - f2 - f3 + f4) * ds
        ud = (g1 - g2 - g3 + g4) * dd

        dz = us + ud

        dtopo = DTopography()
        dtopo.X = X
        dtopo.Y = Y
        dtopo.dZ = np.array(dz, ndmin=3)
        dtopo.times = [0.0]
        self.dtopo = dtopo
