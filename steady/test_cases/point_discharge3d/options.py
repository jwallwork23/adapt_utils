from firedrake import *

import numpy as np

from adapt_utils.steady.test_cases.point_discharge2d.options import PointDischarge2dOptions


__all__ = ["PointDischarge3dOptions"]


# TODO: Analytical solution
# TODO: Calibrate radius

class PointDischarge3dOptions(PointDischarge2dOptions):
    r"""
    Parameters for a 3D extension of the 'Point source with diffusion' test case from TELEMAC-2D
    validation document version 7.0.

    We consider a quantity of interest (QoI) :math:`J` of the form

..  math:: J(\phi) = \int_A \phi \;\mathrm{d}x,

    where :math:`A` is a spherical 'receiver' region.

    :kwarg approach: Mesh adaptation strategy,
    :kwarg offset: Shift in x-direction for source location.
    :kwarg centred: Toggle whether receiver is positioned in the centre of the flow or not.
    """
    def __init__(self, level=0, aligned=True, **kwargs):
        super(PointDischarge3dOptions, self).__init__(aligned=aligned, **kwargs)
        self.default_mesh = BoxMesh(100*2**level, 20*2**level, 20*2**level, 50, 10, 10)
        self.region_of_interest = [(20.0, 5.0, 5.0, 0.5)] if aligned else [(20.0, 7.5, 7.5, 0.5)]
        self.base_velocity = [1.0, 0.0, 0.0]
        # TODO: h_min, h_max

    def set_boundary_conditions(self, prob, i):
        zero = Constant(0.0)
        boundary_conditions = {
            'tracer': {
                1: {'value': zero},
                2: {},
                3: {'diff_flux': zero},
                4: {'diff_flux': zero},
                5: {'diff_flux': zero},
                6: {'diff_flux': zero},
            }
        }
        return boundary_conditions

    def set_calibrated_radius(self):  # TODO: Rerun for 3D case
        loc = super(PointDischarge3dOptions, self).set_calibrated_radius()[0]
        return [(1.0 + self.shift, 5.0, 5.0, loc[-1])]

    def analytical_solution(self, fs):
        raise NotImplementedError

    def analytical_qoi(self, mesh=None):
        raise NotImplementedError
