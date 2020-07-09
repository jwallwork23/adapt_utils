r"""
Test mesh movement and metric based adaptation in the steady state case for analytically defined sensor
functions.

Sensors as defined in

Olivier, Géraldine. Anisotropic metric-based mesh adaptation for unsteady CFD simulations involving moving geometries. Diss. 2011.
"""
import pytest
import os
# import matplotlib.pyplot as plt

from adapt_utils import *
from adapt_utils.test.adapt.sensors import *


# ---------------------------
# standard tests for pytest
# ---------------------------

@pytest.fixture(params=[bowl, hyperbolic, multiscale, interweaved])
def sensor(request):
    return request.param


@pytest.fixture(params=['complexity', 'error'])
def normalisation(request):
    return request.param


@pytest.fixture(params=[1, 2, 'inf'])
def norm_order(request):
    return request.param


def test_metric_based(sensor, normalisation, norm_order, plot_mesh=False):
    if os.environ.get('FIREDRAKE_ADAPT') == '0':
        pytest.xfail("Firedrake installation does not include Pragmatic")

    kwargs = {
        'h_min': 1.0e-06,
        'h_max': 1.0e-01,
        'num_adapt': 4,
        'normalisation': normalisation,
        'norm_order': norm_order,
    }
    op = Options(**kwargs)  # NOQA  TODO


# ---------------------------
# mesh plotting
# ---------------------------

if __name__ == '__main__':
    for f in [bowl, hyperbolic, multiscale, interweaved]:
        test_metric_based(f, 'complexity', 1, plot_mesh=True)
