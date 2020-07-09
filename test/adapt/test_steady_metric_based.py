import os
import pytest

from adapt_utils import *  # NOQA  TODO


# ---------------------------
# standard tests for pytest
# ---------------------------

@pytest.fixture(params=[2, 3])
def dim(request):
    return request.param


def test_metric_based(dim):
    if os.environ.get('FIREDRAKE_ADAPT') == '0':
        pytest.xfail("Firedrake installation does not include Pragmatic")
    # TODO: move over adapt/test/refine2d and adapt/test/refine3d
