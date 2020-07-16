from __future__ import absolute_import

from .misc import *  # NOQA
from .norms import *  # NOQA

# Base classes
from .options import *  # NOQA

# Mesh adaptation
from .adapt.adaptation import *  # NOQA
from .adapt.metric import *  # NOQA
from .adapt.p0_metric import *  # NOQA
from .adapt.r import *  # NOQA
from .adapt.recovery import *  # NOQA

# Thetis
from thetis import *  # NOQA


# Check whether Firedrake has Pragmatic installed
try:
    from firedrake import adapt
    os.environ['FIREDRAKE_ADAPT'] = '1'
except ImportError:
    os.environ['FIREDRAKE_ADAPT'] = '0'
