from __future__ import absolute_import

# Generic imports
from .misc import *  # NOQA
from .norms import *  # NOQA
from .options import *  # NOQA
from .plotting import *  # NOQA

# Utils for coupled solvers
from .steady.solver import *  # NOQA
from .unsteady.solver import *  # NOQA

# Mesh adaptation
from .adapt.metric import *  # NOQA
from .adapt.r import *  # NOQA
from .adapt.recovery import *  # NOQA

# Thetis
from thetis import *  # NOQA


# Check whether Firedrake has Pragmatic installed
try:
    from firedrake import adapt  # NOQA
    os.environ['FIREDRAKE_ADAPT'] = '1'
except ImportError:
    os.environ['FIREDRAKE_ADAPT'] = '0'
