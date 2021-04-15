from __future__ import absolute_import

<<<<<<< HEAD
from .misc import *  # NOQA
from .norms import *  # NOQA

# Base classes
from .options import *  # NOQA

# Mesh adaptation
from .adapt.adaptation import *  # NOQA
from .adapt.metric import *  # NOQA
from .adapt.p0_metric import *  # NOQA
=======
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
>>>>>>> dfe1c0b3a34dfef1765835b64b574a69fe60dd9a
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
