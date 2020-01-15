# Base classes
from adapt_utils.options import *
from adapt_utils.solver import *

# Mesh adaptation
from adapt_utils.adapt.adaptation import *
from adapt_utils.adapt.metric import *
from adapt_utils.adapt.p0_metric import *
from adapt_utils.adapt.recovery import *

from thetis import print_output


__all__ = ["doc"]


def doc(anything):
    """
    Print the docstring of any class or function.
    """
    print_output(anything.__doc__)
