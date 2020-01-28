from thetis import *

import matplotlib.pyplot as plt

from adapt_utils.tsunami.options import TohokuOptions
from adapt_utils.tsunami.solver import TsunamiProblem

op = TohokuOptions(utm=True)
swp = TsunamiProblem(op, levels=0)

# TODO
