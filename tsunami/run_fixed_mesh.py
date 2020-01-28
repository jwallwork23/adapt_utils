from thetis import *

from adapt_utils.tsunami.options import TohokuOptions
from adapt_utils.tsunami.solver import TsunamiProblem

op = TohokuOptions(utm=True)
op.plot_pvd = True
swp = TsunamiProblem(op, levels=0)
swp.solve()

# TODO: Plot gauge measurements
