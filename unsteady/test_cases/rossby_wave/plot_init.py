from adapt_utils.test_cases.rossby_wave.options import BoydOptions
from adapt_utils.swe.solver import UnsteadyShallowWaterProblem

# NOTE: It seems as though [Huang et al 2008] considers n = 4, 8, 20
n = 20
op = BoydOptions(n=n, order=1)
sw = UnsteadyShallowWaterProblem(op)
