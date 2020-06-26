from adapt_utils.steady.swe.spacetime.solver import *
from adapt_utils.test_cases.spacetime_ripple.options import RippleOptions


debug = False
dispersive = True

op = RippleOptions(debug=debug)
solver = SpaceTimeDispersiveShallowWaterProblem if dispersive else SpaceTimeShallowWaterProblem
swp = solver(op)
swp.setup_solver_forward()
swp.solve_forward()
