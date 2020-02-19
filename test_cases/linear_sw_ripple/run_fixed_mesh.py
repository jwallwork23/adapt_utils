from adapt_utils.swe.spacetime.solver import *
from adapt_utils.test_cases.linear_sw_ripple.options import RippleOptions


debug = False
dispersive = True

op = RippleOptions(debug=debug)
solver = SpaceTimeDispersiveShallowWaterProblem if dispersive else SpaceTimeShallowWaterProblem
swp = solver(op)
swp.setup_solver_forward()
swp.solve_forward()
