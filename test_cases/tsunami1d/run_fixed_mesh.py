from adapt_utils.swe.spacetime.solver import SpaceTimeShallowWaterProblem
from adapt_utils.test_cases.tsunami1d.options import Tsunami1dOptions


debug = True

op = Tsunami1dOptions(debug=debug)
swp = SpaceTimeShallowWaterProblem(op)
swp.setup_solver_forward()
swp.solve_forward()
