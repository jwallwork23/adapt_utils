from adapt_utils.swe.spacetime.solver import SpaceTimeShallowWaterProblem
from adapt_utils.test_cases.linear_sw_ripple.options import RippleOptions


debug = False
op = RippleOptions(debug=debug, nx=32, nt=25, shelf=True)
swp = SpaceTimeShallowWaterProblem(op)
swp.setup_solver_forward()
swp.solve_forward()
swp.plot_solution()
