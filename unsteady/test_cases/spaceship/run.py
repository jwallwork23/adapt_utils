from adapt_utils.unsteady.swe.turbine.solver import AdaptiveTurbineProblem
from adapt_utils.unsteady.test_cases.spaceship.options import SpaceshipOptions


kwargs = {
    'stabilisation': 'lax_friedrichs',
    # 'stabilisation': None,
    'family': 'dg-cg',
}

op = SpaceshipOptions()
op.update(kwargs)
tp = AdaptiveTurbineProblem(op)
tp.solve_forward()
