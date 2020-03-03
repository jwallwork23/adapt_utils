from firedrake import *

from adapt_utils.test_cases.ale_advection.options import ALEAdvectionOptions
from adapt_utils.tracer.solver2d import UnsteadyTracerProblem2d
from adapt_utils.adapt.r import MeshMover


op = ALEAdvectionOptions(num_adapt=1, nonlinear_method='relaxation', prescribed_velocity='fluid')
tp = UnsteadyTracerProblem2d(op)
tp.solve_ale()
