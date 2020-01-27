from adapt_utils.test_cases.bubble_shear.options import BubbleOptions
from adapt_utils.tracer.solver2d_thetis import UnsteadyTracerProblem2d_Thetis


op = BubbleOptions(approach='fixed_mesh')
tp = UnsteadyTracerProblem2d_Thetis(op)
