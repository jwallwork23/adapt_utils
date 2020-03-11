from adapt_utils.test_cases.bubble_shear.options import BubbleOptions
# from adapt_utils.tracer.solver2d_thetis import UnsteadyTracerProblem2d_Thetis
from adapt_utils.tracer.solver2d import UnsteadyTracerProblem2d


op = BubbleOptions(approach='fixed_mesh', family='DG')  # FIXME: CG not working
# tp = UnsteadyTracerProblem2d_Thetis(op)
tp = UnsteadyTracerProblem2d(op)
tp.setup_solver_forward()
tp.solve()
