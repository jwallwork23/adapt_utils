from firedrake import *
from firedrake_adjoint import *
import numpy as np

from adapt_utils.tracer.options import *
from adapt_utils.tracer.solver_cg import *
from adapt_utils.solver import *


__all__ = ["TelemacLoop"]


class TelemacLoop(OuterLoop):
    def __init__(self, centred=False, **kwargs):
        super(TelemacLoop, self).__init__(SteadyTracerProblem_CG,
                                          op=TelemacOptions_Centred() if centred else TelemacOptions(),
                                          **kwargs)

    def compare_exact_solution(self, finite_element=FiniteElement('Lagrange', triangle, 1)):
        mesh = self.final_mesh
        fs = FunctionSpace(mesh, finite_element)
        J_exact = self.op.exact_objective(fs)
        err = np.abs(J_exact - self.final_J)
        r_err = err / np.abs(J_exact)
        logfile = open(self.di + 'scale_to_convergence.log', 'a+')
        logfile.write('on this mesh: exact {:.4e} abs {:.4e} rel {:.4e}'.format(J_exact, err, r_err))
        print("Exact solution on this mesh: {:.4e}".format(J_exact))
        logfile.close()
