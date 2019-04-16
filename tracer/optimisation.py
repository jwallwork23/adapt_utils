from firedrake import *
from firedrake_adjoint import *
import numpy as np
import matplotlib.pyplot as plt

from adapt_utils.tracer.options import *
from adapt_utils.tracer.solver_cg import *
from adapt_utils.solver import *


__all__ = ["TelemacLoop"]


class TelemacLoop(OuterLoop):
    def __init__(self, centred=False, **kwargs):
        super(TelemacLoop, self).__init__(SteadyTracerProblem_CG,
                                          op=TelemacOptions_Centred() if centred else TelemacOptions(),
                                          **kwargs)

    def compare_analytical_objective(self, finite_element=FiniteElement('Lagrange', triangle, 1)):
        mesh = self.final_mesh
        fs = FunctionSpace(mesh, finite_element)
        J_exact = self.op.exact_objective(fs)
        err = np.abs(J_exact - self.final_J)
        r_err = err / np.abs(J_exact)
        logfile = open(self.di + 'scale_to_convergence.log', 'a+')
        logfile.write('on this mesh: exact {:.4e} abs {:.4e} rel {:.4e}'.format(J_exact, err, r_err))
        print("Exact solution on this mesh: {:.4e}".format(J_exact))
        logfile.close()

    def compare_slices(self):  # TODO: exact sol
        eps = 1e-5

        approx = []
        X = np.linspace(eps, 50-eps, 100)
        for x in X:
            approx.append(self.op.solution.at([x, 5]))
        plt.plot(X, approx, label='Numerical')
        plt.title('Adaptive approach: {:s}'.format(self.approach))
        plt.xlabel('x-coordinate [m]')
        plt.ylabel('Tracer concentration [g/L]')
        plt.ylim([0, 2.5])
        plt.legend()
        plt.savefig(self.di + 'xslice.pdf')
        plt.clf()

        approx = []
        Y = np.linspace(eps, 10-eps, 20)
        for y in Y:
            approx.append(self.op.solution.at([30, y]))
        plt.plot(Y, approx, label='Numerical')
        plt.title('Adaptive approach: {:s}'.format(self.approach))
        plt.xlabel('y-coordinate [m]')
        plt.ylabel('Tracer concentration [g/L]')
        plt.ylim([0, 0.35])
        plt.legend()
        plt.savefig(self.di + 'yslice.pdf')
