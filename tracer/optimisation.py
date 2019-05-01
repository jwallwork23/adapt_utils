from firedrake import *
from firedrake_adjoint import *
import numpy as np
import matplotlib.pyplot as plt

from adapt_utils.tracer.options import *
from adapt_utils.tracer.solver_cg import *
from adapt_utils.solver import *


__all__ = ["TelemacLoop"]


# TODO: Generalise to allow unsteady DG version
class TelemacLoop(OuterLoop):
    # TODO: doc
    def __init__(self, op, mesh=None):
        super(TelemacLoop, self).__init__(SteadyTracerProblem_CG, op, mesh)

    def compare_analytical_objective(self, finite_element=FiniteElement('Lagrange', triangle, 1)):
        mesh = self.final_mesh
        P0 = FunctionSpace(mesh, "DG", 0)
        fs = FunctionSpace(mesh, finite_element)
        region = self.op.region_of_interest[0]
        self.op.region_of_interest = [(region[0], region[1], 0.01*region[2])]
        J_exact = self.op.exact_objective(fs, P0)
        err = np.abs(J_exact - self.final_J)
        r_err = err / np.abs(J_exact)
        logfile = open(self.di + 'desired_error_test.log', 'a+')  # FIXME: this is not the only mode
        logfile.write('on this mesh: exact {:.4e} abs {:.4e} rel {:.4e}'.format(J_exact, err, r_err))
        print("Exact solution on this mesh: {:.4e}".format(J_exact))
        logfile.close()

    def compare_slices(self):  # FIXME: these plots do not agree with J analysis
        eps = 1e-5

        # Plot slice in x-direction along y=5
        approx = []
        analytic = []
        X = np.linspace(eps, 50-eps, 100)
        for x in X:
            approx.append(self.solution.at([x, 5]))
            analytic.append(self.op.solution.at([x, 5]))
        plt.plot(X, approx, label='Numerical')
        plt.plot(X, analytic, label='Analytic')
        plt.title('Adaptive approach: {:s}'.format(self.op.approach))
        plt.xlabel('x-coordinate [m]')
        plt.ylabel('Tracer concentration [g/L]')
        plt.ylim([0, 2.5])
        plt.legend()
        plt.savefig(self.di + 'xslice.pdf')
        plt.clf()

        # Plot slice in y-direction along x=20  NOTE: originally x=30 was used
        approx = []
        analytic = []
        Y = np.linspace(eps, 10-eps, 20)
        for y in Y:
            approx.append(self.solution.at([20, y]))
            analytic.append(self.op.solution.at([20, y]))
        plt.plot(Y, approx, label='Numerical')
        plt.plot(Y, analytic, label='Analytic')
        plt.title('Adaptive approach: {:s}'.format(self.op.approach))
        plt.xlabel('y-coordinate [m]')
        plt.ylabel('Tracer concentration [g/L]')
        plt.ylim([0, 0.35])
        plt.legend()
        plt.savefig(self.di + 'yslice.pdf')
