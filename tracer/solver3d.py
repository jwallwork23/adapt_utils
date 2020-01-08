from firedrake import *

import numpy as np

from adapt_utils.tracer.options import *
from adapt_utils.tracer.solver2d import SteadyTracerProblem2d
from adapt_utils.tracer.stabilisation import supg_coefficient, anisotropic_stabilisation
from adapt_utils.adapt.adaptation import *
from adapt_utils.adapt.metric import *
from adapt_utils.adapt.kernels import eigen_kernel, matscale, matscale_sum
from adapt_utils.adapt.recovery import *
from adapt_utils.adapt.p0_metric import *


__all__ = ["SteadyTracerProblem3d"]


class SteadyTracerProblem3d(SteadyTracerProblem2d):
    r"""
    General continuous Galerkin solver object for 3D stationary tracer advection problems of the form

..  math::
    \textbf{u} \cdot \nabla(\phi) - \nabla \cdot (\nu \cdot \nabla(\phi)) = f,

    for (prescribed) velocity :math:`\textbf{u}`, diffusivity :math:`\nu \geq 0`, source :math:`f`
    and (prognostic) concentration :math:`\phi`.

    Implemented boundary conditions:
        * Neumann zero;
        * Dirichlet zero;
        * outflow.
    """
    def get_loseille_metric(self, adjoint=False, relax=True):
        adj = self.get_solution(not adjoint)
        sol = self.get_solution(adjoint)
        adj_diff = interpolate(abs(construct_gradient(adj)), self.P1_vec))
        adj = interpolate(abs(adj), self.P1)

        if adjoint:
            source = self.kernel
            F1 = -sol*self.u[0] - self.nu*sol.dx(0)
            F2 = -sol*self.u[1] - self.nu*sol.dx(1)
            F3 = -sol*self.u[2] - self.nu*sol.dx(2)
        else:
            source = self.source
            F1 = sol*self.u[0] - self.nu*sol.dx(0)
            F2 = sol*self.u[1] - self.nu*sol.dx(1)
            F3 = sol*self.u[2] - self.nu*sol.dx(2)

        # Construct Hessians
        H1 = steady_metric(F1, mesh=self.mesh, noscale=True, op=self.op)
        H2 = steady_metric(F2, mesh=self.mesh, noscale=True, op=self.op)
        H3 = steady_metric(F3, mesh=self.mesh, noscale=True, op=self.op)
        Hf = steady_metric(source, mesh=self.mesh, noscale=True, op=self.op)

        # Hessian for conservative part
        M = Function(self.P1_ten).assign(0.0)
        kernel = eigen_kernel(matscale_sum, 3)
        op2.par_loop(kernel,
                     self.P1_ten.node_set,
                     M.dat(op2.RW),
                     H1.dat(op2.READ),
                     H2.dat(op2.READ),
                     H3.dat(op2.READ),
                     adj_diff.dat(op2.READ))

        # Hessian for source term
        Mf = Function(self.P1_ten).assign(0.0)
        kernel = eigen_kernel(matscale, 3)
        op2.par_loop(kernel,
                     self.P1_ten.node_set,
                     Mf.dat(op2.RW),
                     Hf.dat(op2.READ),
                     adj.dat(op2.READ))

        # Combine contributions
        self.M = steady_metric(None, H=combine_metrics(M, Mf, average=relax), op=self.op)

        # Account for boundary contributions TODO: Use EquationBC
