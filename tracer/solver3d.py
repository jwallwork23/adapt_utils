from firedrake import *

import numpy as np

from adapt_utils.tracer.options import *
from adapt_utils.tracer.solver2d import SteadyTracerProblem2d
from adapt_utils.tracer.stabilisation import supg_coefficient, anisotropic_stabilisation
from adapt_utils.adapt.adaptation import *
from adapt_utils.adapt.metric import *
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
    def get_loseille_metric(self, adjoint=False, relax=True, superpose=False):
        assert not (relax and superpose)

        # Solve adjoint problem
        if self.op.order_increase:
            adj = self.solve_high_order(adjoint=not adjoint)
        else:
            adj = self.solution if adjoint else self.adjoint_solution
        sol = self.adjoint_solution if adjoint else self.solution
        adj_diff = Function(self.P1_vec).interpolate(abs(construct_gradient(adj)))
        adj = Function(self.P1).interpolate(abs(adj))

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

        # Hessian for source term
        Mf = Function(self.P1_ten).assign(np.finfo(0.0).min)
        kernel = op2.Kernel(matscale_kernel(3),
                            "matscale",
                            cpp=True,
                            include_dirs=include_dir)
        op2.par_loop(kernel,
                     self.P1_ten.node_set,
                     Mf.dat(op2.RW),
                     Hf.dat(op2.READ),
                     adj.dat(op2.READ))

        # Form metric
        self.M = Function(self.P1_ten).assign(np.finfo(0.0).min)
        kernel = op2.Kernel(matscale_sum_kernel(3),
                            "matscale_sum",
                            cpp=True,
                            include_dirs=include_dir)
        op2.par_loop(kernel,
                     self.P1_ten.node_set,
                     self.M.dat(op2.RW),
                     H1.dat(op2.READ),
                     H2.dat(op2.READ),
                     H3.dat(op2.READ),
                     adj_diff.dat(op2.READ))
        if relax:
            self.M = metric_relaxation(self.M, Mf)
        elif superpose:
            self.M = metric_intersection(self.M, Mf)
        self.M = steady_metric(None, H=self.M, op=self.op)

        # TODO: boundary contributions
