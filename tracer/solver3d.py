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

        # Form metric  # TODO: use pyop2
        self.M = Function(self.P1_ten)
        for i in range(self.mesh.num_vertices()):
            self.M.dat.data[i][:,:] += H1.dat.data[i]*adj_diff.dat.data[i][0]
            self.M.dat.data[i][:,:] += H2.dat.data[i]*adj_diff.dat.data[i][1]
            self.M.dat.data[i][:,:] += H3.dat.data[i]*adj_diff.dat.data[i][2]
            if relax:
                self.M.dat.data[i][:,:] += Hf.dat.data[i]*adj.dat.data[i]
        self.M = steady_metric(None, H=self.M, op=self.op)

        if superpose:
            Mf = Function(self.P1_ten)
            for i in range(self.mesh.num_vertices()):
                Mf.dat.data[i][:,:] += Hf.dat.data[i]*adj.dat.data[i]
            self.M = metric_intersection(self.M, Mf)

        # TODO: boundary contributions
