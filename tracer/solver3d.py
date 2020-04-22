from firedrake import *

from adapt_utils.tracer.solver2d import SteadyTracerProblem2d
from adapt_utils.adapt.metric import steady_metric, combine_metrics
from adapt_utils.adapt.kernels import eigen_kernel, matscale
from adapt_utils.adapt.recovery import construct_gradient


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
        adj_diff = interpolate(abs(construct_gradient(adj)), self.P1_vec)
        adj_diff.rename("Gradient of adjoint solution")
        adj_diff_x = interpolate(adj_diff[0], self.P1)
        adj_diff_x.rename("x-derivative of adjoint solution")
        adj_diff_y = interpolate(adj_diff[1], self.P1)
        adj_diff_y.rename("y-derivative of adjoint solution")
        adj_diff_z = interpolate(adj_diff[2], self.P1)
        adj_diff_z.rename("z-derivative of adjoint solution")
        adj = interpolate(abs(adj), self.P1)
        adj.rename("Adjoint solution in modulus")

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
        H1.rename("Hessian for x-component")
        H2 = steady_metric(F2, mesh=self.mesh, noscale=True, op=self.op)
        H1.rename("Hessian for y-component")
        H3 = steady_metric(F3, mesh=self.mesh, noscale=True, op=self.op)
        H1.rename("Hessian for z-component")
        Hf = steady_metric(source, mesh=self.mesh, noscale=True, op=self.op)
        H1.rename("Hessian for source term")

        # Hessians for conservative parts
        M1 = Function(self.P1_ten).assign(0.0)
        M2 = Function(self.P1_ten).assign(0.0)
        M3 = Function(self.P1_ten).assign(0.0)
        kernel = eigen_kernel(matscale, 3)
        op2.par_loop(kernel, self.P1_ten.node_set, M1.dat(op2.RW), H1.dat(op2.READ), adj_diff_x.dat(op2.READ))
        M1 = steady_metric(None, H=M1, op=self.op)
        M1.rename("Metric for x-component of conservative terms")
        op2.par_loop(kernel, self.P1_ten.node_set, M2.dat(op2.RW), H2.dat(op2.READ), adj_diff_y.dat(op2.READ))
        M2 = steady_metric(None, H=M2, op=self.op)
        M2.rename("Metric for y-component of conservative terms")
        op2.par_loop(kernel, self.P1_ten.node_set, M3.dat(op2.RW), H3.dat(op2.READ), adj_diff_z.dat(op2.READ))
        M3 = steady_metric(None, H=M3, op=self.op)
        M3.rename("Metric for z-component of conservative terms")
        M = combine_metrics(M1, M2, average=relax)
        M = combine_metrics(M, M3, average=relax)

        # Hessian for source term
        Mf = Function(self.P1_ten).assign(0.0)
        op2.par_loop(kernel, self.P1_ten.node_set, Mf.dat(op2.RW), Hf.dat(op2.READ), adj.dat(op2.READ))
        Mf = steady_metric(None, H=Mf, op=self.op)
        Mf.rename("Metric for source term")

        # Combine contributions
        self.M = combine_metrics(M, Mf, average=relax)
        self.M.rename("Loseille metric")

        # Account for boundary contributions TODO: Use EquationBC
