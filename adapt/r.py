from firedrake import *
from thetis import print_output

import os
import numpy as np

from adapt_utils.options import Options


__all__ = ["MeshMover"]


class MeshMover():
    """
    A class dedicated to performing mesh r-adaptation. Given a source mesh and a monitor function
    :math: `m`, the new mesh is established by relocating the vertices of the original mesh. That is,
    the topology remains unchanged.

    At present, the mesh movement is determined by solving the Monge-Ampère type equation

..  math::
        m(x)\det(I + H(\phi)) = \theta,

    for a scalar potential :math:`\phi`, where :math:`I` is the identity, :math:`\theta` is a
    normalising constant and :math:`H(\phi)` denotes the Hessian of :math:`\phi` with respect to
    coordinates on the computational mesh.

    The implementation is an objective-oriented version of that given in [1].

    [1] A.T.T. McRae, C.J. Cotter, and C.J. Budd, "Optimal-transport--based mesh adaptivity on the
        plane and sphere using finite elements." SIAM Journal on Scientific Computing 40.2 (2018):
        A1121-A1148.
    """
    def __init__(self, mesh, monitor_function, op=Options()):
        self.mesh = mesh
        self.dim = self.mesh.topological_dimension()
        try:
            assert self.dim == 2
        except AssertionError:
            raise NotImplementedError("r-adaptation only currently considered in 2D.")
        self.monitor_function = monitor_function
        self.op = op
        self.ξ = Function(self.mesh.coordinates)  # Computational coordinates
        self.x = Function(self.mesh.coordinates)  # Physical coordinates
        self.I = Identity(self.dim)
        self.dt = Constant(op.pseudo_dt)

        # Create functions and solvers
        self.create_function_spaces()
        self.create_functions()
        self.setup_pseudotimestepper()
        self.setup_equidistributor()
        self.setup_l2_projector()

        # Outputs
        self.monitor_file = File(os.path.join(op.di, 'monitor.pvd'))
        self.monitor_file.write(self.monitor)
        self.volume_file = File(os.path.join(op.di, 'volume.pvd'))
        self.volume_file.write(self.volume)

    def create_function_spaces(self):
        self.V = FunctionSpace(self.mesh, "CG", self.op.degree)
        self.V_ten = TensorFunctionSpace(self.mesh, "CG", self.op.degree)
        self.P1 = FunctionSpace(self.mesh, "CG", 1)
        self.P1_vec = VectorFunctionSpace(self.mesh, "CG", 1)
        self.P0 = FunctionSpace(self.mesh, "DG", 0)
        self.p0test = TestFunction(self.P0)

    def create_functions(self):
        self.φ_old = Function(self.V)
        self.φ_new = Function(self.V)
        self.σ_old = Function(self.V_ten)
        self.σ_new = Function(self.V_ten)
        self.θ = Constant(0.0)  # Normalisation parameter
        self.monitor = Function(self.P1, name="Monitor function")
        self.update_monitor()
        self.volume = Function(self.P0, name="Mesh volume").assign(assemble(self.p0test*dx))
        self.grad_φ_cts = Function(self.P1_vec, name="L2 projected gradient")
        self.grad_φ_dg = Function(self.mesh.coordinates, name="Discontinuous gradient")

    def update_monitor(self):
        self.monitor.interpolate(self.monitor_function(self.mesh))

    def setup_pseudotimestepper(self):
        φ, ψ = TrialFunction(self.V), TestFunction(self.V)
        a = dot(grad(ψ), grad(φ))*dx
        L = dot(grad(ψ), grad(self.φ_old))*dx
        L += self.dt*ψ*(self.monitor*det(self.I + self.σ_old) - self.θ)*dx
        prob = LinearVariationalProblem(a, L, self.φ_new)
        nullspace = VectorSpaceBasis(constant=True)
        self.pseudotimestepper = LinearVariationalSolver(prob, nullspace=nullspace,
                                                         transpose_nullspace=nullspace,
                                                         solver_parameters={'ksp_type': 'cg',
                                                                            'pc_type': 'gamg'})

        # Setup residuals
        self.θ_form = self.monitor*det(self.I + self.σ_old)*dx
        residual_form = self.monitor*det(self.I + self.σ_old) - self.θ
        self.residual_l2_form = ψ*residual_form*dx
        self.norm_l2_form = ψ*self.θ*dx

    def setup_equidistributor(self):
        σ, τ = TrialFunction(self.V_ten), TestFunction(self.V_ten)
        n = FacetNormal(self.mesh)
        a = inner(τ, σ)*dx
        L = -dot(div(τ), grad(self.φ_new))*dx
        # FIXME: Neumann condition doesn't seem to work!
        L += (τ[0, 1]*n[1]*self.φ_new.dx(0) + τ[1, 0]*n[0]*self.φ_new.dx(1))*ds
        L += (-τ[0, 0]*n[1]*self.φ_new.dx(1) + τ[1, 1]*n[1]*self.φ_new.dx(1))*ds
        prob = LinearVariationalProblem(a, L, self.σ_new)
        self.equidistributor = LinearVariationalSolver(prob, solver_parameters={'ksp_type': 'cg'})

    def setup_l2_projector(self):
        u_cts, v_cts = TrialFunction(self.P1_vec), TestFunction(self.P1_vec)
        a = dot(v_cts, u_cts)*dx
        L = dot(v_cts, grad(self.φ_old))*dx
        prob = LinearVariationalProblem(a, L, self.grad_φ_cts)
        self.l2_projector = LinearVariationalSolver(prob, solver_parameters={'ksp_type': 'cg'})

    def adapt(self):
        L_p0 = self.p0test*self.monitor*dx
        original_volume = assemble(self.p0test*dx)
        total_volume = assemble(Constant(1.0)*dx(domain=self.mesh))

        maxit = self.op.r_adapt_maxit
        tol = self.op.r_adapt_rtol
        msg = "Iteration {:4d}   Min/Max {:10.4e}   Residual {:10.4e}   Equidistribution {:10.4e}"
        for i in range(maxit):

            # Perform L2 projection and generate coordinates appropriately
            self.l2_projector.solve()
            par_loop(('{[i, j] : 0 <= i < cg.dofs and 0 <= j < 2}', 'dg[i, j] = cg[i, j]'), dx,
                     {'cg': (self.grad_φ_cts, READ), 'dg': (self.grad_φ_dg, WRITE)},
                     is_loopy_kernel=True)
            self.x.assign(self.ξ + self.grad_φ_dg)  # x = ξ + grad(φ)

            # Update monitor function
            self.mesh.coordinates.assign(self.x)
            self.update_monitor()
            assemble(L_p0, tensor=self.volume)  # For equidistribution measure
            self.volume /= original_volume
            if i % 10 == 0:
                self.monitor_file.write(self.monitor)
                self.volume_file.write(self.volume)
            self.mesh.coordinates.assign(self.ξ)

            # Evaluate normalisation coefficient
            self.θ.assign(assemble(self.θ_form)/total_volume)

            # Convergence criteria
            residual_l2 = assemble(self.residual_l2_form).dat.norm
            norm_l2 = assemble(self.norm_l2_form).dat.norm
            residual_l2_norm = residual_l2 / norm_l2
            if i == 0:
                initial_norm = residual_l2_norm  # Store to check for divergence
            minmax = self.volume.vector().gather().min()/self.volume.vector().gather().max()
            equi = np.std(self.volume.dat.data)/np.mean(self.volume.dat.data)  # TODO: PyOP2
            if i % 10 == 0 and self.op.debug:
                print_output(msg.format(i, minmax, residual_l2_norm, equi))
            if residual_l2_norm < tol:
                print_output("r-adaptation converged in {:d} iterations.".format(i+1))
                break
            if residual_l2_norm > 2.0*initial_norm:
                raise ConvergenceError("r-adaptation failed to converge in {:d} iterations.".format(i+1))
            self.pseudotimestepper.solve()
            self.equidistributor.solve()
            self.φ_old.assign(self.φ_new)
            self.σ_old.assign(self.σ_new)
