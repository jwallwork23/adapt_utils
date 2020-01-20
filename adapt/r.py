from firedrake import *

import os

from adapt_utils.options import Options


__all__ = []


class MeshMover():
    # TODO: doc  x:Ω_C→ Ω_P
    def __init__(self, mesh, monitor_function, op=Options()):
        self.mesh = mesh
        self.dim = self.mesh.topological_dimension()
        try:
            assert self.dim == 2:
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
        self.setup_scalar_solver()
        self.setup_tensor_solver()
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
        self.phi_old = Function(self.V)
        self.phi_new = Function(self.V)
        self.sigma_old = Function(self.V_ten)
        self.sigma_new = Function(self.V_ten)
        self.theta = Constant(0.0)  # Normalisation parameter
        self.monitor = Function(self.P1, name="Monitor function")
        self.update_monitor()
        self.volume = Function(self.P0, name="Mesh volume").assign(assemble(self.p0test*dx))
        self.grad_phi_cts = Function(self.P1_vec, name="L2 projected gradient")
        self.grad_phi_dg = Function(self.mesh.coordinates, name="Discontinuous gradient")
        self.total_volume = assemble(Constant(1.0)*dx(domain=self.mesh))

    def update_monitor(self):
        self.monitor.interpolate(self.monitor_function(self.mesh))

    def setup_scalar_solver(self):
        phi, v = TrialFunction(self.V), TestFunction(self.V)
        a = dot(grad(v), grad(phi))*dx
        L = dot(grad(v), grad(self.phi_old))*dx
        L += self.dt*v*(self.monitor*det(self.I + self.sigma_old) - self.theta)*dx
        prob = LinearVariationalProblem(a, L, self.phi_new)
        nullspace = VectorSpaceBasis(constant=True)
        self.scalar_solver = LinearVariationalSolver(prob, nullspace=nullspace,
                                                     transpose_nullspace=nullspace,
                                                     solver_parameters={'ksp_type': 'cg',
                                                                        'pc_type': 'gamg'})

        # Setup residuals
        self.theta_form = self.monitor*det(self.I + self.sigma_old)*dx
        residual_form = self.monitor*det(self.I + self.sigma_old) - self.theta
        self.residual_l2_form = v*residual_form*dx
        self.norm_l2_form = v*self.theta*dx

    def setup_tensor_solver(self):
        sigma, tau = TrialFunction(self.V_ten), TestFunction(self.V_ten)
        n = FacetNormal(mesh)
        a = inner(tau, sigma)*dx
        L = -dot(div(tau), grad(self.phi_new))*dx
        # FIXME: Neumann condition doesn't seem to work!
        L += (tau[0, 1]*n[1]*self.phi_new.dx(0) + tau[1, 0]*n[0]*self.phi_new.dx(1))*ds
        L += (-tau[0, 0]*n[1]*self.phi_new.dx(1) + tau[1, 1]*n[1]*self.phi_new.dx(1))*ds
        prob = LinearVariationalProblem(a, L, self.sigma_new)
        self.tensor_solver = LinearVariationalSolver(prob, solver_parameters={'ksp_type': 'cg'})

    def setup_l2_projector(self):
        u_cts, v_cts = TrialFunction(self.P1_vec), TestFunction(self.P1_vec)
        a = dot(v_cts, u_cts)*dx
        L = dot(v_cts, grad(self.phi_old))*dx
        prob = LinearVariationalProblem(a, L, self.grad_phi_cts)
        self.l2_projector = LinearVariationalSolver(prob, solver_parameters={'ksp_type': 'cg'})

    def adapt(self):
        maxit = self.op.r_adapt_maxit
        tol = self.op.r_adapt_rtol
        
        raise NotImplementedError  # TODO
