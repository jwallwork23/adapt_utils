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

    The implementation is an objective-oriented version of that given in [1]. Most of the code
    presented here is copied directly from that associated with [1].

    [1] A.T.T. McRae, C.J. Cotter, and C.J. Budd, "Optimal-transport--based mesh adaptivity on the
        plane and sphere using finite elements." SIAM Journal on Scientific Computing 40.2 (2018):
        A1121-A1148.
    """
    def __init__(self, mesh, monitor_function, method='quasi_newton', op=Options()):
        self.mesh = mesh
        self.dim = self.mesh.topological_dimension()
        try:
            assert self.dim == 2
        except AssertionError:
            raise NotImplementedError("r-adaptation only currently considered in 2D.")
        self.monitor_function = monitor_function
        assert method in ('quasi_newton', 'relaxation')
        self.method = method
        self.op = op
        self.ξ = Function(self.mesh.coordinates)  # Computational coordinates
        self.x = Function(self.mesh.coordinates)  # Physical coordinates
        self.I = Identity(self.dim)
        self.dt = Constant(op.pseudo_dt)

        # Create functions and solvers
        self.create_function_spaces()
        self.create_functions()
        self.setup_equidistribution()
        self.setup_l2_projector()
        self.initialise_sigma()

        # Outputs
        if self.op.debug:
            self.monitor_file = File(os.path.join(op.di, 'monitor_debug.pvd'))
            self.monitor_file.write(self.monitor)
            self.volume_file = File(os.path.join(op.di, 'volume_debug.pvd'))
            self.volume_file.write(self.volume)
        self.msg = "{:4d}   Min/Max {:10.4e}   Residual {:10.4e}   Equidistribution {:10.4e}"

    def create_function_spaces(self):
        self.V = FunctionSpace(self.mesh, "CG", self.op.degree)
        self.V_nullspace = VectorSpaceBasis(constant=True)
        self.V_ten = TensorFunctionSpace(self.mesh, "CG", self.op.degree)
        self.W = self.V*self.V_ten
        self.P1 = FunctionSpace(self.mesh, "CG", 1)
        self.P1_vec = VectorFunctionSpace(self.mesh, "CG", 1)
        self.P0 = FunctionSpace(self.mesh, "DG", 0)
        self.p0test = TestFunction(self.P0)

    def create_functions(self):
        if self.method == 'relaxation':
            self.φ_old = Function(self.V)
            self.φ_new = Function(self.V)
            self.σ_old = Function(self.V_ten)
            self.σ_new = Function(self.V_ten)
        else:
            self.φσ = Function(self.W)
            self.φ_new, self.σ_new = split(self.φσ)
            self.φσ_temp = Function(self.W)
            self.φ_old, self.σ_old = split(self.φσ_temp)
        self.θ = Constant(0.0)  # Normalisation parameter
        self.monitor = Function(self.P1, name="Monitor function")
        self.update_monitor()
        self.volume = Function(self.P0, name="Mesh volume").assign(assemble(self.p0test*dx))
        self.original_volume = assemble(self.p0test*dx)
        self.total_volume = assemble(Constant(1.0)*dx(domain=self.mesh))
        self.L_p0 = self.p0test*self.monitor*dx
        self.grad_φ_cts = Function(self.P1_vec, name="L2 projected gradient")
        self.grad_φ_dg = Function(self.mesh.coordinates, name="Discontinuous gradient")

    def update_monitor(self):
        self.monitor.interpolate(self.monitor_function(self.mesh))

    def setup_pseudotimestepper(self):
        assert self.method == 'relaxation'
        φ, ψ = TrialFunction(self.V), TestFunction(self.V)
        a = dot(grad(ψ), grad(φ))*dx
        L = dot(grad(ψ), grad(self.φ_old))*dx
        L += self.dt*ψ*(self.monitor*det(self.I + self.σ_old) - self.θ)*dx
        prob = LinearVariationalProblem(a, L, self.φ_new)
        self.pseudotimestepper = LinearVariationalSolver(prob, nullspace=self.V_nullspace,
                                                         transpose_nullspace=self.V_nullspace,
                                                         solver_parameters={'ksp_type': 'cg',
                                                                            'pc_type': 'gamg'})

    def setup_residuals(self):
        ψ = TestFunction(self.V)
        self.θ_form = self.monitor*det(self.I + self.σ_old)*dx
        residual_form = self.monitor*det(self.I + self.σ_old) - self.θ
        self.residual_l2_form = ψ*residual_form*dx
        self.norm_l2_form = ψ*self.θ*dx

    def apply_map(self):
        """
        Transfer L2 projected gradient from P1 to P1DG space and use it to perform the coordinate
        transform between computational and physical mesh.
        """
        par_loop(('{[i, j] : 0 <= i < cg.dofs and 0 <= j < 2}', 'dg[i, j] = cg[i, j]'), dx,
                 {'cg': (self.grad_φ_cts, READ), 'dg': (self.grad_φ_dg, WRITE)},
                 is_loopy_kernel=True)
        self.x.assign(self.ξ + self.grad_φ_dg)  # x = ξ + grad(φ)

    def get_diagnostics(self):
        """
        Compute
          1) ratio of smallest and largest elemental volumes;
          2) equidistribution of elemental volumes;
          3) relative L2 norm residual.
        """
        v = self.volume.vector().gather()
        minmax = v.min()/v.max()
        mean = v.sum()/v.size
        w = v.copy() - mean
        w *= w
        std = sqrt(w.sum()/w.size)
        equi = std/mean
        residual_l2 = assemble(self.residual_l2_form).dat.norm
        norm_l2 = assemble(self.norm_l2_form).dat.norm
        residual_l2_norm = residual_l2 / norm_l2
        return minmax, equi, residual_l2_norm

    def setup_equidistribution(self):  # TODO: Other options, e.g. MMPDE
        """
        Setup solvers for nonlinear iteration. Two approaches are considered, as specified by
        `self.method` - either a relaxation using pseudo-timestepping ('relaxation'), or a
        quasi-Newton nonlinear solver ('quasi_newton').

        The former approach solves linear systems at each pseudo-timestep. Whilst each of these
        solves can be done efficiently, the algorithm can take O(100) nonlinear iterations to
        converge. In many cases, the algorithm diverges.

        The latter approach provides increased robustness and typically converges within O(10)
        nonlinear iterations.
        """
        n = FacetNormal(self.mesh)
        if self.method == 'relaxation':
            self.setup_pseudotimestepper()
            σ, τ = TrialFunction(self.V_ten), TestFunction(self.V_ten)
            a = inner(τ, σ)*dx
            L = -dot(div(τ), grad(self.φ_new))*dx
            # L += (τ[0, 0]*n[0]*self.φ_new.dx(0) + τ[1, 1]*n[1]*self.φ_new.dx(1))*ds
            L += (τ[0, 1]*n[1]*self.φ_new.dx(0) + τ[1, 0]*n[0]*self.φ_new.dx(1))*ds
            prob = LinearVariationalProblem(a, L, self.σ_new)
            self.equidistribution = LinearVariationalSolver(prob, solver_parameters={'ksp_type': 'cg'})
        else:
            φ, σ = TrialFunctions(self.W)
            ψ, τ = TestFunctions(self.W)
            F = inner(τ, self.σ_new)*dx + dot(div(τ), grad(self.φ_new))*dx
            # F -= (τ[0, 0]*n[0]*self.φ_new.dx(0) + τ[1, 1]*n[1]*self.φ_new.dx(1))*ds
            F -= (τ[0, 1]*n[1]*self.φ_new.dx(0) + τ[1, 0]*n[0]*self.φ_new.dx(1))*ds
            F -= ψ*(self.monitor*det(self.I + self.σ_new) - self.θ)*dx

            def generate_m(cursol):
                with self.φσ_temp.dat.vec as v:
                    cursol.copy(v)

                # Perform L2 projection and generate coordinates appropriately
                self.l2_projector.solve()
                self.apply_map()

                self.mesh.coordinates.assign(self.x)
                self.update_monitor()
                self.mesh.coordinates.assign(self.ξ)

                # Evaluate normalisation coefficient
                self.θ.assign(assemble(self.θ_form)/self.total_volume)

            self.generate_m = generate_m

            # Custom preconditioning matrix
            Jp = inner(τ, σ)*dx + φ*ψ*dx + dot(grad(φ), grad(ψ))*dx

            prob = NonlinearVariationalProblem(F, self.φσ, Jp=Jp)
            nullspace = MixedVectorSpaceBasis(self.W, [self.V_nullspace, self.W.sub(1)])

            params = {"ksp_type": "gmres",
                      "pc_type": "fieldsplit",
                      "pc_fieldsplit_type": "multiplicative",
                      "pc_fieldsplit_off_diag_use_amat": True,
                      "fieldsplit_0_pc_type": "gamg",
                      "fieldsplit_0_ksp_type": "preonly",
                      "fieldsplit_0_mg_levels_ksp_max_it": 5,
                      # "fieldsplit_0_mg_levels_pc_type": "bjacobi",  # parallel
                      # "fieldsplit_0_mg_levels_sub_ksp_type": "preonly",  # parallel
                      # "fieldsplit_0_mg_levels_sub_pc_type": "ilu",  # parallel
                      "fieldsplit_0_mg_levels_pc_type": "ilu",  # serial
                      # "fieldsplit_1_pc_type": "bjacobi",  # parallel
                      # "fieldsplit_1_sub_ksp_type": "preonly",  # parallel
                      # "fieldsplit_1_sub_pc_type": "ilu",  # parallel
                      "fieldsplit_1_pc_type": "ilu",  # serial
                      "fieldsplit_1_ksp_type": "preonly",
                      "ksp_max_it": 200,
                      "snes_max_it": 125,
                      "ksp_gmres_restart": 200,
                      "snes_rtol": self.op.r_adapt_rtol,
                      "snes_linesearch_type": "l2",
                      "snes_linesearch_max_it": 5,
                      "snes_linesearch_maxstep": 1.05,
                      "snes_linesearch_damping": 0.8,
                      # "ksp_monitor": True,
                      # "snes_monitor": True,
                      # "snes_linesearch_monitor": True,
                      "snes_lag_preconditioner": -1}

            self.equidistribution = NonlinearVariationalSolver(prob, nullspace=nullspace,
                                                               transpose_nullspace=nullspace,
                                                               pre_jacobian_callback=self.generate_m,
                                                               pre_function_callback=self.generate_m,
                                                               solver_parameters=params)

            def fakemonitor(snes, i, rnorm):
                cursol = snes.getSolution()
                self.generate_m(cursol)  # Updates monitor function and normalisation constant

                self.mesh.coordinates.assign(self.x)
                assemble(self.L_p0, tensor=self.volume)  # For equidistribution measure
                self.volume /= self.original_volume
                if self.op.debug:
                    self.volume_file.write(self.volume)
                self.mesh.coordinates.assign(self.ξ)

                # Convergence criteria
                if self.op.debug:
                    minmax, equi, residual_l2_norm = self.get_diagnostics()
                    print_output(self.msg.format(i, minmax, residual_l2_norm, equi))

            self.fakemonitor = fakemonitor

        self.setup_residuals()

    def initialise_sigma(self):
        σ, τ = TrialFunction(self.V_ten), TestFunction(self.V_ten)
        n = FacetNormal(self.mesh)
        a = inner(τ, σ)*dx
        L = -dot(div(τ), grad(self.φ_new))*dx
        # L += (τ[0, 0]*n[0]*self.φ_new.dx(0) + τ[1, 1]*n[1]*self.φ_new.dx(1))*ds
        L += (τ[0, 1]*n[1]*self.φ_new.dx(0) + τ[1, 0]*n[0]*self.φ_new.dx(1))*ds
        σ_init = Function(self.V_ten)
        solve(a == L, σ_init, solver_parameters={'ksp_type': 'cg'})
        self.φσ.sub(1).assign(σ_init)

    def setup_l2_projector(self):
        u_cts, v_cts = TrialFunction(self.P1_vec), TestFunction(self.P1_vec)
        a = dot(v_cts, u_cts)*dx
        L = dot(v_cts, grad(self.φ_old))*dx
        bcs = []
        for i in self.mesh.exterior_facets.unique_markers:
            n = [assemble(FacetNormal(self.mesh)[0]*ds(i)), assemble(FacetNormal(self.mesh)[1]*ds(i))]
            if np.allclose(n[0], 0.0) and np.allclose(n[1], 0.0):
                raise ValueError("Invalid normal vector {:}".format(n))
            elif np.allclose(n[0], 0.0):
                bcs.append(DirichletBC(self.P1_vec.sub(1), 0.0, i))
            elif np.allclose(n[1], 0.0):
                bcs.append(DirichletBC(self.P1_vec.sub(0), 0.0, i))
            else:
                raise NotImplementedError("Have not yet considered non-axes-aligned boundaries.")  # TODO
        prob = LinearVariationalProblem(a, L, self.grad_φ_cts, bcs=bcs)
        self.l2_projector = LinearVariationalSolver(prob, solver_parameters={'ksp_type': 'cg'})

    def adapt(self):
        if self.method == 'quasi_newton':
            self.equidistribution.snes.setMonitor(self.fakemonitor)
            self.equidistribution.solve()
            return

        assert self.method == 'relaxation'
        maxit = self.op.r_adapt_maxit
        for i in range(maxit):

            # Perform L2 projection and generate coordinates appropriately
            self.l2_projector.solve()
            self.apply_map()

            # Update monitor function
            self.mesh.coordinates.assign(self.x)
            self.update_monitor()
            assemble(self.L_p0, tensor=self.volume)  # For equidistribution measure
            self.volume /= self.original_volume
            if i % 10 == 0 and self.op.debug:
                self.monitor_file.write(self.monitor)
                self.volume_file.write(self.volume)
            self.mesh.coordinates.assign(self.ξ)

            # Evaluate normalisation coefficient
            self.θ.assign(assemble(self.θ_form)/self.total_volume)

            # Convergence criteria
            minmax, equi, residual_l2_norm = self.get_diagnostics()
            if i == 0:
                initial_norm = residual_l2_norm  # Store to check for divergence
            if i % 10 == 0 and self.op.debug:
                print_output(self.msg.format(i, minmax, residual_l2_norm, equi))
            if residual_l2_norm < self.op.r_adapt_rtol:
                print_output("r-adaptation converged in {:d} iterations.".format(i+1))
                break
            if residual_l2_norm > 2.0*initial_norm:
                raise ConvergenceError("r-adaptation failed to converge in {:d} iterations.".format(i+1))
            self.pseudotimestepper.solve()
            self.equidistribution.solve()
            self.φ_old.assign(self.φ_new)
            self.σ_old.assign(self.σ_new)
