from thetis import *

import os

from adapt_utils.options import Options


__all__ = ["MeshMover"]


# TODO: Test Laplacian smoothing
# TODO: Other options, e.g. MMPDE
class MeshMover():
    r"""
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
    def __init__(self, mesh, monitor_function, method='monge_ampere', bc=None, bbc=None, op=Options()):
        self.mesh = mesh
        self.dim = self.mesh.topological_dimension()
        try:
            assert self.dim == 2
        except AssertionError:
            raise NotImplementedError("r-adaptation only currently considered in 2D.")
        self.monitor_function = monitor_function
        assert method in ('monge_ampere', 'laplacian_smoothing')
        self.method = method
        assert op.nonlinear_method in ('quasi_newton', 'relaxation')
        self.bc = bc
        self.bbc = bbc
        self.op = op
        self.ξ = Function(self.mesh.coordinates)  # Computational coordinates
        self.x = Function(self.mesh.coordinates)  # Physical coordinates
        self.I = Identity(self.dim)
        self.dt = Constant(op.dt)
        self.pseudo_dt = Constant(op.pseudo_dt)

        # Create functions and solvers
        self._create_function_spaces()
        self._create_functions()
        self._setup_equidistribution()
        self._setup_l2_projector()
        self._initialise_sigma()
        if self.op.nonlinear_method != 'quasi_newton':
            self._setup_pseudotimestepper()

        # Outputs
        if self.op.debug and self.op.debug_mode == 'full':
            self.monitor_file = File(os.path.join(op.di, 'monitor_debug.pvd'))
            self.monitor_file.write(self.monitor)
            self.volume_file = File(os.path.join(op.di, 'volume_debug.pvd'))
            self.volume_file.write(self.volume)
        self.msg = "{:4d}   Min/Max {:10.4e}   Residual {:10.4e}   Equidistribution {:10.4e}"

    def _create_function_spaces(self):
        self.V = FunctionSpace(self.mesh, "CG", self.op.degree)
        self.V_nullspace = VectorSpaceBasis(constant=True)
        self.V_ten = TensorFunctionSpace(self.mesh, "CG", self.op.degree)
        self.W = self.V*self.V_ten
        self.P1 = FunctionSpace(self.mesh, "CG", 1)
        self.P1_vec = VectorFunctionSpace(self.mesh, "CG", 1)
        self.P1DG_vec = VectorFunctionSpace(self.mesh, "DG", 1)
        self.P0 = FunctionSpace(self.mesh, "DG", 0)
        self.p0test = TestFunction(self.P0)

    def _create_functions(self):
        if self.op.nonlinear_method == 'relaxation':
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
        self.update_monitor_function()
        self.volume = Function(self.P0, name="Mesh volume").assign(assemble(self.p0test*dx))
        self.original_volume = assemble(self.p0test*dx)
        self.total_volume = assemble(Constant(1.0)*dx(domain=self.mesh))
        self.L_p0 = self.p0test*self.monitor*dx
        self.grad_φ_cts = Function(self.P1_vec, name="L2 projected gradient")
        self.grad_φ_dg = Function(self.mesh.coordinates, name="Discontinuous gradient")

    def update_monitor_function(self):
        """Update the monitor function based on the current mesh."""
        if self.monitor_function is not None:
            self.monitor.interpolate(self.monitor_function(self.mesh))

    def _setup_pseudotimestepper(self):
        if not self.op.nonlinear_method == 'relaxation':
            raise ValueError("Pseudotimestepping is only used when a 'relaxation' approach is used.")
        φ, ψ = TrialFunction(self.V), TestFunction(self.V)
        a = dot(grad(ψ), grad(φ))*dx
        L = dot(grad(ψ), grad(self.φ_old))*dx
        L += self.pseudo_dt*ψ*(self.monitor*det(self.I + self.σ_old) - self.θ)*dx
        prob = LinearVariationalProblem(a, L, self.φ_new)
        kwargs = {'solver_parameters': {'ksp_type': 'cg', 'pc_type': 'gamg'},
                  'nullspace': self.V_nullspace, 'transpose_nullspace': self.V_nullspace}
        self.pseudotimestepper = LinearVariationalSolver(prob, **kwargs)

    def _setup_residuals(self):
        ψ = TestFunction(self.V)
        self.θ_form = self.monitor*det(self.I + self.σ_old)*dx
        self.residual_l2_form = ψ*(self.monitor*det(self.I + self.σ_old) - self.θ)*dx
        self.norm_l2_form = ψ*self.θ*dx

    def _apply_map(self):
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
        residual_l2_norm = residual_l2/norm_l2
        return minmax, residual_l2_norm, equi

    def _setup_equidistribution(self):
        """
        Setup solvers for nonlinear iteration. Two approaches are considered, as specified by
        `self.nonlinear_method` - either a relaxation using pseudo-timestepping ('relaxation'), or a
        quasi-Newton nonlinear solver ('quasi_newton').

        The former approach solves linear systems at each pseudo-timestep. Whilst each of these
        solves can be done efficiently, the algorithm can take O(100) nonlinear iterations to
        converge. In many cases, the algorithm diverges.

        The latter approach provides increased robustness and typically converges within O(10)
        nonlinear iterations.
        """
        n = FacetNormal(self.mesh)
        if self.op.nonlinear_method == 'relaxation':
            σ, τ = TrialFunction(self.V_ten), TestFunction(self.V_ten)
            a = inner(τ, σ)*dx
            L = -dot(div(τ), grad(self.φ_new))*dx
            # L += (τ[0, 0]*n[0]*self.φ_new.dx(0) + τ[1, 1]*n[1]*self.φ_new.dx(1))*ds
            L += (τ[0, 1]*n[1]*self.φ_new.dx(0) + τ[1, 0]*n[0]*self.φ_new.dx(1))*ds
            prob = LinearVariationalProblem(a, L, self.σ_new)
            params = {
                'ksp_type': 'cg',
            }
            self.equidistribution = LinearVariationalSolver(prob, solver_parameters=params)
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
                self._apply_map()

                self.mesh.coordinates.assign(self.x)
                self.update_monitor_function()
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
                      "snes_lag_preconditioner": -1}
            if self.op.debug:
                # params["ksp_monitor"] = None
                # params["ksp_monitor_singular_value"] = None
                params["snes_monitor"] = None
                # params["snes_linesearch_monitor"] = None

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
                if self.op.debug and self.op.debug_mode == 'full':
                    self.monitor_file.write(self.monitor)
                    self.volume_file.write(self.volume)
                self.mesh.coordinates.assign(self.ξ)

                # Convergence criteria
                self.op.print_debug(self.msg.format(i, *self.get_diagnostics()))

            self.fakemonitor = fakemonitor

        self._setup_residuals()

    def _initialise_sigma(self):
        """Set an initial guess for Monge-Ampere type mesh movement."""
        σ, τ = TrialFunction(self.V_ten), TestFunction(self.V_ten)
        n = FacetNormal(self.mesh)
        a = inner(τ, σ)*dx
        L = -dot(div(τ), grad(self.φ_new))*dx
        # L += (τ[0, 0]*n[0]*self.φ_new.dx(0) + τ[1, 1]*n[1]*self.φ_new.dx(1))*ds
        L += (τ[0, 1]*n[1]*self.φ_new.dx(0) + τ[1, 0]*n[0]*self.φ_new.dx(1))*ds
        σ_init = Function(self.V_ten)
        solve(a == L, σ_init, solver_parameters={'ksp_type': 'cg'})
        if self.op.nonlinear_method == 'quasi_newton':
            self.φσ.sub(1).assign(σ_init)
        else:
            self.σ_new.assign(σ_init)


    def _setup_l2_projector(self):
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
        print('got here')
        self.l2_projector = LinearVariationalSolver(prob, solver_parameters={'ksp_type': 'cg'})

    def adapt(self):
        """Run the desired mesh movement algorithm."""
        if self.method == 'laplacian_smoothing':
            self.laplacian_smoothing()
            self._apply_map()
        elif self.method == 'monge_ampere':
            self.adapt_monge_ampere()
        else:
            raise NotImplementedError  # TODO

    def laplacian_smoothing(self):
        # TODO: doc
        assert self.bc is not None
        v, v_test = TrialFunction(self.P1_vec), TestFunction(self.P1_vec)
        n = FacetNormal(self.mesh)
        a = -inner(grad(v_test), grad(v))*dx + inner(dot(grad(v), v_test), n)*ds
        L = inner(v_test, Constant(as_vector([0.0, 0.0])))*dx
        solve(a == L, self.grad_φ_cts, bcs=self.bc)

    def adapt_monge_ampere(self):
        # TODO: doc
        if self.op.nonlinear_method == 'quasi_newton':
            self.equidistribution.snes.setMonitor(self.fakemonitor)
            self.equidistribution.solve()
            return

        assert self.op.nonlinear_method == 'relaxation'
        maxit = self.op.r_adapt_maxit
        for i in range(maxit):

            # Perform L2 projection and generate coordinates appropriately
            self.l2_projector.solve()
            self._apply_map()

            # Update monitor function
            self.mesh.coordinates.assign(self.x)
            self.update_monitor_function()
            assemble(self.L_p0, tensor=self.volume)  # For equidistribution measure
            self.volume /= self.original_volume
            if self.op.debug and self.op.debug_mode == 'full':
                self.monitor_file.write(self.monitor)
                self.volume_file.write(self.volume)
            self.mesh.coordinates.assign(self.ξ)

            # Evaluate normalisation coefficient
            self.θ.assign(assemble(self.θ_form)/self.total_volume)

            # Convergence criteria
            minmax, residual_l2_norm, equi = self.get_diagnostics()
            if i == 0:
                initial_norm = residual_l2_norm  # Store to check for divergence
            self.op.print_debug(self.msg.format(i, minmax, residual_l2_norm, equi))
            if residual_l2_norm < self.op.r_adapt_rtol:
                print_output("r-adaptation converged in {:2d} iterations.".format(i+1))
                break
            if residual_l2_norm > 2.0*initial_norm:
                raise ConvergenceError("r-adaptation failed to converge in {:d} iterations.".format(i+1))
            self.pseudotimestepper.solve()
            self.equidistribution.solve()
            self.φ_old.assign(self.φ_new)
            self.σ_old.assign(self.σ_new)
