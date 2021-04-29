"""
***************************************************************************************************
*   NOTE: Much of the code presented here is copied directly from the code associated with [1].   *
***************************************************************************************************
[1] A.T.T. McRae, C.J. Cotter, and C.J. Budd, "Optimal-transport--based mesh adaptivity on the
    plane and sphere using finite elements." SIAM Journal on Scientific Computing 40.2 (2018):
    A1121-A1148.
"""
from thetis import *
from adapt_utils.options import Options
from adapt_utils.params import serial_quasi_newton, parallel_quasi_newton
from pyadjoint import AdjFloat, no_annotations
__all__ = ["MeshMover"]

class MeshMover(object):
    r"""
    A class dedicated to performing mesh r-adaptation. Given a source mesh and a monitor function
    :math: `m`, the new mesh is established by relocating the vertices of the original mesh. That
    is, the topology remains unchanged.
    The mesh movement is determined by solving the Monge-Ampère type equation
..  math::
        m(x)\det(I + H(\phi)) = \theta,
    for a scalar potential :math:`\phi`, where :math:`I` is the identity, :math:`\theta` is a
    normalising constant and :math:`H(\phi)` denotes the Hessian of :math:`\phi` with respect to
    coordinates on the computational mesh.
    The implementation is an object-oriented version of that given in [1], extended to account
    for boundary conditions:
        * `bc is None` implies that the mesh is only moved tangential to the boundary, as a
          post-processing step. This uses the `EquationBC` construct;
        * `bc == []` implies free-slip conditions with no post-processing. This does not
          guarantee that the mesh will only move tangential to the boundary, although it is
          approximately true for simple domains;
        * setting `bc` to a `DirichletBC` will fix the boundary mesh nodes as a post-processing
          step.
    """
    def __init__(self, mesh, monitor_function, method='monge_ampere', bc=None, bbc=None, op=Options()):
        self.mesh = mesh
        self.dim = self.mesh.topological_dimension()
        if self.dim != 2:
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
        self._initialise_sigma()        
        self._setup_equidistribution()
        self._setup_l2_projector()
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
        assert self.monitor_function is not None
        self.monitor.interpolate(self.monitor_function(mesh=self.mesh, x=self.x))

    def _setup_pseudotimestepper(self):
        if not self.op.nonlinear_method == 'relaxation':
            return
        φ, ψ = TrialFunction(self.V), TestFunction(self.V)
        a = dot(grad(ψ), grad(φ))*dx
        L = dot(grad(ψ), grad(self.φ_old))*dx
        L += self.pseudo_dt*ψ*(self.total_volume*self.monitor*det(self.I + self.σ_old) - self.θ)*dx
        prob = LinearVariationalProblem(a, L, self.φ_new)
        kwargs = {'solver_parameters': {'ksp_type': 'cg', 'pc_type': 'gamg'},
                  'nullspace': self.V_nullspace, 'transpose_nullspace': self.V_nullspace}
        self.pseudotimestepper = LinearVariationalSolver(
            prob,
            solver_parameters={
                'snes_type': 'ksponly',
                'ksp_type': 'cg',
                'pc_type': 'gamg',
            },
            nullspace=self.V_nullspace,
            transpose_nullspace=self.V_nullspace,
        )

    def _setup_residuals(self):
        ψ = TestFunction(self.V)
        self.θ_form = self.monitor*det(self.I + self.σ_old)*dx
        self.residual_l2_form = ψ*(self.monitor*det(self.I + self.σ_old) - self.θ/self.total_volume)*dx
        self.norm_l2_form = ψ*self.θ/self.total_volume*dx

    def _apply_map(self):
        r"""
        Transfer L2 projected gradient from P1 to P1DG space and use it to perform the coordinate
        transform between computational and physical mesh:
      ..math::
            \mathbf x = \boldsymbol\xi + \nabla\phi.
        """
        self.grad_φ_dg.interpolate(self.grad_φ_cts)
        self.x.assign(self.ξ + self.grad_φ_dg)

    @no_annotations
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
            L = -dot(div(τ), grad(self.φ_new))*dx \
                + τ[0, 1]*n[1]*self.φ_new.dx(0)*ds \
                + τ[1, 0]*n[0]*self.φ_new.dx(1)*ds
            prob = LinearVariationalProblem(a, L, self.σ_new)
            params = {
                'ksp_type': 'cg',
            }
            self.equidistribution = LinearVariationalSolver(prob, solver_parameters=params)
        else:
            φ, σ = TrialFunctions(self.W)
            ψ, τ = TestFunctions(self.W)
            F = inner(τ, self.σ_new)*dx \
                + dot(div(τ), grad(self.φ_new))*dx \
                - τ[0, 1]*n[1]*self.φ_new.dx(0)*ds \
                - τ[1, 0]*n[0]*self.φ_new.dx(1)*ds \
                - ψ*(self.monitor*det(self.I + self.σ_new) - self.θ)*dx

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
                self.θ.assign(assemble(self.θ_form))

            self.generate_m = generate_m

            # Custom preconditioning matrix
            Jp = inner(τ, σ)*dx + φ*ψ*dx + dot(grad(φ), grad(ψ))*dx

            prob = NonlinearVariationalProblem(F, self.φσ, Jp=Jp)
            nullspace = MixedVectorSpaceBasis(self.W, [self.V_nullspace, self.W.sub(1)])

            params = serial_quasi_newton if COMM_WORLD.size == 1 else parallel_quasi_newton
            params["snes_rtol"] = self.op.r_adapt_rtol
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
                self.volume.assign(self.volume/self.original_volume)
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
        L = -dot(div(τ), grad(self.φ_new))*dx \
            + τ[0, 1]*n[1]*self.φ_new.dx(0)*ds \
            + τ[1, 0]*n[0]*self.φ_new.dx(1)*ds
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

    def adapt(self, **kwargs):
        """Run the desired mesh movement algorithm."""
        if self.method == 'monge_ampere':
            self.adapt_monge_ampere(**kwargs)

    def adapt_monge_ampere(self, rtol=None, maxiter=None):
        """
        Apply mesh movement of Monge-Ampere type.
        """
        rtol = rtol or self.op.r_adapt_rtol
        maxit = maxiter or self.op.r_adapt_maxit
        if self.nonlinear_method == 'quasi_newton':
            self.equidistribution.snes.setTolerances(rtol=rtol, max_it=maxit)
            self.equidistribution.snes.setMonitor(self.fakemonitor)
            self.equidistribution.solve()
            i = self.equidistribution.snes.getIterationNumber()
            print_output(f"r-adaptation converged in {i+1} iterations")
            return

        assert self.nonlinear_method == 'relaxation'
        for i in range(maxit):

            # Perform L2 projection and generate coordinates appropriately
            self.l2_projector.solve()
            self._apply_map()

            # Update monitor function
            self.mesh.coordinates.assign(self.x)
            self.update_monitor_function()
            assemble(self.L_p0, tensor=self.volume)  # For equidistribution measure
            self.volume.assign(self.volume/self.original_volume)
            if self.op.debug and self.op.debug_mode == 'full':
                self.monitor_file.write(self.monitor)
                self.volume_file.write(self.volume)
            self.mesh.coordinates.assign(self.ξ)

            # Evaluate normalisation coefficient
            self.θ.assign(AdjFloat(assemble(self.θ_form)))

            # Convergence criteria
            minmax, residual_l2_norm, equi = self.get_diagnostics()
            if i == 0:
                initial_norm = residual_l2_norm  # Store to check for divergence
            print_output(self.msg.format(i, minmax, residual_l2_norm, equi))
            if residual_l2_norm < rtol:
                print_output(f"r-adaptation converged in {i+1} iterations.")
                break
            if residual_l2_norm > 2.0*initial_norm:
                raise ConvergenceError(f"r-adaptation failed to converge in {i+1} iterations.")

            # Solve
            self.pseudotimestepper.solve()
            self.equidistribution.solve()
            self.φ_old.assign(self.φ_new)
            self.σ_old.assign(self.σ_new)
            if i == maxit-1:
                raise ConvergenceError(f"r-adaptation failed to converge in {i+1} iterations.")