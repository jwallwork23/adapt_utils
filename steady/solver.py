from thetis import *

import numpy as np
import os

from adapt_utils.adapt.adaptation import pragmatic_adapt
from adapt_utils.adapt.kernels import *
from adapt_utils.adapt.metric import *
from adapt_utils.norms import *
from adapt_utils.unsteady.solver import AdaptiveProblem


__all__ = ["AdaptiveSteadyProblem"]


# TODO:
#  * PressureProjectionPicard

class AdaptiveSteadyProblem(AdaptiveProblem):
    """
    Default steady state model: 2D coupled shallow water + tracer transport.

    This class inherits from its unsteady counterpart, :class:`AdaptiveProblem`. Whilst its parent
    supports solving PDEs on a sequence of meshes using adaptation techniques, this child class
    relates to time-independent problems and so this does not make sense.
    """
    def __init__(self, op, discrete_adjoint=False, **kwargs):
        super(AdaptiveSteadyProblem, self).__init__(op, **kwargs)
        op = self.op
        self.discrete_adjoint = discrete_adjoint
        if self.num_meshes > 1:
            raise ValueError("`AdaptiveSteadyProblem` only supports single meshes.")
        ts = op.timestepper
        # if ts not in ["SteadyState", "PressureProjectionPicard"]:  # TODO
        if ts not in ["SteadyState", ]:
            raise ValueError("Timestepper {:s} not allowed for steady-state problems.".format(ts))
        if op.solve_swe:
            self.equation_set = 'shallow_water'
        elif op.solve_tracer:
            self.equation_set = 'tracer'
        else:
            raise ValueError("Steady-state solver only supports one of hydrodynamics and tracers.")

    @property
    def function_space(self):
        return self.V[0] if self.solve_swe else self.Q[0]

    @property
    def timestepper(self):  # TODO: PressureProjectionPicard
        steppers = self.timesteppers[0]
        return steppers.shallow_water if self.op.solve_swe else steppers.tracer

    @property
    def kernel(self):
        op = self.op
        return op.set_qoi_kernel(self, 0) if op.solve_swe else op.set_qoi_kernel_tracer(self, 0)

    @property
    def indicator(self):
        return self.indicators[0]

    @property
    def estimator(self):
        return self.estimators[0]

    def solve_adjoint(self, **kwargs):
        if self.discrete_adjoint:
            self._solve_discrete_adjoint(**kwargs)
        else:
            self._solve_continuous_adjoint(**kwargs)

    def _solve_continuous_adjoint(self, **kwargs):
        super(AdaptiveSteadyProblem, self).solve_adjoint(**kwargs)

    def _solve_discrete_adjoint(self, **kwargs):  # TODO: PressureProjectionPicard
        fs = self.function_space
        F = self.timestepper.F

        # Linearise the form, if requested
        if not self.nonlinear:
            tmp_u = Function(fs)
            F = action(lhs(F), tmp_u) - rhs(F)
            F = replace(F, {tmp_u: self.fwd_solution})

        # Take the adjoint
        dFdu = derivative(F, self.fwd_solution, TrialFunction(fs))
        dFdu_form = adjoint(dFdu)
        dJdu = derivative(self.quantity_of_interest_form(), self.fwd_solution, TestFunction(fs))
        bcs = None  # TODO: Hook up as in setup_solver_adjoint in the unsteady solver
        params = self.op.adjoint_solver_parameters[self.equation_set]
        solve(dFdu_form == dJdu, self.adj_solution, bcs=bcs, solver_parameters=params)

    def quantity_of_interest(self):
        """
        Returns the value of the quantity of interest (QoI) for the current forward solution.
        """
        if self.op.solve_tracer:
            return assemble(inner(self.fwd_solution_tracer, self.kernel)*dx(degree=12))
        elif self.op.solve_swe:
            return assemble(inner(self.fwd_solution, self.kernel)*dx(degree=12))
        else:
            raise NotImplementedError  # TODO

    def quantity_of_interest_form(self):
        """
        UFL form describing the quantity of interest (QoI).
        """
        return inner(self.fwd_solution, self.kernel)*dx(degree=12)

    def get_scaled_residual(self, adjoint=False, **kwargs):
        r"""
        Evaluate the scaled form of the residual, as used in [Becker & Rannacher, 2001].
        i.e. the $\rho_K$ term.
        """
        raise NotImplementedError  # TODO

    def get_scaled_weights(self, adjoint=False, **kwargs):
        r"""
        Evaluate the scaled form of the residual weights, as used in [Becker & Rannacher, 2001].
        i.e. the $\omega_K$ term.
        """
        raise NotImplementedError  # TODO

    def get_dwr_upper_bound(self, adjoint=False, **kwargs):
        r"""
        Evaluate an upper bound for the DWR given by the product of residual and weights,
        as used in [Becker & Rannacher, 2001].
        i.e. $\rho_K \omega_K$.
        """
        raise NotImplementedError  # TODO

    def get_difference_quotient(self, adjoint=False, **kwargs):
        """
        Evaluate difference quotient approximation to the DWR given by the product of residual and
        flux term evaluated at the adjoint solution, as used in [Becker & Rannacher, 2001].
        """
        raise NotImplementedError  # TODO

    def get_strong_residual(self, adjoint=False, **kwargs):
        """
        Compute the strong residual for the forward or adjoint PDE, as specified by the `adjoint`
        boolean kwarg.
        """
        if adjoint:
            return self.get_strong_residual_adjoint(**kwargs)
        else:
            return self.get_strong_residual_forward(**kwargs)

    def get_strong_residual_forward(self, **kwargs):
        ts = self.timesteppers[0][self.op.adapt_field]
        strong_residual = abs(ts.error_estimator.strong_residual)
        # strong_residual_cts = project(strong_residual, self.P1[0])
        strong_residual_cts = interpolate(strong_residual, self.P1[0])
        return strong_residual_cts

    def get_strong_residual_adjoint(self, **kwargs):
        raise NotImplementedError  # TODO

    def get_flux(self, adjoint=False, **kwargs):
        """
        Evaluate flux terms for forward or adjoint PDE, as specified by the `adjoint` boolean kwarg.
        """
        if adjoint:
            return self.get_flux_adjoint(**kwargs)
        else:
            return self.get_flux_forward(**kwargs)

    def get_flux_forward(self):
        raise NotImplementedError  # TODO

    def get_flux_adjoint(self):
        raise NotImplementedError  # TODO

    def get_dwr_residual(self, adjoint=False):
        """
        Evaluate the cellwise component of the forward or adjoint 'Dual Weighted Residual' (DWR)
        error estimator (see [Becker and Rannacher, 2001]), as specified by the boolean kwarg
        `adjoint`.
        """
        return self.get_dwr_residual_adjoint() if adjoint else self.get_dwr_residual_forward()

    def get_dwr_residual_forward(self):
        raise NotImplementedError  # TODO

    def get_dwr_residual_adjoint(self):
        raise NotImplementedError  # TODO

    def get_dwr_flux(self, adjoint=False):
        """
        Evaluate the edgewise component of the forward or adjoint Dual Weighted Residual (DWR) error
        estimator (see [Becker and Rannacher, 2001]), as specified by the boolean kwarg `adjoint`.
        """
        return self.get_dwr_flux_adjoint() if adjoint else self.get_dwr_flux_forward()

    def get_dwr_flux_forward(self):
        raise NotImplementedError  # TODO

    def get_dwr_flux_adjoint(self):
        raise NotImplementedError  # TODO

    def indicate_error(self, adapt_field, approach=None):
        op = self.op
        approach = approach or op.approach
        if approach == 'dwr':
            return self.dwr_indicator(adapt_field, adjoint=False)
        elif approach == 'dwr_adjoint':
            return self.dwr_indicator(adapt_field, adjoint=True)
        else:
            raise NotImplementedError  # TODO

    def dwr_indicator(self, adapt_field, adjoint=False):
        """
        Indicate errors in the quantity of interest by the 'Dual Weighted Residual' (DWR) method of
        [Becker and Rannacher, 2001].

        A P1 field to be used for isotropic mesh adaptation is stored as `self.indicator`.
        """
        op = self.op
        self.indicator[op.approach] = Function(self.P1[0], name=op.approach + " indicator")

        # Setup problem on enriched space
        hierarchy = MeshHierarchy(self.mesh, 1)
        refined_mesh = hierarchy[1]
        ep = type(self)(
            op,
            meshes=refined_mesh,
            nonlinear=self.nonlinear,
        )
        ep.outer_iteration = self.outer_iteration
        enriched_space = ep.get_function_space(adapt_field)
        tm = dmhooks.get_transfer_manager(self.get_plex(0))

        # Setup forward solver for enriched problem
        if adjoint:
            raise NotImplementedError  # TODO
        else:
            ep.create_error_estimators_step(0)
            ep.solve_adjoint()
            enriched_adj_solution = ep.get_solutions(adapt_field, adjoint=True)[0]

            # Create solution fields
            fwd_solution = self.get_solutions(adapt_field)[0]
            adj_solution = self.get_solutions(adapt_field, adjoint=True)[0]
            indicator_enriched = Function(ep.P0[0])
            fwd_proj = Function(enriched_space[0])
            adj_error = Function(enriched_space[0])
            bcs = self.boundary_conditions[0][adapt_field]

            # Setup error estimator
            ep.setup_solver_forward_step(0)  # Needed to create timestepper
            ets = ep.timesteppers[0][adapt_field]
            ets.setup_error_estimator(fwd_proj, fwd_proj, adj_error, bcs)

            # Prolong forward solution at current timestep
            tm.prolong(fwd_solution, fwd_proj)

            # Approximate adjoint error in enriched space
            tm.prolong(adj_solution, adj_error)
            adj_error *= -1
            adj_error += enriched_adj_solution

        # Compute dual weighted residual
        dwr = ets.error_estimator.weighted_residual()
        self.estimator[op.approach].append(dwr.vector().gather().sum())
        indicator_enriched.interpolate(abs(dwr))
        # indicator_enriched_cts = project(indicator_enriched, ep.P1[0])
        indicator_enriched_cts = interpolate(indicator_enriched, ep.P1[0])
        # self.indicator[op.approach].project(indicator_enriched_cts)
        tm.inject(indicator_enriched_cts, self.indicator[op.approach])
        return self.indicator[op.approach]

    def get_hessian_metric(self, adjoint=False, **kwargs):
        """
        Compute an appropriate Hessian metric for the problem at hand. This is inherently
        problem-dependent, since the choice of field for adaptation is not universal.

        Hessian metric should be computed and stored as `self.M`.
        """
        raise NotImplementedError  # TODO

    def get_hessian(self, adjoint=False):
        """
        Compute an appropriate Hessian for the problem at hand. This is inherently
        problem-dependent, since the choice of field for adaptation is not universal.
        """
        raise NotImplementedError  # TODO

    def get_metric(self, adapt_field):
        if self.op.approach in ('dwr', 'dwr_adjoint'):
            metric = self.get_isotropic_metric(self.op.adapt_field)
        elif self.op.approach == 'weighted_hessian':
            metric = self.get_weighted_hessian_metric(adjoint=False)
        elif self.op.approach == 'weighted_gradient':
            metric = self.get_weighted_gradient_metric(adjoint=False)
        elif self.op.approach == 'anisotropic_dwr':
            metric = self.get_anisotropic_dwr_metric(adjoint=False)
        else:
            raise NotImplementedError  # TODO
        if self.op.plot_pvd:
            File(os.path.join(self.di, 'metric.pvd')).write(metric)
        return metric

    def get_isotropic_metric(self, adapt_field):
        """
        Scale an identity matrix by the indicator field in order to drive
        isotropic mesh refinement.
        """
        indicator = self.indicate_error(adapt_field)
        if self.op.plot_pvd:
            File(os.path.join(self.di, 'indicator.pvd')).write(indicator)
        metric = Function(self.P1_ten[0], name="Metric")
        metric.assign(isotropic_metric(indicator, normalise=True, op=self.op))
        return metric

    def get_weighted_hessian_metric(self, adjoint=False):
        """
        Construct an anisotropic metric using an approach inspired by [Power et al. 2006].

        If `adjoint` mode is turned off, weight the Hessian of the adjoint solution with a residual
        for the forward PDE. Otherwise, weight the Hessian of the forward solution with a residual
        for the adjoint PDE.
        """
        strong_residual = self.get_strong_residual(adjoint=adjoint)
        self.recover_hessian_metric(normalise=False, enforce_constraints=False, adjoint=not adjoint)
        scaled_hessian = interpolate(strong_residual*self.metrics[0], self.P1_ten[0])
        return steady_metric(H=scaled_hessian, normalise=True, enforce_constraints=True, op=self.op)

    def get_weighted_gradient_metric(self, adjoint=False, source=True):
        """
        Construct an anisotropic metric using an approach inspired by [Loseille et al. 2009].
        """
        from adapt_utils.adapt.recovery import recover_gradient, recover_boundary_hessian

        op = self.op
        mesh = self.mesh
        P1_ten = self.P1_ten[0]
        adapt_field = op.adapt_field
        if adapt_field not in ('tracer', 'sediment', 'bathymetry'):
            adapt_field = 'shallow_water'
        if op.solve_tracer and adapt_field == 'tracer':
            c = self.fwd_solution_tracer
            c_star = self.adj_solution_tracer

            # Interior gradient term
            grad_c_star = recover_gradient(c_star, op=op)

            # Interior Hessian term
            u, eta = split(self.fwd_solutions[0])
            D = self.fields[0].horizontal_diffusivity
            F1 = u[0]*c - D*c.dx(0)
            F2 = u[1]*c - D*c.dx(1)
            kwargs = dict(normalise=True, noscale=True, enforce_constraints=False, mesh=mesh, op=op)
            interior_hessians = [
                interpolate(steady_metric(F1, **kwargs)*abs(grad_c_star[0]), P1_ten),
                interpolate(steady_metric(F2, **kwargs)*abs(grad_c_star[1]), P1_ten)
            ]

            # Interior source Hessian term
            if source:
                S = self.fields[0].tracer_source_2d
                interior_hessians.append(interpolate(steady_metric(S, **kwargs)*abs(c_star), P1_ten))

            # Average interior Hessians
            interior_hessian = metric_average(*interior_hessians)

            # Boundary Hessian
            n = FacetNormal(mesh)
            Fbar = c*dot(u, n) - D*dot(grad(c), n)  # NOTE: Minus zero (imposed boundary value)
            bcs = self.boundary_conditions[0]['tracer']
            tags = [tag for tag in bcs if 'diff_flux' in bcs[tag]]
            Hs = recover_boundary_hessian(Fbar, mesh=mesh, op=op, boundary_tag=tags)
            boundary_hessian = abs(c_star)*as_matrix([[Constant(1/op.h_max**2), 0], [0, abs(Hs)]])

            # Get target complexities based on interior and boundary Hessians
            integrals = volume_and_surface_contributions(interior_hessian, boundary_hessian, op=op)

            # Assemble and combine metrics
            kwargs = dict(normalise=True, enforce_constraints=True, mesh=mesh, op=op)
            kwargs['integral'] = integrals[0]
            interior_metric = steady_metric(H=interior_hessian, **kwargs)
            kwargs['integral'] = integrals[1]
            boundary_metric = boundary_steady_metric(boundary_hessian, **kwargs)
            return metric_intersection(interior_metric, boundary_metric, boundary_tag=tags)
        else:
            raise NotImplementedError  # TODO

    def get_anisotropic_dwr_metric(self, adjoint=False):
        """
        Construct an anisotropic metric using an approach inspired by [Carpio et al. 2013].
        """
        dim = self.mesh.topological_dimension()
        adapt_field = self.op.adapt_field
        if adapt_field not in ('tracer', 'sediment', 'bathymetry'):
            adapt_field = 'shallow_water'
        fwd_solution = self.get_solutions(adapt_field)[0]

        # Compute error indicators
        dwr = self.indicate_error('tracer', approach='dwr_adjoint' if adjoint else 'dwr')

        # Get current element volume
        K = Function(self.P0[0], name="Element volume")
        K_hat = 0.5  # Area of a triangular reference element
        K.interpolate(K_hat*abs(JacobianDeterminant(self.mesh)))

        # Get optimal element volume
        K_opt = Function(self.P0[0], name="Optimal element volume")
        alpha = self.op.convergence_rate
        K_opt.interpolate(pow(dwr, 1/(alpha + 1)))
        Sum = K_opt.vector().gather().sum()
        if self.op.normalisation == 'error':
            scaling = pow(Sum*self.op.target, -1/alpha)  # FIXME
        else:
            scaling = Sum/self.op.target
        K_opt.interpolate(min_value(max_value(scaling*K/K_opt, self.op.h_min**2), self.op.h_max**2))

        # Recover Hessian and compute eigendecomposition
        H = Function(self.P0_ten[0], name="Elementwise Hessian")
        kwargs = dict(mesh=self.mesh, normalise=False, enforce_constrants=False, op=self.op)
        if self.op.adapt_field == 'shallow_water':
            raise NotImplementedError  # TODO
        else:
            # TODO: Could the Hessian not be recovered in P0 space?
            H.project(steady_metric(fwd_solution, **kwargs))
        evectors = Function(self.P0_ten[0], name="Elementwise Hessian eigenvectors")
        evalues = Function(self.P0_vec[0], name="Elementwise Hessian eigenvalues")
        kernel = eigen_kernel(get_reordered_eigendecomposition, dim)
        op2.par_loop(kernel, self.P0_ten[0].node_set, evectors.dat(op2.RW), evalues.dat(op2.RW), H.dat(op2.READ))

        # Compute stretching factor
        s = sqrt(abs(evalues[0]/evalues[-1]))

        # Build metric
        M = Function(self.P0_ten[0], name="Elementwise metric")
        if dim == 2:
            # NOTE: Here the abs replaces a squared square root
            evalues.interpolate(as_vector([abs(K_hat/K_opt*s), abs(K_hat/K_opt/s)]))
        else:
            raise NotImplementedError  # TODO
        kernel = eigen_kernel(set_eigendecomposition_transpose, dim)
        op2.par_loop(kernel, self.P0_ten[0].node_set, M.dat(op2.RW), evectors.dat(op2.READ), evalues.dat(op2.READ))

        # Project metric
        M_p1 = Function(self.P1_ten[0], name="Anisotropic DWR metric")
        M_p1.project(M)
        return M_p1

    def run_dwr(self, **kwargs):
        """
        Apply a goal-oriented mesh adaptation loop, until a convergence criterion is met.

        Convergence criteria:
          * Convergence of quantity of interest (relative tolerance `op.qoi_rtol`);
          * Convergence of mesh element count (relative tolerance `op.element_rtol`);
          * Convergence of error estimator (relative tolerance `op.estimator_rtol`);
          * Maximum number of iterations reached (`op.max_adapt`).

        A minimum number of iterations may also be imposed via `op.min_adapt`.
        """
        op = self.op
        adapt_field = op.adapt_field
        if adapt_field not in ('tracer', 'sediment', 'bathymetry'):
            adapt_field = 'shallow_water'
        self.estimator[op.approach] = []
        for n in range(op.max_adapt):
            self.outer_iteration = n
            self.create_error_estimators_step(0)

            # Solve forward in base space
            self.solve_forward()

            # Check QoI convergence
            qoi = self.quantity_of_interest()
            self.print("Quantity of interest {:d}: {:.4e}".format(n+1, qoi))
            self.qois.append(qoi)
            if len(self.qois) > 1 and n >= op.min_adapt:
                if np.abs(self.qois[-1] - self.qois[-2]) < op.qoi_rtol*self.qois[-2]:
                    self.print("Converged quantity of interest!")
                    break

            # Check maximum number of iterations
            if n == op.max_adapt - 1:
                break

            # Solve adjoint equation in base space
            self.solve_adjoint()

            # Construct metric
            metric = self.get_metric(adapt_field)

            # Check convergence of error estimator
            if 'dwr' in op.approach:
                estimators = self.estimator[op.approach]
                if len(estimators) > 1 and n >= op.min_adapt:
                    if np.abs(estimators[-1] - estimators[-2]) <= op.estimator_rtol*estimators[-2]:
                        self.print("Converged error estimator!")
                        break

            # TODO: Log complexities

            # Adapt mesh
            self.print("\nStarting mesh adaptation for iteration {:d}...".format(n+1))
            self.meshes[0] = pragmatic_adapt(self.mesh, metric)

            # Setup for next run
            self.set_meshes(self.mesh)
            self.setup_all()
            self.dofs.append(np.sum(self.get_function_space(adapt_field)[0].dof_count))

            # Print to screen
            msg = "\nResulting mesh: {:7d} vertices, {:7d} elements"
            num_cells = self.num_cells
            self.print(msg.format(self.num_vertices[-1][0], num_cells[-1][0]))

            # Ensure minimum number of adaptations met
            if n < op.min_adapt:
                continue

            # Check convergence of element count
            if np.abs(num_cells[-1][0] - num_cells[-2][0]) <= op.element_rtol*num_cells[-2][0]:
                self.print("Converged number of mesh elements!")
                break
