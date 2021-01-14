from thetis import *

import os

from adapt_utils.adapt.kernels import *
from adapt_utils.adapt.metric import *
from adapt_utils.norms import *
from adapt_utils.unsteady.solver import AdaptiveProblem


__all__ = ["AdaptiveSteadyProblem"]


class AdaptiveSteadyProblem(AdaptiveProblem):
    """
    Default steady state model: 2D coupled shallow water + tracer transport.

    This class inherits from its unsteady counterpart, :class:`AdaptiveProblem`. Whilst its parent
    supports solving PDEs on a sequence of meshes using adaptation techniques, this child class
    relates to time-independent problems and so this does not make sense.
    """
    def __init__(self, op, discrete_adjoint=False, **kwargs):
        super(AdaptiveSteadyProblem, self).__init__(op, **kwargs)
        self.discrete_adjoint = discrete_adjoint
        if self.num_meshes > 1:
            raise ValueError("`AdaptiveSteadyProblem` only supports single meshes.")
        ts = op.timestepper
        if ts != "SteadyState":
            raise ValueError("Timestepper {:s} not allowed for steady-state problems.".format(ts))
        if op.solve_tracer:
            self.equation_set = 'tracer'
            self.nonlinear = False
        elif op.solve_swe:
            self.equation_set = 'shallow_water'
            self.nonlinear = True
        else:
            raise ValueError("Steady-state solver only supports one of hydrodynamics and tracers.")

    @property
    def function_space(self):
        return self.V[0] if self.op.solve_swe else self.Q[0]

    @property
    def timestepper(self):
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
    def metric(self):
        return self.metrics[0]

    def solve_adjoint(self, **kwargs):
        if self.discrete_adjoint:
            self._solve_discrete_adjoint(**kwargs)
        else:
            self._solve_continuous_adjoint(**kwargs)

    def _solve_continuous_adjoint(self, **kwargs):
        super(AdaptiveSteadyProblem, self).solve_adjoint(**kwargs)

    def _solve_discrete_adjoint(self, **kwargs):
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
        op = self.op
        deg = op.qoi_quadrature_degree
        op.print_debug("DIAGNOSTICS: Evaluating QoI using quadrature degree {:d}".format(deg))
        if op.solve_tracer:
            return assemble(inner(self.fwd_solution_tracer, self.kernel)*dx(degree=deg))
        elif op.solve_swe:
            return assemble(inner(self.fwd_solution, self.kernel)*dx(degree=deg))
        else:
            raise NotImplementedError  # TODO

    def quantity_of_interest_form(self):
        """
        UFL form describing the quantity of interest (QoI).
        """
        deg = self.op.qoi_quadrature_degree
        op.print_debug("DIAGNOSTICS: Generating QoI form using quadrature degree {:d}".format(deg))
        return inner(self.fwd_solution, self.kernel)*dx(degree=deg)

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
        if 'dwr_adjoint' in approach:
            indicator = self.dwr_indicator(adapt_field, adjoint=True)
        elif 'dwr_avg' in approach or 'dwr_int' in approach:
            indicator = self.dwr_indicator(adapt_field, forward=True, adjoint=True)
        elif 'dwr' in approach:
            indicator = self.dwr_indicator(adapt_field, forward=True)
        else:
            raise NotImplementedError  # TODO
        self._have_indicated_error = True
        return indicator

    def dwr_indicator(self, adapt_field, forward=False, adjoint=False, mode='GE_h'):
        """
        Indicate errors in the quantity of interest by the 'Dual Weighted Residual' (DWR) method of
        [Becker and Rannacher, 2001].

        A P1 field to be used for isotropic mesh adaptation is stored as `self.indicator`.
        """
        if self._have_indicated_error:
            return

        op = self.op
        self.indicator['cell'] = Function(self.P0[0], name="DWR cell indicator")
        self.indicator['flux'] = Function(self.P0[0], name="DWR flux indicator")
        if not forward and not adjoint:
            raise ValueError("Specify either forward or adjoint model.")
        both = forward and adjoint

        if adjoint and self.discrete_adjoint:
            raise NotImplementedError  # TODO

        # Setup problem on enriched space
        if mode == 'GE_hp':
            raise NotImplementedError  # TODO: use degree_increase
        elif mode == 'GE_h':
            hierarchy = MeshHierarchy(self.mesh, 1)
            refined_mesh = hierarchy[1]
            ep = type(self)(
                op,
                meshes=refined_mesh,
                nonlinear=self.nonlinear,
                discrete_adjoint=self.discrete_adjoint,
            )
            ep.outer_iteration = self.outer_iteration
            enriched_space = ep.get_function_space(adapt_field)
            tm = dmhooks.get_transfer_manager(self.get_plex(0))
            bcs = self.boundary_conditions[0][adapt_field]
            cell = Function(ep.P0[0])
            flux = Function(ep.P0[0])
            fwd_solution = self.get_solutions(adapt_field, adjoint=False)[0]
            adj_solution = self.get_solutions(adapt_field, adjoint=True)[0]

            if forward:

                # Prolong forward solution and solve, if requested
                fwd_proj = Function(enriched_space[0])
                adj_error = Function(enriched_space[0])
                tm.prolong(fwd_solution, fwd_proj)
                if self.nonlinear:
                    if op.solve_enriched_forward:
                        ep.solve_forward()
                    else:
                        ep.fwd_solution.assign(fwd_proj)

                # Setup forward solver for enriched problem
                ep.create_error_estimators_step(0, adjoint=False)
                ep.setup_solver_forward_step(0)  # Needed to create timestepper
                ep.solve_adjoint()
                enriched_adj_solution = ep.get_solutions(adapt_field, adjoint=True)[0]

                # Setup forward error estimator
                ets = ep.get_timestepper(0, adapt_field, adjoint=False)
                ets.setup_error_estimator(fwd_proj, fwd_proj, adj_error, bcs)

                # Approximate adjoint error in enriched space
                tm.prolong(adj_solution, adj_error)
                adj_error *= -1
                adj_error += enriched_adj_solution

                # Compute dual weighted residual
                dwr_cell = ets.error_estimator.element_residual()
                dwr_flux = ets.error_estimator.inter_element_flux()
                dwr_flux += ets.error_estimator.boundary_flux()
                cell += dwr_cell
                flux += dwr_flux
                if both:
                    indicator_enriched = interpolate(abs(cell + flux), ep.P0[0])
                    indicator_enriched_cts = interpolate(indicator_enriched, ep.P1[0])
                    self.indicator['dwr'] = Function(self.P1[0])
                    tm.inject(indicator_enriched_cts, self.indicator['dwr'])

            if adjoint:

                # Prolong adjoint solution
                adj_proj = Function(enriched_space[0])
                fwd_error = Function(enriched_space[0])
                tm.prolong(adj_solution, adj_proj)

                # Setup adjoint solver for enriched problem
                ep.create_error_estimators_step(0, adjoint=True)
                ep.setup_solver_adjoint_step(0)  # Needed to create timestepper
                ep.solve_forward()
                enriched_fwd_solution = ep.get_solutions(adapt_field, adjoint=False)[0]

                # Setup adjoint error estimator
                ets = ep.get_timestepper(0, adapt_field, adjoint=True)
                ets.setup_error_estimator(adj_proj, adj_proj, fwd_error, bcs)

                # Approximate forward error in enriched space
                tm.prolong(fwd_solution, fwd_error)
                fwd_error *= -1
                fwd_error += enriched_fwd_solution

                # Compute dual weighted residual
                dwr_cell = ets.error_estimator.element_residual()
                dwr_flux = ets.error_estimator.inter_element_flux()
                dwr_flux += ets.error_estimator.boundary_flux()
                cell += dwr_cell
                flux += dwr_flux
                if both:
                    indicator_enriched = interpolate(abs(cell + flux), ep.P0[0])
                    indicator_enriched_cts = interpolate(indicator_enriched, ep.P1[0])
                    self.indicator['dwr_adjoint'] = Function(self.P1[0])
                    tm.inject(indicator_enriched_cts, self.indicator['dwr_adjoint'])

            if both:
                cell *= 0.5
                flux *= 0.5

            # Error indicator components on base space
            tm.inject(cell, self.indicator['cell'])
            tm.inject(flux, self.indicator['flux'])

            # Indicate error on enriched space
            indicator_enriched = Function(ep.P0[0])
            indicator_enriched.interpolate(abs(cell + flux))

            # Global error estimate
            label = 'dwr_avg' if both else 'dwr_adjoint' if adjoint else 'dwr'
            if label not in self.estimators:
                self.estimators[label] = []
            self.estimators[label].append(indicator_enriched.vector().gather().sum())

            # Project into P1 space and inject into base mesh
            indicator_enriched_cts = project(indicator_enriched, ep.P1[0])
            indicator_enriched_cts.interpolate(abs(indicator_enriched_cts))  # Ensure positive
            self.indicator[label] = Function(self.P1[0], name=label)
            tm.inject(indicator_enriched_cts, self.indicator[label])
            return self.indicator[label]

        elif mode == 'GE_p':
            raise NotImplementedError  # TODO: Use degree_increase
        elif mode == 'PR':
            raise NotImplementedError  # TODO: Use recover_zz
        elif mode == 'DQ':
            raise NotImplementedError  # TODO
        else:
            raise ValueError("Enrichment mode {:s} not recognised.".format(mode))

    def get_hessian_metric(self, adapt_field, adjoint=False, elementwise=False):
        """
        Compute an appropriate Hessian for the problem at hand. This is inherently
        problem-dependent, since the choice of field for adaptation is not universal.
        """
        if elementwise:
            from adapt_utils.adapt.recovery import recover_gradient

            sol = self.get_solutions(adapt_field, adjoint=adjoint)[0]
            if adapt_field == 'shallow_water':
                u, eta = sol.split()
                hessians = [
                    interpolate(grad(recover_gradient(u[0], self.op)), self.P0_ten[0]),
                    interpolate(grad(recover_gradient(u[1], self.op)), self.P0_ten[0]),
                    interpolate(grad(recover_gradient(eta, self.op)), self.P0_ten[0]),
                ]
                return combine_metrics(*hessians, average='avg' in self.op.adapt_field)
            else:
                return interpolate(grad(recover_gradient(sol, op=self.op)), self.P0_ten[0])
        else:
            hessian_kwargs = dict(normalise=False, enforce_constraints=False)
            hessians = self.recover_hessian_metrics(0, adjoint=adjoint, **hessian_kwargs)
            if self.op.adapt_field in ('tracer', 'sediment', 'bathymetry'):
                return hessians[0]
            else:
                return combine_metrics(*hessians, average='avg' in self.op.adapt_field)

    def get_metric(self, adapt_field, approach=None):
        """
        Compute metric associated with adaptation approach of choice.
        """
        approach = approach or self.op.approach
        adjoint = 'adjoint' in approach
        forward = 'adjoint' not in approach
        if 'dwr' in approach:
            self.indicate_error(adapt_field)
        if approach in ('dwr', 'dwr_adjoint', 'dwr_avg'):
            metric = self.get_isotropic_metric(self.op.adapt_field, approach=approach)
        elif approach in ('isotropic_dwr', 'isotropic_dwr_adjoint', 'isotropic_dwr_avg'):
            metric = self.get_isotropic_dwr_metric(forward=forward, adjoint=adjoint)
        elif approach in ('anisotropic_dwr', 'anisotropic_dwr_adjoint'):
            metric = self.get_anisotropic_dwr_metric(adjoint=adjoint)
        elif approach in ('weighted_hessian', 'weighted_hessian_adjoint'):
            metric = self.get_weighted_hessian_metric(adjoint=adjoint)
        elif approach in ('weighted_gradient', 'weighted_gradient_adjoint'):
            metric = self.get_weighted_gradient_metric(adjoint=adjoint)
        elif approach[-3:] in ('int', 'avg'):
            fwd_metric = self.get_metric(adapt_field, approach[:-4])
            adj_metric = self.get_metric(adapt_field, approach[:-3] + 'adjoint')
            metric = combine_metrics(fwd_metric, adj_metric, average=approach[-3:] == 'avg')
        else:
            raise NotImplementedError  # TODO
        if self.op.plot_pvd:
            File(os.path.join(self.di, 'metric.pvd')).write(metric)
        return metric

    def get_isotropic_metric(self, adapt_field, approach=None):
        """
        Scale an identity matrix by the indicator field in order to drive
        isotropic mesh refinement.
        """
        approach = approach or self.op.approach
        indicator = self.indicator[approach]
        if self.op.plot_pvd:
            File(os.path.join(self.di, 'indicator.pvd')).write(indicator)
        metric = Function(self.P1_ten[0], name="Metric")
        metric.assign(isotropic_metric(indicator, normalise=True, op=self.op))
        return metric

    def get_weighted_hessian_metric(self, adjoint=False, average=True):
        """
        Construct an anisotropic metric using an approach inspired by [Power et al. 2006].

        If `adjoint` mode is turned off, weight the Hessian of the adjoint solution with a residual
        for the forward PDE. Otherwise, weight the Hessian of the forward solution with a residual
        for the adjoint PDE.
        """

        # Compute strong residual
        strong_residual_cts = self.get_strong_residual(0, adjoint=adjoint)

        # Recover Hessian
        kwargs = dict(normalise=False, enforce_constraints=False)
        hessians = self.recover_hessian_metrics(0, adjoint=not adjoint, **kwargs)

        # Weight
        kwargs = dict(normalise=True, enforce_constraints=True, op=self.op, V=self.P1_ten[0])
        metrics = [
            steady_metric(H=abs(res)*H, **kwargs) for res, H in zip(strong_residual_cts, hessians)
        ]

        # Combine
        return metrics[0] if len(metrics) == 1 else combine_metrics(*metrics, average=average)

    def get_weighted_gradient_metric(self, adjoint=False, **kwargs):
        """
        Construct an anisotropic metric using an approach inspired by [Loseille et al. 2010].
        """
        if adjoint:
            self.get_adjoint_weighted_gradient_metric(self, **kwargs)
        else:
            self.get_forward_weighted_gradient_metric(self, **kwargs)

    def get_forward_weighted_gradient_metric(self, source=True):
        from adapt_utils.adapt.recovery import recover_gradient, recover_boundary_hessian

        op = self.op
        mesh = self.mesh
        dim = mesh.topological_dimension()
        dims = range(dim)
        P1_ten = self.P1_ten[0]
        if op.adapt_field != 'tracer':
            raise NotImplementedError

        c = self.fwd_solution_tracer
        c_star = self.adj_solution_tracer

        # Interior gradient term
        grad_c_star = recover_gradient(c_star, op=op)

        # Interior Hessian term
        u, eta = split(self.fwd_solutions[0])
        D = self.fields[0].horizontal_diffusivity
        F = [u[i]*c - D*c.dx(i) for i in dims]
        kwargs = dict(normalise=True, noscale=True, enforce_constraints=False, mesh=mesh, op=op)
        interior_hessians = [
            interpolate(steady_metric(F[i], **kwargs)*abs(grad_c_star[i]), P1_ten) for i in dims
        ]

        # Interior source Hessian term
        if source:
            S = self.fields[0].tracer_source_2d
            interior_hessians.append(interpolate(steady_metric(S, **kwargs)*abs(c_star), P1_ten))

        # Average interior Hessians
        interior_hessian = metric_average(*interior_hessians)

        # Boundary Hessian
        n = FacetNormal(mesh)
        Fbar = -D*dot(grad(c), n)  # NOTE: Minus zero (imposed boundary value)
        # TODO: weakly imposed Dirichlet conditions
        bcs = self.boundary_conditions[0]['tracer']
        tags = [tag for tag in bcs if 'diff_flux' in bcs[tag]]
        H = recover_boundary_hessian(Fbar, mesh=mesh, op=op, boundary_tag=tags)
        boundary_hessian = abs(c_star)*abs(H)

        # Get target complexities based on interior and boundary Hessians
        C = volume_and_surface_contributions(interior_hessian, boundary_hessian, op=op)

        # Assemble and combine metrics
        kwargs = dict(normalise=True, enforce_constraints=True, mesh=mesh, integral=C, op=op)
        interior_metric = steady_metric(H=interior_hessian, **kwargs)
        boundary_metric = boundary_steady_metric(boundary_hessian, **kwargs)
        return metric_intersection(interior_metric, boundary_metric, boundary_tag=tags)

    def get_adjoint_weighted_gradient_metric(self, source=True):
        from adapt_utils.adapt.recovery import recover_gradient, recover_boundary_hessian

        op = self.op
        mesh = self.mesh
        dim = mesh.topological_dimension()
        dims = range(dim)
        P1_ten = self.P1_ten[0]
        if op.adapt_field != 'tracer':
            raise NotImplementedError

        c = self.fwd_solution_tracer
        c_star = self.adj_solution_tracer

        # Interior gradient term
        grad_c = recover_gradient(c, op=op)

        # Interior Hessian term
        u, eta = split(self.fwd_solutions[0])
        D = self.fields[0].horizontal_diffusivity
        F = [-u[i]*c_star - D*c_star.dx(i) for i in dims]
        kwargs = dict(normalise=True, noscale=True, enforce_constraints=False, mesh=mesh, op=op)
        interior_hessians = [
            interpolate(steady_metric(F[i], **kwargs)*abs(grad_c[i]), P1_ten) for i in dims
        ]

        # Interior source Hessian term
        if source:
            S = self.kernels_tracer[i]
            interior_hessians.append(interpolate(steady_metric(S, **kwargs)*abs(c), P1_ten))

        # Average interior Hessians
        interior_hessian = metric_average(*interior_hessians)

        # Boundary Hessian
        n = FacetNormal(mesh)
        Fbar = -(D*dot(grad(c_star), n) + c_star*dot(u, n))
        # TODO: weakly imposed Dirichlet conditions
        bcs = self.boundary_conditions[0]['tracer']
        tags = [tag for tag in bcs if 'value' not in bcs[tag]]
        H = recover_boundary_hessian(Fbar, mesh=mesh, op=op, boundary_tag=tags)
        boundary_hessian = abs(c)*abs(H)

        # Get target complexities based on interior and boundary Hessians
        C = volume_and_surface_contributions(interior_hessian, boundary_hessian, op=op)

        # Assemble and combine metrics
        kwargs = dict(normalise=True, enforce_constraints=True, mesh=mesh, integral=C, op=op)
        interior_metric = steady_metric(H=interior_hessian, **kwargs)
        boundary_metric = boundary_steady_metric(boundary_hessian, **kwargs)
        return metric_intersection(interior_metric, boundary_metric, boundary_tag=tags)

    def get_isotropic_dwr_metric(self, forward=False, adjoint=False):
        """
        Construct an isotropic metric using an approach inspired by [Carpio et al. 2013].
        """
        adapt_field = self.op.adapt_field
        if adapt_field not in ('tracer', 'sediment', 'bathymetry'):
            adapt_field = 'shallow_water'
        if not forward and not adjoint:
            raise ValueError("Must specify forward or adjoint.")
        approach = 'dwr_avg' if forward and adjoint else 'dwr_adjoint' if adjoint else 'dwr'

        # Compute error indicators
        self.indicate_error(adapt_field, approach=approach)
        dwr = self.indicator[approach]

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

        # Build metric
        indicator = project(K_hat/K_opt, self.P1[0])
        kwargs = dict(mesh=self.mesh, normalise=True, enforce_constrants=True, op=self.op)
        return isotropic_metric(indicator, **kwargs)

    def get_anisotropic_dwr_metric(self, adjoint=False):
        """
        Construct an anisotropic metric using an approach inspired by [Carpio et al. 2013].
        """
        dim = self.mesh.topological_dimension()
        adapt_field = self.op.adapt_field
        if adapt_field not in ('tracer', 'sediment', 'bathymetry'):
            adapt_field = 'shallow_water'
        approach = 'dwr_adjoint' if adjoint else 'dwr'

        # Compute error indicators
        self.indicate_error(adapt_field, approach=approach)
        dwr = self.indicator[approach]

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
        H = self.get_hessian_metric(adapt_field, adjoint=adjoint, elementwise=True)
        evectors = Function(self.P0_ten[0], name="Elementwise Hessian eigenvectors")
        evalues = Function(self.P0_vec[0], name="Elementwise Hessian eigenvalues")
        kernel = eigen_kernel(get_reordered_eigendecomposition, dim)
        op2.par_loop(kernel, self.P0_ten[0].node_set, evectors.dat(op2.RW), evalues.dat(op2.RW), H.dat(op2.READ))

        # Compute stretching factors, in descending order
        if dim == 2:
            S = as_vector([
                sqrt(abs(evalues[0]/evalues[1])),
                sqrt(abs(evalues[1]/evalues[0])),
            ])
        else:
            S = as_vector([
                pow(abs((evalues[0]*evalues[0])/(evalues[1]*evalues[2])), 1/3),
                pow(abs((evalues[1]*evalues[1])/(evalues[2]*evalues[0])), 1/3),
                pow(abs((evalues[2]*evalues[2])/(evalues[0]*evalues[1])), 1/3),
            ])

        # Build metric
        M = Function(self.P0_ten[0], name="Elementwise metric")
        evalues.interpolate(abs(K_hat/K_opt)*S)
        kernel = eigen_kernel(set_eigendecomposition, dim)
        op2.par_loop(kernel, self.P0_ten[0].node_set, M.dat(op2.RW), evectors.dat(op2.READ), evalues.dat(op2.READ))

        # Project metric
        M_p1 = Function(self.P1_ten[0], name="Anisotropic DWR metric")
        M_p1.project(M)
        return M_p1

    def log_complexities(self):
        """
        Log metric complexity.
        """
        self.print("\nRiemannian metric\n=================")
        self.print("  complexity {:13.4e}".format(metric_complexity(self.metrics[0])))

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
        if 'dwr' in op.approach:
            self.estimators['dwr'] = []
        for n in range(op.max_adapt):
            self._have_indicated_error = False
            self.outer_iteration = n
            self.create_error_estimators_step(0, adjoint='adjoint' in op.approach)

            # Solve forward in base space
            self.solve_forward()

            # Check convergence
            if (self.qoi_converged or self.maximum_adaptations_met) and self.minimum_adaptations_met:
                break

            # Solve adjoint equation in base space
            self.solve_adjoint()

            # Construct metric
            self.metrics[0] = self.get_metric(adapt_field)
            self.log_complexities()

            # Check convergence of error estimator
            if self.estimator_converged:
                break

            # Adapt mesh
            self.adapt_meshes()

            # Check convergence of element count
            if not self.minimum_adaptations_met:
                continue
            if self.elements_converged:
                break

    def run_no_dwr(self, **kwargs):
        self.run_dwr(**kwargs)
