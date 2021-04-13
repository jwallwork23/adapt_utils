from thetis import *

import os

from adapt_utils.adapt.kernels import *
from adapt_utils.adapt.metric import *
from adapt_utils.adapt.recovery import *
from adapt_utils.params import flux_params
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
        if op.solve_tracer:
            self.equation_set = 'tracer'
            kwargs.setdefault('nonlinear', False)
        elif op.solve_swe:
            self.equation_set = 'shallow_water'
            kwargs.setdefault('nonlinear', True)
        else:
            raise ValueError("Steady-state solver only supports one of hydrodynamics and tracers.")
        super(AdaptiveSteadyProblem, self).__init__(op, **kwargs)
        self.discrete_adjoint = discrete_adjoint
        if self.num_meshes > 1:
            raise ValueError("`AdaptiveSteadyProblem` only supports single meshes.")
        ts = op.timestepper
        if ts != "SteadyState":
            raise ValueError("Timestepper {:s} not allowed for steady-state problems.".format(ts))

    @property
    def function_space(self):
        return self.get_function_space(self.equation_set)[0]

    @property
    def timestepper(self):
        return self.timesteppers[0][self.equation_set]

    @property
    def kernel(self):
        op = self.op
        if self.op.solve_tracer:
            return op.set_qoi_kernel_tracer(self, 0)
        elif self.op.solve_swe:
            return op.set_qoi_kernel(self, 0)
        else:
            raise NotImplementedError

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
        F = self.timestepper.F
        sol = self.get_solutions(self.equation_set, adjoint=False)[0]
        fs = self.function_space

        # Take the adjoint
        dFdu = derivative(F, sol, TrialFunction(fs))
        dFdu_form = adjoint(dFdu)
        dJdu = derivative(self.quantity_of_interest_form(), sol, TestFunction(fs))

        # Account for strong boundary conditions
        bcs = self.boundary_conditions[0][self.equation_set]
        adj_bcs = []
        if self.equation_set == 'shallow_water':
            if self.op.family == 'cg':
                raise NotImplementedError  # TODO
        elif self.equation_set == 'tracer':
            if self.op.tracer_family == 'cg':
                for segment in bcs:
                    if 'diff_flux' not in bcs:
                        adj_bcs.append(DirichletBC(fs, 0, segment))
        else:
            raise NotImplementedError

        # Solve using adjoint solver parameters
        adj_sol = self.get_solutions(self.equation_set, adjoint=True)[0]
        params = self.op.adjoint_solver_parameters[self.equation_set].copy()
        params['snes_type'] = 'ksponly'
        solve(dFdu_form == dJdu, adj_sol, bcs=adj_bcs, solver_parameters=params)

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
        self.op.print_debug("DIAGNOSTICS: Generating QoI form using quadrature degree {:d}".format(deg))
        if self.op.solve_tracer:
            return inner(self.fwd_solution_tracer, self.kernel)*dx(degree=deg)
        elif self.op.solve_swe:
            return inner(self.fwd_solution, self.kernel)*dx(degree=deg)
        else:
            raise NotImplementedError  # TODO

    def indicate_error(self, field, approach=None):
        """
        Evaluate error indicator for some solution field.
        """
        op = self.op
        assert field in op.solve_fields
        approach = approach or op.approach
        if 'dwr' not in approach:
            return
        elif approach[-7:] == 'adjoint':
            indicator = self.dwr_indicator(field, forward=False, adjoint=True)
        elif approach == 'dwr_both' or approach[-3:] in ('avg', 'int'):
            indicator = self.dwr_indicator(field, forward=True, adjoint=True)
        else:
            indicator = self.dwr_indicator(field, forward=True, adjoint=False)
        self._have_indicated_error = True
        return indicator

    def dwr_indicator(self, adapt_field, forward=False, adjoint=False):
        """
        Indicate errors in the quantity of interest by the 'Dual Weighted Residual' (DWR) method of
        [Becker and Rannacher, 2001].

        A P1 field to be used for isotropic mesh adaptation is stored as `self.indicator`.
        """
        if self._have_indicated_error:
            return
        if not forward and not adjoint:
            raise ValueError("Specify either forward or adjoint model.")
        if adjoint and self.discrete_adjoint:
            raise NotImplementedError
        if self.op.enrichment_method in ('GE_hp', 'GE_h', 'GE_p', 'PR'):
            return self.dwr_indicator_GE(adapt_field, forward=forward, adjoint=adjoint)
        elif self.op.enrichment_method == 'DQ':
            return self.dwr_indicator_DQ(adapt_field, forward=forward, adjoint=adjoint)
        else:
            raise ValueError("Enrichment mode {:s} not recognised.".format(self.op.enrichment_method))

    def get_enriched_problem(self, field, **kwargs):
        """
        Generate a globally enriched version of this problem class.

        :arg field: solution field to be refined.
        """
        kwargs.setdefault('nonlinear', self.nonlinear)
        kwargs.setdefault('discrete_adjoint', self.discrete_adjoint)
        kwargs.setdefault('print_progress', self.print_progress)
        op = self.op
        mode = 'p' if op.enrichment_method == 'PR' else op.enrichment_method[3:]
        assert field in op.solve_fields
        assert mode in ('hp', 'h', 'p')
        eop = op.copy()
        if 'p' in mode:
            eop.increase_degree(field)  # Apply p-refinement
        mesh = self.mesh
        if 'h' in mode:
            mesh = MeshHierarchy(mesh, op.num_refinements)[1]  # Apply h-refinement
        ep = type(self)(eop, meshes=mesh, **kwargs)
        ep.outer_iteration = self.outer_iteration
        if 'h' in mode:
            self.transfer_manager = dmhooks.get_transfer_manager(self.plex)
            ep.transfer_manager = self.transfer_manager
        return ep

    def prolong(self, f, fs=None):
        """
        Prolong field `f` into the prognostic space.
        """
        g = Function(fs or self.function_space)
        if hasattr(self, 'transfer_manager'):
            self.transfer_manager.prolong(f, g)
        elif self.equation_set == 'shallow_water':
            fu, feta = f.split()
            gu, geta = g.split()
            gu.interpolate(fu)
            geta.interpolate(feta)
        else:
            g.interpolate(f)
        return g

    def inject(self, f, fs=None):
        """
        Inject field `f` into the prognostic space.
        """
        g = Function(fs or self.function_space)
        if hasattr(self, 'transfer_manager'):
            self.transfer_manager.inject(f, g)
        else:
            g.assign(f)
        return g

    def dwr_indicator_GE(self, adapt_field, forward=False, adjoint=False):
        """
        Indicate DWR errors using an enriched space. The type of enriched space is determined by
        :attr:`enrichment_method` of the :class:`Options` parameter object:

          * 'GE_h': iso-P2 (h-)refinement;
          * 'GE_p': increment of the polynomial order (p-refinement);
          * 'GE_hp': both of the above.

        The number of p-refinements is determined by :attr:`degree_increase`.

        :arg adapt_field: solution field to be refined.
        :kwarg forward: use forward error estimator?
        :kwarg adjoint: use adjoint error estimator?
        """
        op = self.op
        assert adapt_field in op.solve_fields
        method = op.enrichment_method
        both = forward and adjoint

        # Generate enriched space
        ep = self.get_enriched_problem(adapt_field)
        bcs = self.boundary_conditions[0][adapt_field]
        cell = Function(ep.P0[0])
        flux = Function(ep.P0[0])
        fwd_solution = self.get_solutions(adapt_field, adjoint=False)[0]
        adj_solution = self.get_solutions(adapt_field, adjoint=True)[0]

        if forward:

            # Prolong forward solution and solve, if requested
            fwd_proj = ep.prolong(fwd_solution)
            if method == 'PR':
                pass
            elif self.nonlinear:
                if op.solve_enriched_forward:
                    ep.solve_forward()
                else:
                    ep.fwd_solution.assign(fwd_proj)
            elif hasattr(ep, 'tape'):
                ep.solve_forward()

            # Setup forward solver for enriched problem
            ep.create_error_estimators_step(0, adjoint=False)
            ep.setup_solver_forward_step(0)  # Needed to create timestepper

            # Approximate adjoint error in enriched space
            if method == 'PR':
                adj_error = recover_zz(adj_solution, to_recover='field')
            else:
                ep.solve_adjoint()
                adj_error = ep.get_solutions(adapt_field, adjoint=True)[0]
            adj_error -= ep.prolong(adj_solution)

            # Setup forward error estimator
            ets = ep.get_timestepper(0, adapt_field, adjoint=False)
            ets.setup_error_estimator(fwd_proj, fwd_proj, adj_error, bcs)

            # Compute dual weighted residual
            dwr_cell = ets.error_estimator.element_residual()
            dwr_flux = ets.error_estimator.inter_element_flux()
            dwr_flux += ets.error_estimator.boundary_flux()
            cell += dwr_cell
            flux += dwr_flux
            if both:
                if 'h' in method:
                    indicator_enriched = interpolate(abs(dwr_cell + dwr_flux), ep.P0[0])  # TODO: Necessary?
                    indicator_enriched_cts = interpolate(indicator_enriched, ep.P1[0])
                    self.indicator['dwr'] = self.inject(indicator_enriched_cts, self.P1[0])
                else:
                    indicator = interpolate(abs(dwr_cell + dwr_flux), self.P0[0])
                    self.indicator['dwr'] = interpolate(indicator, self.P1[0])

        if adjoint:

            # Prolong adjoint solution
            adj_proj = ep.prolong(adj_solution)

            # Setup adjoint solver for enriched problem
            ep.create_error_estimators_step(0, adjoint=True)
            ep.setup_solver_adjoint_step(0)  # Needed to create timestepper
            if method == 'PR':
                enriched_fwd_solution = recover_zz(fwd_solution, to_recover='field')
            else:
                ep.solve_forward()
                enriched_fwd_solution = ep.get_solutions(adapt_field, adjoint=False)[0]

            # Approximate forward error in enriched space
            fwd_error = ep.prolong(fwd_solution)
            fwd_error *= -1
            fwd_error += enriched_fwd_solution

            # Setup adjoint error estimator
            ets = ep.get_timestepper(0, adapt_field, adjoint=True)
            ets.setup_error_estimator(adj_proj, adj_proj, fwd_error, bcs)

            # Compute dual weighted residual
            dwr_cell = ets.error_estimator.element_residual()
            dwr_flux = ets.error_estimator.inter_element_flux()
            dwr_flux += ets.error_estimator.boundary_flux()
            cell += dwr_cell
            flux += dwr_flux
            if both:
                if 'h' in method:
                    indicator_enriched = interpolate(abs(dwr_cell + dwr_flux), ep.P0[0])  # TODO: Necessary?
                    indicator_enriched_cts = interpolate(indicator_enriched, ep.P1[0])
                    self.indicator['dwr_adjoint'] = self.inject(indicator_enriched_cts, self.P1[0])
                else:
                    indicator = interpolate(abs(dwr_cell + dwr_flux), self.P0[0])
                    self.indicator['dwr_adjoint'] = interpolate(indicator, self.P1[0])

        if both:
            cell *= 0.5
            flux *= 0.5

        # Global error estimate
        indicator_enriched = cell.copy(deepcopy=True)
        indicator_enriched += flux
        label = 'dwr_both' if both else 'dwr_adjoint' if adjoint else 'dwr'
        if label not in self.estimators:
            self.estimators[label] = []
        self.estimators[label].append(abs(indicator_enriched.vector().gather().sum()))

        # Error indicator components
        self.indicator['cell'] = self.inject(cell, self.P0[0])
        self.indicator['flux'] = self.inject(flux, self.P0[0])
        self.indicator[method] = self.indicator['cell'].copy(deepcopy=True)
        self.indicator[method] += self.indicator['flux']
        self.indicator[method].interpolate(abs(self.indicator[method]))

        # P1 error indicator
        if 'h' in method:
            indicator_enriched = interpolate(abs(cell + flux), ep.P0[0])
            indicator_enriched_cts = project(indicator_enriched, ep.P1[0])
            indicator_enriched_cts.interpolate(abs(indicator_enriched_cts))  # Ensure positive
            self.indicator[label] = self.inject(indicator_enriched_cts, self.P1[0])
        else:
            self.indicator[label] = project(self.indicator[method], self.P1[0])
            self.indicator[label].interpolate(abs(self.indicator[label]))  # Ensure positive
        self.indicator[label].rename(label)

        return self.indicator[label]

    def dwr_indicator_DQ(self, adapt_field, forward=False, adjoint=False):
        """
        Indicate DWR errors using difference quotients.
        """
        from adapt_utils.mesh import anisotropic_cell_size

        op = self.op
        both = forward and adjoint
        if adapt_field != 'tracer' or op.tracer_family != 'cg':
            raise NotImplementedError  # TODO

        bcs = self.boundary_conditions[0][adapt_field]
        p0test = TestFunction(self.P0[0])
        p0trial = TrialFunction(self.P0[0])
        c = self.get_solutions(adapt_field, adjoint=False)[0].copy(deepcopy=True)
        c_star = self.get_solutions(adapt_field, adjoint=True)[0].copy(deepcopy=True)
        u, eta = self.fwd_solution.split()
        n = FacetNormal(self.mesh)
        D = self.fields[0].horizontal_diffusivity*Identity(2)
        h = anisotropic_cell_size(self.mesh) if op.anisotropic_stabilisation else CellSize(self.mesh)

        self.indicator['DQ'] = Function(self.P0[0])
        self.indicator['cell'] = Function(self.P0[0])
        self.indicator['flux'] = Function(self.P0[0])
        mass_term = p0test*p0trial*dx
        if forward:

            # Cell residual
            Psi = self.fields[0].tracer_source_2d - dot(u, grad(c)) + div(dot(D, grad(c)))
            cell = assemble(p0test*inner(Psi, Psi)*dx)
            self.indicator['cell'] += cell

            # Fluxes
            psi = dot(dot(D, grad(c)), n)
            psi_sq = p0test*inner(psi, psi)
            flux_terms = (psi_sq('+') + psi_sq('-'))*dS
            for seg in bcs:
                if 'diff_flux' in bcs[seg]:
                    g_N = bcs[seg]['diff_flux']
                    flux_terms += p0test*inner(psi - g_N, psi - g_N)*ds(seg)
            flux = Function(self.P0[0])
            solve(mass_term == flux_terms, flux, solver_parameters=flux_params)
            self.indicator['flux'] += flux

            # Recover Laplacian of adjoint solution  # TODO: Flux version
            if op.stabilisation == 'su':
                raise NotImplementedError
            elif op.stabilisation == 'supg':
                tau = self.tracer_options[0].supg_stabilisation
                g = recover_gradient(c_star + tau*dot(u, grad(c_star)), op=self.op)
            else:
                g = recover_gradient(c_star, op=self.op)
            laplacian = assemble(p0test*inner(div(g), div(g))*dx)

            # Sum cell and flux contributions
            self.indicator['DQ'] += interpolate(
                (sqrt(abs(cell)) + sqrt(abs(flux/h)))*sqrt(abs(laplacian)), self.P0[0])

        if adjoint:

            # Cell residual
            Psi = self.kernels_tracer[0] + div(u*c_star) + div(dot(D, grad(c_star)))
            cell = assemble(p0test*inner(Psi, Psi)*dx)
            self.indicator['cell'] += cell

            # Fluxes
            psi = dot(dot(D, grad(c_star)), n) + c_star*dot(u, n)
            psi_sq = p0test*inner(psi, psi)
            flux_terms = (psi_sq('+') + psi_sq('-'))*dS
            for seg in bcs:
                if 'value' not in bcs[seg]:
                    flux_terms += p0test*inner(psi, psi)*ds(seg)
            flux = Function(self.P0[0])
            solve(mass_term == flux_terms, flux, solver_parameters=flux_params)
            self.indicator['flux'] += flux

            # Recover Laplacian of forward solution  # TODO: Flux version
            if op.stabilisation == 'su':
                raise NotImplementedError
            elif op.stabilisation == 'supg':
                tau = self.tracer_options[0].supg_stabilisation
                g = recover_gradient(c + tau*dot(u, grad(c)), op=self.op)
            else:
                g = recover_gradient(c, op=self.op)
            laplacian = assemble(p0test*inner(div(g), div(g))*dx)

            # Sum cell and flux contributions
            self.indicator['DQ'] += interpolate(
                (sqrt(abs(cell)) + sqrt(abs(flux/h)))*sqrt(abs(laplacian)), self.P0[0])

        if both:
            self.indicator['cell'] *= 0.5
            self.indicator['flux'] *= 0.5

        # Global error estimate
        label = 'dwr_both' if both else 'dwr_adjoint' if adjoint else 'dwr'
        if label not in self.estimators:
            self.estimators[label] = []
        self.estimators[label].append(self.indicator['DQ'].vector().gather().sum())

        # Interpolate into P1 space
        self.indicator[label] = Function(self.P1[0], name=label)
        self.indicator[label].interpolate(self.indicator['DQ'])
        # self.indicator[label].project(indicator)
        # self.indicator[label].interpolate(abs(self.indicator[label]))  # Ensure positive
        return self.indicator[label]

    def get_metric(self, adapt_field, approach=None):
        """
        Compute metric associated with adaptation approach of choice.
        """
        approach = approach or self.op.approach
        adjoint = 'adjoint' in approach
        forward = 'adjoint' not in approach
        if approach in ('dwr', 'dwr_adjoint', 'dwr_both'):
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
            return self.get_adjoint_weighted_gradient_metric(self, **kwargs)
        else:
            return self.get_forward_weighted_gradient_metric(self, **kwargs)

    def get_forward_weighted_gradient_metric(self, source=True):
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
            S = self.kernels_tracer[0]
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
        approach = 'dwr_both' if forward and adjoint else 'dwr_adjoint' if adjoint else 'dwr'

        # Compute error indicators
        self.indicate_error(adapt_field, approach=approach)
        dwr = self.indicator[self.op.enrichment_method]

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
        dwr = self.indicator[self.op.enrichment_method]

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
        H = self.get_static_hessian_metric(adapt_field, adjoint=adjoint, elementwise=False)
        H = project(H, self.P0_ten[0])
        # H = self.get_static_hessian_metric(adapt_field, adjoint=adjoint, elementwise=True)
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

        # Project metric and ensure SPD
        M_p1 = Function(self.P1_ten[0], name="Anisotropic DWR metric")
        M_p1.project(M)
        M_p1.interpolate(steady_metric(H=M_p1, normalise=False, enforce_constraints=False, op=self.op))
        return M_p1

    def log_complexities(self):
        """
        Log metric complexity.
        """
        if self.print_progress:
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
        adjoint = 'adjoint' in op.approach
        both = 'avg' in op.approach or 'int' in op.approach
        for n in range(op.max_adapt):
            self._have_indicated_error = False
            self.outer_iteration = n
            self.create_error_estimators_step(0, adjoint=adjoint)
            if both:
                self.create_error_estimators_step(0, adjoint=not adjoint)

            # Solve forward in base space
            self.solve_forward(keep=not adjoint or both)

            # Check convergence
            if (self.qoi_converged or self.maximum_adaptations_met) and self.minimum_adaptations_met:
                break

            # Solve adjoint equation in base space
            self.solve_adjoint(keep=adjoint or both)

            # Indicate error
            self.indicate_error(adapt_field)

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

    def _check_element_convergence(self):
        converged = super(AdaptiveSteadyProblem, self)._check_element_convergence()
        if converged:
            if self.equation_set == 'shallow_water':
                self.solution_file.__init__(self.solution_file.filename)
                self.solution_file.write(*self.fwd_solution.split())
            elif self.equation_set == 'tracer':
                self.tracer_file.__init__(self.tracer_file.filename)
                self.tracer_file.write(self.fwd_solution_tracer)
            else:
                raise NotImplementedError
        return converged
