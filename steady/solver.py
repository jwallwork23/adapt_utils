from thetis import *

import numpy as np
import os

from adapt_utils.adapt.adaptation import pragmatic_adapt
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

    def get_strong_residual(self, adjoint=False, **kwargs):
        """
        Compute the strong residual for the forward or adjoint PDE, as specified by the `adjoint`
        boolean kwarg.
        """
        raise NotImplementedError  # TODO

    def get_flux(self, adjoint=False, **kwargs):
        """
        Evaluate flux terms for forward or adjoint PDE, as specified by the `adjoint` boolean kwarg.
        """
        raise NotImplementedError  # TODO

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

    def get_strong_residual_forward(self, norm_type=None):
        raise NotImplementedError  # TODO

    def get_strong_residual_adjoint(self, norm_type=None):
        raise NotImplementedError  # TODO

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

    def dwr_indication(self, adjoint=False):
        """
        Indicate errors in the quantity of interest by the 'Dual Weighted Residual' (DWR) method of
        [Becker and Rannacher, 2001].

        A P1 field to be used for isotropic mesh adaptation is stored as `self.indicator`.
        """
        raise NotImplementedError  # TODO

    def dwp_indication(self):
        """
        Indicate significance by the product of forward and adjoint solutions. This approach was
        used for mesh adaptive tsunami modelling in [Davis and LeVeque, 2016]. Here 'DWP' is used
        to stand for 'Dual Weighted Primal'.
        """
        raise NotImplementedError  # TODO

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

    def get_isotropic_metric(self):
        """
        Scale an identity matrix by the indicator field in order to drive
        isotropic mesh refinement.
        """
        raise NotImplementedError  # TODO

    def get_loseille_metric(self, adjoint=False, relax=True):
        """
        Construct an anisotropic metric using an approach inspired by [Loseille et al. 2009].
        """
        raise NotImplementedError  # TODO

    def get_power_metric(self, adjoint=False):
        """
        Construct an anisotropic metric using an approach inspired by [Power et al. 2006].

        If `adjoint` mode is turned off, weight the Hessian of the adjoint solution with a residual
        for the forward PDE. Otherwise, weight the Hessian of the forward solution with a residual
        for the adjoint PDE.
        """
        raise NotImplementedError  # TODO

    @property
    def indicator(self):
        return self.indicators[0]

    def run_dwr(self, **kwargs):
        # TODO: doc
        op = self.op
        adapt_field = op.adapt_field
        if adapt_field not in ('tracer', 'sediment', 'bathymetry'):
            adapt_field = 'shallow_water'
        for n in range(op.max_adapt):
            self.outer_iteration = n
            fwd_solution = self.get_solutions(adapt_field)[0]
            adj_solution = self.get_solutions(adapt_field, adjoint=True)[0]


            # --- Setup problem on enriched space

            hierarchy = MeshHierarchy(self.mesh, 1)
            refined_mesh = hierarchy[1]
            ep = type(self)(
                op,
                meshes=refined_mesh,
                nonlinear=self.nonlinear,
            )
            ep.outer_iteration = n
            enriched_space = ep.get_function_space(adapt_field)

            # Create indicator and metric
            self.indicator['dwr'] = Function(self.P1[0], name="DWR indicator")
            metric = Function(self.P1_ten[0], name="Metric")

            tm = dmhooks.get_transfer_manager(self.get_plex(0))

            # Setup forward solver for enriched problem
            ep.create_error_estimators_step(0)
            ep.setup_solver_forward_step(0)
            ets = ep.timesteppers[0][adapt_field]
            enriched_adj_solution = ep.get_solutions(adapt_field, adjoint=True)[0]


            # --- Solve forward

            self.solve_forward()

            # Check QoI convergence
            qoi = self.quantity_of_interest()
            self.print("Quantity of interest {:d}: {:.4e}".format(n+1, qoi))
            self.qois.append(qoi)
            if len(self.qois) > 1:
                if np.abs(self.qois[-1] - self.qois[-2]) < op.qoi_rtol*self.qois[-2]:
                    self.print("Converged quantity of interest!")
                    break

            # Check maximum number of iterations
            if n == op.max_adapt - 1:
                break


            # --- Solve adjoint

            self.solve_adjoint()  # Base space
            ep.solve_adjoint()    # Enriched space


            # --- Assemble indicator and metric

            indicator_enriched = Function(ep.P0[0])
            fwd_proj = Function(enriched_space[0])
            adj_error = Function(enriched_space[0])
            bcs = self.boundary_conditions[0][adapt_field]
            ets.setup_error_estimator(fwd_proj, fwd_proj, adj_error, bcs)

            # Prolong forward solution at current timestep
            tm.prolong(fwd_solution, fwd_proj)

            # Approximate adjoint error in enriched space
            tm.prolong(adj_solution, adj_error)
            adj_error *= -1
            adj_error += enriched_adj_solution

            # Compute dual weighted residual
            indicator_enriched.interpolate(abs(ets.error_estimator.weighted_residual()))
            indicator_enriched_cts = interpolate(indicator_enriched, ep.P1[0])  # TODO: Project?
            tm.inject(indicator_enriched_cts, self.indicator['dwr'])

            # Construct metric
            metric.assign(isotropic_metric(self.indicator['dwr'], normalise=True))

            # Output to .pvd and .vtu
            if op.plot_pvd:
                File(os.path.join(self.di, 'indicator.pvd')).write(self.indicator['dwr'])
                metric_file = File(os.path.join(self.di, 'metric.pvd')).write(metric)

            # TODO: Log complexities


            # --- Adapt mesh

            self.print("\nStarting mesh adaptation for iteration {:d}...".format(n+1))
            self.meshes[0] = pragmatic_adapt(self.mesh, metric)


            # --- Setup for next run / logging

            self.set_meshes(self.mesh)
            self.setup_all()
            self.dofs.append(np.sum(self.get_function_space(adapt_field)[0].dof_count))

            # Print to screen
            msg = "\nResulting mesh: {:7d} vertices, {:7d} elements"
            num_cells = self.num_cells
            self.print(msg.format(self.num_vertices[-1][0], num_cells[-1][0]))

            # Check convergence of element count
            if np.abs(num_cells[-1][0] - num_cells[-2][0]) > op.element_rtol*num_cells[-2][0]:
                self.print("Converged number of mesh elements!")
                break
