from thetis import *

# import numpy as np
# import os

# from adapt_utils.adapt.adaptation import pragmatic_adapt
# from adapt_utils.adapt.metric import *
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
