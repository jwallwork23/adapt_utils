from thetis import *

from adapt_utils.swe.solver import AdaptiveShallowWaterProblem


__all__ = ["AdaptiveTurbineProblem"]


class AdaptiveTurbineProblem(AdaptiveShallowWaterProblem):
    """General solver object for adaptive tidal turbine problems."""

    # --- Setup

    # TODO: Hook up turbines

    # --- Goal-oriented

    def quantity_of_interest(self):
        self.qoi = self.cb.average_power
        return self.qoi

    def get_qoi_kernel(self):
        self.kernel = Function(self.V)
        u = self.solution.split()[0]
        k_u = self.kernel.split()[0]
        k_u.interpolate(Constant(1/3)*self.turbine_density*sqrt(inner(u, u))*u)

    def quantity_of_interest_form(self):
        return self.C_D*pow(inner(split(self.solution)[0], split(self.solution)[0]), 1.5)*dx
