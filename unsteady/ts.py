from thetis.utility import *
import thetis.timeintegrator as thetis_ts


__all__ = ["SteadyState", "CrankNicolson"]


class SteadyState(thetis_ts.SteadyState):
    """
    Extension of Thetis SteadyState time integrator for error estimation.

    See `thetis/timeintegrator.py` for original version.
    """
    def __init__(self, equation, solution, fields, dt, error_estimator=None, **kwargs):
        super(SteadyState, self).__init__(equation, solution, fields, dt, **kwargs)
        self.error_estimator = error_estimator
        if self.error_estimator is not None:
            if hasattr(self.error_estimator, 'setup_strong_residual'):
                self.error_estimator.setup_strong_residual('all', solution, solution, fields, fields)

    def setup_error_estimator(self, solution, adjoint, bnd_conditions):
        assert self.error_estimator is not None
        u = solution
        f = self.fields
        bnd = bnd_conditions
        self.error_estimator.setup_components('all', u, u, z, z, f, f, bnd)


class CrankNicolson(thetis_ts.CrankNicolson):
    """
    Extension of Thetis CrankNicolson time integrator for error estimation.

    See `thetis/timeintegrator.py` for original version.
    """
    def __init__(self, *args, error_estimator=None, adjoint=False, **kwargs):
        super(CrankNicolson, self).__init__(*args, **kwargs)
        self.semi_implicit = kwargs.get('semi_implicit')
        self.theta = kwargs.get('theta')
        self.adjoint = adjoint
        self.error_estimator = error_estimator
        print_output("#### TODO: Setup strong residual for Crank-Nicolson")  # TODO

    def advance(self, t, update_forcings=None):
        """Advances equations for one time step."""
        if update_forcings is not None:
            update_forcings(t - self.dt if self.adjoint else t + self.dt)
        self.solution_old.assign(self.solution)
        self.solver.solve()
        for k in sorted(self.fields_old):
            self.fields_old[k].assign(self.fields[k])

    def setup_error_estimator(self, solution, solution_old, adjoint, bnd_conditions):
        ee = self.error_estimator
        assert ee is not None
        u = solution
        u_old = solution_old
        u_nl = u_old if self.semi_implicit else u
        z = adjoint
        f = self.fields
        f_old = self.fields_old
        bnd = bnd_conditions
        theta_const = Constant(self.theta)

        # The three forms
        residual_terms = 0
        inter_element_flux_terms = 0
        bnd_flux_terms = 0

        # Time derivative
        residual_terms += ee.mass_term(u, z)
        residual_terms += -ee.mass_term(u_old, z)

        # Term from current timestep
        ee.setup_components('all', u, u_nl, z, z, f, f, bnd)
        residual_terms += -self.dt_const*theta_const*ee.residual_terms
        inter_element_flux_terms += -self.dt_const*theta_const*ee.inter_element_flux_terms
        bnd_flux_terms += -self.dt_const*theta_const*ee.bnd_flux_terms

        # Term from previous timestep
        ee.setup_components('all', u_old, u_old, z, z, f_old, f_old, bnd)
        residual_terms += -self.dt_const*(1-theta_const)*ee.residual_terms
        inter_element_flux_terms += -self.dt_const*(1-theta_const)*ee.inter_element_flux_terms
        bnd_flux_terms += -self.dt_const*(1-theta_const)*ee.bnd_flux_terms

        # Pass forms back to error estimator
        ee.residual_terms = residual_terms
        ee.inter_element_flux_terms = inter_element_flux_terms
        ee.bnd_flux_terms = bnd_flux_terms
