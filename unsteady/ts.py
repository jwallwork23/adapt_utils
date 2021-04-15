<<<<<<< HEAD
=======
"""
Extend some of Thetis' time integration schemes to incorporate goal-oriented error estimation
functionality.

**********************************************************************************************
*  NOTE: This file is based on the Thetis project (https://thetisproject.org) and contains   *
*        some copied code.                                                                   *
**********************************************************************************************
"""
>>>>>>> dfe1c0b3a34dfef1765835b64b574a69fe60dd9a
from thetis.utility import *
import thetis.timeintegrator as thetis_ts


<<<<<<< HEAD
__all__ = ["SteadyState", "CrankNicolson"]
=======
__all__ = ["SteadyState", "CrankNicolson", "PressureProjectionPicard"]
>>>>>>> dfe1c0b3a34dfef1765835b64b574a69fe60dd9a


class SteadyState(thetis_ts.SteadyState):
    """
    Extension of Thetis SteadyState time integrator for error estimation.

    See `thetis/timeintegrator.py` for original version.
    """
    def __init__(self, equation, solution, fields, dt, error_estimator=None, **kwargs):
<<<<<<< HEAD
=======
        if 'adjoint' in kwargs:
            try:
                fields.uv_2d = -fields.uv_2d
            except AttributeError:
                fields.uv_3d = -fields.uv_3d
            kwargs.pop('adjoint')  # Unused in steady-state case
>>>>>>> dfe1c0b3a34dfef1765835b64b574a69fe60dd9a
        super(SteadyState, self).__init__(equation, solution, fields, dt, **kwargs)
        self.error_estimator = error_estimator
        if self.error_estimator is not None:
            if hasattr(self.error_estimator, 'setup_strong_residual'):
                self.error_estimator.setup_strong_residual('all', solution, solution, fields, fields)

<<<<<<< HEAD
    def setup_error_estimator(self, solution, adjoint, bnd_conditions):
        assert self.error_estimator is not None
        u = solution
        f = self.fields
        bnd = bnd_conditions
        self.error_estimator.setup_components('all', u, u, z, z, f, f, bnd)
=======
    def setup_error_estimator(self, solution, solution_old, adjoint, bnd_conditions):
        assert self.error_estimator is not None
        self.error_estimator.setup_components(
            'all', solution, solution, adjoint, adjoint, self.fields, self.fields, bnd_conditions
        )
>>>>>>> dfe1c0b3a34dfef1765835b64b574a69fe60dd9a


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
<<<<<<< HEAD
        print_output("#### TODO: Setup strong residual for Crank-Nicolson")  # TODO
=======
        # TODO: Setup strong residual for Crank-Nicolson
>>>>>>> dfe1c0b3a34dfef1765835b64b574a69fe60dd9a

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
<<<<<<< HEAD
=======


class PressureProjectionPicard(thetis_ts.PressureProjectionPicard):
    """
    Extension of Thetis PressureProjectionPicard time integrator for error estimation.

    See `thetis/timeintegrator.py` for original version.
    """
    def __init__(self, *args, error_estimator=None, adjoint=False, **kwargs):
        super(PressureProjectionPicard, self).__init__(*args, **kwargs)
        self.semi_implicit = kwargs.get('semi_implicit')
        self.theta = kwargs.get('theta')
        self.adjoint = adjoint
        if adjoint:
            raise NotImplementedError
        self.error_estimator = error_estimator
        if error_estimator is not None:
            raise NotImplementedError
        # TODO: Setup strong residual for Picard iteration
        # TODO: Setup error estimators for Picard iteration
>>>>>>> dfe1c0b3a34dfef1765835b64b574a69fe60dd9a
