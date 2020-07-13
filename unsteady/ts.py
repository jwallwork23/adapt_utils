from __future__ import absolute_import
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
    def __init__(self, equation, solution, fields, dt, bnd_conditions=None, solver_parameters={}, theta=0.5, semi_implicit=False, error_estimator=None, adjoint=False):
        """
        :arg equation: the equation to solve
        :type equation: :class:`Equation` object
        :arg solution: :class:`Function` where solution will be stored
        :arg fields: Dictionary of fields that are passed to the equation
        :type fields: dict of :class:`Function` or :class:`Constant` objects
        :arg float dt: time step in seconds
        :kwarg dict bnd_conditions: Dictionary of boundary conditions passed to the equation
        :kwarg dict solver_parameters: PETSc solver options
        :kwarg float theta: Implicitness parameter, default 0.5
        :kwarg bool semi_implicit: If True use a linearized semi-implicit scheme
        """
        super(thetis_ts.CrankNicolson, self).__init__(equation, solution, fields, dt, solver_parameters)
        self.semi_implicit = semi_implicit
        self.theta = theta

        self.solver_parameters.setdefault('snes_type', 'ksponly' if semi_implicit else 'newtonls')
        self.solution_old = Function(self.equation.function_space, name='solution_old')
        # create functions to hold the values of previous time step
        # TODO is this necessary? is self.fields sufficient?
        self.fields_old = {}
        for k in sorted(self.fields):
            if self.fields[k] is not None:
                if isinstance(self.fields[k], Function):
                    self.fields_old[k] = Function(
                        self.fields[k].function_space(), name=self.fields[k].name()+'_old')
                elif isinstance(self.fields[k], Constant):
                    self.fields_old[k] = Constant(self.fields[k])

        u = self.solution
        u_old = self.solution_old
        u_nl = u_old if semi_implicit else u
        bnd = bnd_conditions
        f = self.fields
        f_old = self.fields_old

        # Crank-Nicolson
        theta_const = Constant(theta)
        self.F = self.equation.mass_term(u) - self.equation.mass_term(u_old)
        if adjoint:
            self.F = -self.F  # Account for reversed time direction in adjoint
        self.F += -self.dt_const*theta_const*self.equation.residual('all', u, u_nl, f, f, bnd)
        self.F += -self.dt_const*(1-theta_const)*self.equation.residual('all', u_old, u_old, f_old, f_old, bnd)
        self.update_solver()

        # Setup error estimator
        self.error_estimator = error_estimator
        print_output("#### TODO: Setup strong residual for Crank-Nicolson")  # TODO

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
