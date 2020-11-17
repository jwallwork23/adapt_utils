"""
Extend some of Thetis' time integration schemes to incorporate goal-oriented error estimation
functionality.

**********************************************************************************************
*  NOTE: This file is based on the Thetis project (https://thetisproject.org) and contains   *
*        some copied code.                                                                   *
**********************************************************************************************
"""
from thetis.utility import *
import thetis.timeintegrator as thetis_ts

CFL_UNCONDITIONALLY_STABLE = np.inf


__all__ = ["SteadyState", "CrankNicolson"]


class SteadyState(thetis_ts.SteadyState):
    """
    Extension of Thetis SteadyState time integrator for error estimation.

    See `thetis/timeintegrator.py` for original version.
    """
    def __init__(self, equation, solution, fields, dt, error_estimator=None, **kwargs):
        if 'adjoint' in kwargs:
            try:
                fields.uv_2d = -fields.uv_2d
            except AttributeError:
                fields.uv_3d = -fields.uv_3d
            kwargs.pop('adjoint')  # Unused in steady-state case
        super(SteadyState, self).__init__(equation, solution, fields, dt, **kwargs)
        self.error_estimator = error_estimator
        if self.error_estimator is not None:
            if hasattr(self.error_estimator, 'setup_strong_residual'):
                self.error_estimator.setup_strong_residual('all', solution, solution, fields, fields)

    def setup_error_estimator(self, solution, solution_old, adjoint, bnd_conditions):
        assert self.error_estimator is not None
        self.error_estimator.setup_components(
            'all', solution, solution, adjoint, adjoint, self.fields, self.fields, bnd_conditions
        )


class CrankNicolson(thetis_ts.TimeIntegrator):
    """
    Extension of Thetis CrankNicolson time integrator for error estimation.

    *** LARGELY COPIED FROM `thetis/timeintegrator.py` ***
    """
    cfl_coeff = CFL_UNCONDITIONALLY_STABLE

    def __init__(self, equation, solution, fields, dt, bnd_conditions=None, solver_parameters={},
                 theta=0.5, semi_implicit=False, error_estimator=None, adjoint=False):
        super(CrankNicolson, self).__init__(equation, solution, fields, dt, solver_parameters)
        self.semi_implicit = semi_implicit
        self.theta_const = Constant(theta)
        self.adjoint = adjoint
        self.error_estimator = error_estimator
        if semi_implicit:
            self.solver_parameters.setdefault('snes_type', 'ksponly')
        else:
            self.solver_parameters.setdefault('snes_type', 'newtonls')
        self.solution_old = Function(self.equation.function_space, name='solution_old')

        # Create functions to hold the values of previous time step
        self.fields_old = {}
        for k in sorted(self.fields):
            if self.fields[k] is not None:
                if isinstance(self.fields[k], Function):
                    self.fields_old[k] = Function(
                        self.fields[k].function_space(), name=self.fields[k].name() + '_old')
                elif isinstance(self.fields[k], Constant):
                    self.fields_old[k] = Constant(self.fields[k])

        # Get fields etc.
        u = self.solution
        u_old = self.solution_old
        u_nl = u_old if semi_implicit else u
        bnd = bnd_conditions
        f = self.fields
        f_old = self.fields_old
        kwargs = {}
        if 'Tracer' in equation.__class__.__name__:
            uv = f_old.get('uv_2d', None)
            uv = f_old.get('uv_3d', uv)
            kwargs['velocity'] = uv

        # Crank-Nicolson
        self.F = (self.equation.mass_term(u, **kwargs)
                  - self.equation.mass_term(u_old, **kwargs)
                  - self.dt_const*(self.theta_const*self.equation.residual('all', u, u_nl, f, f, bnd)
                                   + (1-self.theta_const)*self.equation.residual('all', u_old, u_old, f_old, f_old, bnd))
                  )

        self.update_solver()

    def update_solver(self):
        """
        Create solver objects
        """
        # Ensure LU assembles monolithic matrices
        if self.solver_parameters.get('pc_type') == 'lu':
            self.solver_parameters['mat_type'] = 'aij'
        prob = NonlinearVariationalProblem(self.F, self.solution)
        self.solver = NonlinearVariationalSolver(prob,
                                                 solver_parameters=self.solver_parameters,
                                                 options_prefix=self.name)

    def initialize(self, solution):
        """
        Assigns initial conditions to all required fields.
        """
        self.solution_old.assign(solution)
        # assign values to old functions
        for k in sorted(self.fields_old):
            self.fields_old[k].assign(self.fields[k])

    def advance(self, t, update_forcings=None):
        """
        Advances equations for one time step.
        """
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

        # The three forms
        residual_terms = 0
        inter_element_flux_terms = 0
        bnd_flux_terms = 0

        # Time derivative
        kwargs = {}
        if 'Tracer' in self.equation.__class__.__name__:
            uv = f_old.get('uv_2d', None)
            uv = f_old.get('uv_3d', uv)
            kwargs['velocity'] = uv
        residual_terms += ee.mass_term(u, z, **kwargs)
        residual_terms += -ee.mass_term(u_old, z, **kwargs)

        # Term from current timestep
        ee.setup_components('all', u, u_nl, z, z, f, f, bnd)
        residual_terms += -self.dt_const*self.theta_const*ee.residual_terms
        inter_element_flux_terms += -self.dt_const*self.theta_const*ee.inter_element_flux_terms
        bnd_flux_terms += -self.dt_const*self.theta_const*ee.bnd_flux_terms

        # Term from previous timestep
        ee.setup_components('all', u_old, u_old, z, z, f_old, f_old, bnd)
        residual_terms += -self.dt_const*(1-self.theta_const)*ee.residual_terms
        inter_element_flux_terms += -self.dt_const*(1-self.theta_const)*ee.inter_element_flux_terms
        bnd_flux_terms += -self.dt_const*(1-self.theta_const)*ee.bnd_flux_terms

        # Pass forms back to error estimator
        ee.residual_terms = residual_terms
        ee.inter_element_flux_terms = inter_element_flux_terms
        ee.bnd_flux_terms = bnd_flux_terms

    def setup_strong_residual(self, solution, solution_old):  # TODO: Account for non-scalar spaces
        ee = self.error_estimator
        assert ee is not None
        u = solution
        u_old = solution_old
        u_nl = u_old if self.semi_implicit else u
        f = self.fields
        f_old = self.fields_old

        # Time derivative
        kwargs = {}
        if 'Tracer' in self.equation.__class__.__name__:
            uv = f_old.get('uv_2d', None)
            uv = f_old.get('uv_3d', uv)
            kwargs['velocity'] = uv
        one = Function(solution.function_space()).assign(1.0)
        residual = ee.mass_term(u, one, **kwargs)
        residual += -ee.mass_term(u_old, one, **kwargs)

        # Term from current timestep
        ee.setup_strong_residual('all', u, u_nl, f, f)
        residual += -self.dt_const*self.theta_const*ee.strong_residual_terms

        # Term from previous timestep
        ee.setup_strong_residual('all', u_old, u_old, f_old, f_old)
        residual += -self.dt_const*(1-self.theta_const)*ee.strong_residual_terms

        # Pass forms back to error estimator
        ee.strong_residual_termss = residual
