from __future__ import absolute_import

from thetis import *
from thetis.conservative_tracer_eq_2d import ConservativeTracerEquation2D
from thetis.limiter import VertexBasedP1DGLimiter

import os
import numpy as np

from adapt_utils.adapt.adaptation import pragmatic_adapt
from adapt_utils.adapt.metric import *
from .swe.equation import ShallowWaterEquations
from .swe.adjoint import AdjointShallowWaterEquations
from .swe.error_estimation import ShallowWaterGOErrorEstimator
from .swe.utils import *
from .tracer.equation import TracerEquation2D
from .tracer.error_estimation import TracerGOErrorEstimator
from .base import AdaptiveProblemBase


__all__ = ["AdaptiveProblem"]


# TODO SOON:
#  * Mesh movement
#    - ALE formulation
#  * Sediment
#  * Exner

# TODO LATER:
#  * Multiple tracers
#  * Turbines

class AdaptiveProblem(AdaptiveProblemBase):
    """Default model: 2D coupled shallow water + tracer transport."""
    # TODO: equations and supported terms
    def __init__(self, op, nonlinear=True, **kwargs):

        # Problem-specific options
        self.shallow_water_options = [AttrDict() for i in range(op.num_meshes)]
        static_options = {
            'use_nonlinear_equations': nonlinear,
            'element_family': op.family,
            'polynomial_degree': op.degree,
            'use_grad_div_viscosity_term': op.grad_div_viscosity,
            'use_grad_depth_viscosity_term': op.grad_depth_viscosity,
            'use_automatic_sipg_parameter': op.use_automatic_sipg_parameter,
            'use_lax_friedrichs_velocity': op.stabilisation == 'lax_friedrichs',
            'use_wetting_and_drying': op.wetting_and_drying,
            'wetting_and_drying_alpha': op.wetting_and_drying_alpha,
            # 'check_volume_conservation_2d': True,  # TODO
            'norm_smoother': op.norm_smoother,
            'sipg_parameter': None,
        }
        for i, swo in enumerate(self.shallow_water_options):
            swo.update(static_options)
            swo.tidal_turbine_farms = {}  # TODO
            if hasattr(op, 'sipg_parameter') and op.sipg_parameter is not None:
                swo['sipg_parameter'] = op.sipg_parameter
        if not nonlinear:
            for model in op.solver_parameters:
                op.solver_parameters[model]['snes_type'] = 'ksponly'
                op.adjoint_solver_parameters[model]['snes_type'] = 'ksponly'
        self.tracer_options = [AttrDict() for i in range(op.num_meshes)]
        static_options = {
            'use_automatic_sipg_parameter': op.use_automatic_sipg_parameter,
            # 'check_tracer_conservation': True,  # TODO
            'use_lax_friedrichs_tracer': op.stabilisation == 'lax_friedrichs',
            'use_limiter_for_tracers': op.use_limiter_for_tracers and op.tracer_family == 'dg',
            'sipg_parameter': None,
            'tracer_advective_velocity_factor': op.tracer_advective_velocity_factor,
            'use_tracer_conservative_form': op.use_tracer_conservative_form,
        }
        if op.use_tracer_conservative_form and op.approach == 'lagrangian':
            raise NotImplementedError  # TODO
        self.tracer_limiters = [None for i in range(op.num_meshes)]
        for i, to in enumerate(self.tracer_options):
            to.update(static_options)
            if hasattr(op, 'sipg_parameter_tracer') and op.sipg_parameter_tracer is not None:
                swo['sipg_parameter_tracer'] = op.sipg_parameter_tracer
        super(AdaptiveProblem, self).__init__(op, nonlinear=nonlinear, **kwargs)

    def create_outfiles(self):
        if self.op.solve_swe:
            self.solution_file = File(os.path.join(self.di, 'solution.pvd'))
            self.adjoint_solution_file = File(os.path.join(self.di, 'adjoint_solution.pvd'))
        if self.op.solve_tracer:
            self.tracer_file = File(os.path.join(self.di, 'tracer.pvd'))
            self.adjoint_tracer_file = File(os.path.join(self.di, 'adjoint_tracer.pvd'))

    def set_finite_elements(self):
        """
        There are three options for the shallow water mixed finite element pair:
          * Taylor-Hood (continuous Galerkin)   P2-P1      'cg-cg';
          * equal order discontinuous Galerkin  PpDG-PpDG  'dg-dg';
          * mixed continuous-discontinuous      P1DG-P2    'dg-cg'.

        There are two options for the tracer finite element:
          * Continuous Galerkin     Pp    'cg';
          * Discontinuous Galerkin  PpDG  'dg'.
        """
        p = self.op.degree
        family = self.op.family
        if family == 'cg-cg':
            assert p == 1
            u_element = VectorElement("Lagrange", triangle, p+1)
            eta_element = FiniteElement("Lagrange", triangle, p, variant='equispaced')
        elif family == 'dg-dg':
            u_element = VectorElement("DG", triangle, p)
            eta_element = FiniteElement("DG", triangle, p, variant='equispaced')
        elif family == 'dg-cg':
            assert p == 1
            u_element = VectorElement("DG", triangle, p)
            eta_element = FiniteElement("Lagrange", triangle, p+1, variant='equispaced')
        else:
            raise NotImplementedError("Cannot build order {:d} {:s} element".format(p, family))
        self.finite_element = u_element*eta_element

        if self.op.solve_tracer:
            p = self.op.degree_tracer
            family = self.op.tracer_family
            if family == 'cg':
                self.finite_element_tracer = FiniteElement("Lagrange", triangle, p)
            elif family == 'dg':
                self.finite_element_tracer = FiniteElement("DG", triangle, p)
            else:
                raise NotImplementedError("Cannot build order {:d} {:s} element".format(p, family))

    def create_function_spaces(self):
        """
        Build finite element spaces `V` and `Q`, for the prognostic solutions of the shallow water
        and tracer models, along with various other useful spaces.
        """
        self.P0 = [FunctionSpace(mesh, "DG", 0) for mesh in self.meshes]
        self.P1 = [FunctionSpace(mesh, "CG", 1) for mesh in self.meshes]
        self.P1_vec = [VectorFunctionSpace(mesh, "CG", 1) for mesh in self.meshes]
        self.P1_ten = [TensorFunctionSpace(mesh, "CG", 1) for mesh in self.meshes]
        self.P1DG = [FunctionSpace(mesh, "DG", 1) for mesh in self.meshes]
        self.P1DG_vec = [VectorFunctionSpace(mesh, "DG", 1) for mesh in self.meshes]

        # Shallow water space
        self.V = [FunctionSpace(mesh, self.finite_element) for mesh in self.meshes]

        # Tracer space
        if self.op.solve_tracer:
            self.Q = [FunctionSpace(mesh, self.finite_element_tracer) for mesh in self.meshes]

    def create_solutions(self):
        """
        Set up `Function`s in the prognostic spaces to hold forward and adjoint solutions.
        """
        self.fwd_solutions = [None for mesh in self.meshes]
        self.adj_solutions = [None for mesh in self.meshes]
        self.fwd_solutions_tracer = [None for mesh in self.meshes]
        self.adj_solutions_tracer = [None for mesh in self.meshes]
        for i, V in enumerate(self.V):
            self.fwd_solutions[i] = Function(V, name='Forward solution')
            u, eta = self.fwd_solutions[i].split()
            u.rename("Fluid velocity")
            eta.rename("Elevation")
            self.adj_solutions[i] = Function(V, name='Adjoint solution')
            z, zeta = self.adj_solutions[i].split()
            z.rename("Adjoint fluid velocity")
            zeta.rename("Adjoint elevation")
        if self.op.solve_tracer:
            self.fwd_solutions_tracer = [Function(Q, name="Forward tracer solution") for Q in self.Q]
            self.adj_solutions_tracer = [Function(Q, name="Adjoint tracer solution") for Q in self.Q]

    def set_fields(self):
        """Set velocity field, viscosity, etc *on each mesh*."""
        self.fields = [AttrDict() for P1 in self.P1]
        for i, P1 in enumerate(self.P1):
            self.fields[i].update({
                'horizontal_viscosity': self.op.set_viscosity(P1),
                'horizontal_diffusivity': self.op.set_diffusivity(P1),
                'coriolis_frequency': self.op.set_coriolis(P1),
                'quadratic_drag_coefficient': self.op.set_quadratic_drag_coefficient(P1),
                'manning_drag_coefficient': self.op.set_manning_drag_coefficient(P1),
                'tracer_source_2d': self.op.set_tracer_source(P1)
            })
        self.inflow = [self.op.set_inflow(P1_vec) for P1_vec in self.P1_vec]
        self.bathymetry = [self.op.set_bathymetry(P1) for P1 in self.P1]
        self.depth = [None for bathymetry in self.bathymetry]
        for i, bathymetry in enumerate(self.bathymetry):
            self.depth[i] = DepthExpression(
                bathymetry,
                use_nonlinear_equations=self.shallow_water_options[i].use_nonlinear_equations,
                use_wetting_and_drying=self.shallow_water_options[i].use_wetting_and_drying,
                wetting_and_drying_alpha=self.shallow_water_options[i].wetting_and_drying_alpha,
            )

    # --- Stabilisation

    def set_stabilisation_step(self, i):
        """ Set stabilisation mode and corresponding parameter on the ith mesh."""
        self.minimum_angles = [None for mesh in self.meshes]
        if self.op.use_automatic_sipg_parameter:
            for i, mesh in enumerate(self.meshes):
                self.minimum_angles[i] = get_minimum_angles_2d(mesh)
        if self.op.solve_swe:
            self._set_shallow_water_stabilisation_step(i)
        if self.op.solve_tracer:
            self._set_tracer_stabilisation_step(i)

    def _set_shallow_water_stabilisation_step(self, i):
        op = self.op

        # Symmetric Interior Penalty Galerkin (SIPG) method
        sipg = None
        if hasattr(op, 'sipg_parameter'):
            sipg = op.sipg_parameter
        if self.shallow_water_options[i].use_automatic_sipg_parameter:
            for i, mesh in enumerate(self.meshes):
                cot_theta = 1.0/tan(self.minimum_angles[i])

                # Penalty parameter for shallow water
                nu = self.fields[i].horizontal_viscosity
                if nu is not None:
                    p = self.V[i].sub(0).ufl_element().degree()
                    alpha = Constant(5.0*p*(p+1)) if p != 0 else 1.5
                    alpha = alpha*get_sipg_ratio(nu)*cot_theta
                    sipg = interpolate(alpha, self.P0[i])
        self.shallow_water_options[i].sipg_parameter = sipg

        # Stabilisation
        if self.stabilisation is None:
            return
        elif self.stabilisation == 'lax_friedrichs':
            assert hasattr(op, 'lax_friedrichs_velocity_scaling_factor')
            self.shallow_water_options[i]['lax_friedrichs_velocity_scaling_factor'] = op.lax_friedrichs_velocity_scaling_factor  # TODO: Allow mesh dependent
        else:
            msg = "Stabilisation method {:s} not recognised for {:s}"
            raise ValueError(msg.format(self.stabilisation, self.__class__.__name__))

    def _set_tracer_stabilisation_step(self, i):
        op = self.op

        # Symmetric Interior Penalty Galerkin (SIPG) method
        sipg = None
        if hasattr(op, 'sipg_parameter_tracer'):
            sipg = op.sipg_parameter_tracer
        if self.tracer_options[i].use_automatic_sipg_parameter:
            for i, mesh in enumerate(self.meshes):
                cot_theta = 1.0/tan(self.minimum_angles[i])

                # Penalty parameter for shallow water
                nu = self.fields[i].horizontal_diffusivity
                if nu is not None:
                    p = self.Q[i].ufl_element().degree()
                    alpha = Constant(5.0*p*(p+1)) if p != 0 else 1.5
                    alpha = alpha*get_sipg_ratio(nu)*cot_theta
                    sipg = interpolate(alpha, self.P0[i])
        self.tracer_options[i].sipg_parameter = sipg

        # Stabilisation
        if self.stabilisation is None:
            return
        elif self.stabilisation == 'lax_friedrichs':
            assert hasattr(op, 'lax_friedrichs_tracer_scaling_factor')
            self.tracer_options[i]['lax_friedrichs_tracer_scaling_factor'] = op.lax_friedrichs_tracer_scaling_factor  # TODO: Allow mesh dependent
        elif self.stabilisation == 'su':
            raise NotImplementedError  # TODO
        elif self.stabilisation == 'supg':
            raise NotImplementedError  # TODO
        else:
            msg = "Stabilisation method {:s} not recognised for {:s}"
            raise ValueError(msg.format(self.stabilisation, self.__class__.__name__))

    # --- Solution initialisation and transfer

    def set_initial_condition(self):
        """Apply initial condition(s) for forward solution(s) on first mesh."""
        self.op.set_initial_condition(self)
        if self.op.solve_tracer:
            self.op.set_initial_condition_tracer(self)

    def set_terminal_condition(self):
        """Apply terminal condition(s) for adjoint solution(s) on terminal mesh."""
        self.op.set_terminal_condition(self)
        if self.op.solve_tracer:
            self.op.set_terminal_condition_tracer(self)

    def project_forward_solution(self, i, j):
        """Project forward solution(s) from mesh `i` to mesh `j`."""
        self.project(self.fwd_solutions, i, j)  # TODO: Interpolate from data if not solve_swe
        if self.op.solve_tracer:
            self.project(self.fwd_solutions_tracer, i, j)

    def project_adjoint_solution(self, i, j):
        """Project adjoint solution(s) from mesh `i` to mesh `j`."""
        self.project(self.adj_solutions, i, j)  # TODO: Interpolate from data if not solve_swe
        if self.op.solve_tracer:
            self.project(self.adj_solutions_tracer, i, j)

    # --- Equations

    def create_forward_equations(self, i):
        if self.op.solve_swe:
            self._create_forward_shallow_water_equations(i)
        if self.op.solve_tracer:
            self._create_forward_tracer_equation(i)

    def create_adjoint_equations(self, i):
        if self.op.solve_swe:
            self._create_adjoint_shallow_water_equations(i)
        if self.op.solve_tracer:
            self._create_adjoint_tracer_equation(i)

    def _create_forward_shallow_water_equations(self, i):
        self.equations[i].shallow_water = ShallowWaterEquations(
            self.V[i],
            self.depth[i],
            self.shallow_water_options[i],
        )
        self.equations[i].shallow_water.bnd_functions = self.boundary_conditions[i]['shallow_water']

    def _create_adjoint_shallow_water_equations(self, i):
        self.equations[i].adjoint_shallow_water = AdjointShallowWaterEquations(
            self.V[i],
            self.depth[i],
            self.shallow_water_options[i],
        )
        self.equations[i].adjoint_shallow_water.bnd_functions = self.boundary_conditions[i]['shallow_water']

    def _create_forward_tracer_equation(self, i):
        op = self.tracer_options[i]
        model = ConservativeTracerEquation2D if op.use_tracer_conservative_form else TracerEquation2D
        self.equations[i].tracer = model(
            self.Q[i],
            self.depth[i],
            use_lax_friedrichs=self.tracer_options[i].use_lax_friedrichs_tracer,
            sipg_parameter=self.tracer_options[i].sipg_parameter,
        )
        if op.use_limiter_for_tracers and self.Q[i].ufl_element().degree() > 0:
            self.tracer_limiters[i] = VertexBasedP1DGLimiter(self.Q[i])
        self.equations[i].tracer.bnd_functions = self.boundary_conditions[i]['tracer']

    def _create_adjoint_tracer_equation(self, i):
        op = self.tracer_options[i]
        model = TracerEquation2D if op.use_tracer_conservative_form else ConservativeTracerEquation2D
        self.equations[i].adjoint_tracer = model(
            self.Q[i],
            self.depth[i],
            use_lax_friedrichs=self.tracer_options[i].use_lax_friedrichs_tracer,
            sipg_parameter=self.tracer_options[i].sipg_parameter,
        )
        if op.use_limiter_for_tracers and self.Q[i].ufl_element().degree() > 0:
            self.tracer_limiters[i] = VertexBasedP1DGLimiter(self.Q[i])
        self.equations[i].adjoint_tracer.bnd_functions = self.boundary_conditions[i]['tracer']

    # --- Error estimators

    def create_error_estimators(self, i):
        if self.op.solve_swe:
            self._create_shallow_water_error_estimator(i)
        if self.op.solve_tracer:
            self._create_tracer_error_estimator(i)

    def _create_shallow_water_error_estimator(self, i):
        self.error_estimators[i].shallow_water = ShallowWaterGOErrorEstimator(
            self.V[i],
            self.depth[i],
            self.shallow_water_options[i],
        )

    def _create_tracer_error_estimator(self, i):
        if self.tracer_options[i].use_tracer_conservative_form:
            raise NotImplementedError("Error estimators for conservative tracer not yet implemented.")  # TODO
        else:
            estimator = TracerGOErrorEstimator
        self.error_estimators[i].tracer = estimator(
            self.Q[i],
            self.depth[i],
            use_lax_friedrichs=self.tracer_options[i].use_lax_friedrichs_tracer,
            sipg_parameter=self.tracer_options[i].sipg_parameter,
        )

    # --- Timestepping

    def create_forward_timesteppers(self, i):
        if i == 0:
            self.simulation_time = 0.0
        if self.op.solve_swe:
            self._create_forward_shallow_water_timestepper(i, self.integrator)
        if self.op.solve_tracer:
            self._create_forward_tracer_timestepper(i, self.integrator)

    def create_adjoint_timesteppers(self, i):
        if i == self.num_meshes-1:
            self.simulation_time = self.op.end_time
        if self.op.solve_swe:
            self._create_adjoint_shallow_water_timestepper(i, self.integrator)
        if self.op.solve_tracer:
            self._create_adjoint_tracer_timestepper(i, self.integrator)

    def _get_fields_for_shallow_water_timestepper(self, i):
        fields = AttrDict({
            'linear_drag_coefficient': None,
            'quadratic_drag_coefficient': self.fields[i].quadratic_drag_coefficient,
            'manning_drag_coefficient': self.fields[i].manning_drag_coefficient,
            'viscosity_h': self.fields[i].horizontal_viscosity,
            'coriolis': self.fields[i].coriolis_frequency,
            'wind_stress': None,
            'atmospheric_pressure': None,
            'momentum_source': None,
            'volume_source': None,
        })
        if self.stabilisation == 'lax_friedrichs':
            fields['lax_friedrichs_velocity_scaling_factor'] = self.shallow_water_options[i].lax_friedrichs_velocity_scaling_factor
        return fields

    def _get_fields_for_tracer_timestepper(self, i):
        u, eta = self.fwd_solutions[i].split()
        fields = AttrDict({
            'elev_2d': eta,
            'uv_2d': u,
            'diffusivity_h': self.fields[i].horizontal_diffusivity,
            'source': self.fields[i].tracer_source_2d,
            'tracer_advective_velocity_factor': self.tracer_options[i].tracer_advective_velocity_factor,
            'lax_friedrichs_tracer_scaling_factor': self.tracer_options[i].lax_friedrichs_tracer_scaling_factor,
            'mesh_velocity': None,
        })
        if self.mesh_velocities[i] is not None:
            fields['mesh_velocity'] = self.mesh_velocities[i]
        if self.op.approach == 'lagrangian':
            self.mesh_velocities[i] = u
            fields['uv_2d'] = Constant(as_vector([0.0, 0.0]))
        if self.stabilisation == 'lax_friedrichs':
            fields['lax_friedrichs_tracer_scaling_factor'] = self.tracer_options[i].lax_friedrichs_tracer_scaling_factor
        return fields

    def _create_forward_shallow_water_timestepper(self, i, integrator):
        fields = self._get_fields_for_shallow_water_timestepper(i)
        dt = self.op.dt
        args = (self.equations[i].shallow_water, self.fwd_solutions[i], fields, dt, )
        kwargs = {
            'bnd_conditions': self.boundary_conditions[i]['shallow_water'],
            'solver_parameters': self.op.solver_parameters['shallow_water'],
        }
        if self.op.timestepper == 'CrankNicolson':
            kwargs['semi_implicit'] = self.op.use_semi_implicit_linearisation
            kwargs['theta'] = self.op.implicitness_theta
        if 'shallow_water' in self.error_estimators[i]:
            kwargs['error_estimator'] = self.error_estimators[i].shallow_water
        self.timesteppers[i].shallow_water = integrator(*args, **kwargs)

    def _create_forward_tracer_timestepper(self, i, integrator):
        fields = self._get_fields_for_tracer_timestepper(i)
        dt = self.op.dt
        args = (self.equations[i].tracer, self.fwd_solutions_tracer[i], fields, dt, )
        kwargs = {
            'bnd_conditions': self.boundary_conditions[i]['tracer'],
            'solver_parameters': self.op.solver_parameters['tracer'],
        }
        if self.op.timestepper == 'CrankNicolson':
            kwargs['semi_implicit'] = self.op.use_semi_implicit_linearisation
            kwargs['theta'] = self.op.implicitness_theta
        if 'tracer' in self.error_estimators[i]:
            kwargs['error_estimator'] = self.error_estimators[i].tracer
        self.timesteppers[i].tracer = integrator(*args, **kwargs)

    def _create_adjoint_shallow_water_timestepper(self, i, integrator):
        fields = self._get_fields_for_shallow_water_timestepper(i)

        # Account for dJdq
        self.kernels[i] = self.op.set_qoi_kernel(self.V[i])
        dJdu, dJdeta = self.kernels[i].split()
        self.time_kernel = Constant(1.0 if self.simulation_time >= self.op.start_time else 0.0)
        fields['momentum_source'] = self.time_kernel*dJdu
        fields['volume_source'] = self.time_kernel*dJdeta

        # Construct time integrator
        dt = self.op.dt
        args = (self.equations[i].adjoint_shallow_water, self.adj_solutions[i], fields, dt, )
        kwargs = {
            'bnd_conditions': self.boundary_conditions[i]['shallow_water'],
            'solver_parameters': self.op.adjoint_solver_parameters['shallow_water'],
        }
        if self.op.timestepper == 'CrankNicolson':
            kwargs['semi_implicit'] = self.op.use_semi_implicit_linearisation
            kwargs['theta'] = self.op.implicitness_theta
        self.timesteppers[i].adjoint_shallow_water = integrator(*args, **kwargs)

    def _create_adjoint_tracer_timestepper(self, i, integrator):
        fields = self._get_fields_for_tracer_timestepper(i)

        # Reverse the flow because we are going backwards in time
        fields.uv_2d *= -1

        # Account for dJdc
        dJdc = self.op.set_qoi_kernel_tracer(self, i)
        self.time_kernel = Constant(1.0 if self.simulation_time >= self.op.start_time else 0.0)
        fields['source'] = self.time_kernel*dJdc

        # Construct time integrator
        dt = self.op.dt
        args = (self.equations[i].adjoint_tracer, self.adj_solutions_tracer[i], fields, dt, )
        kwargs = {
            'bnd_conditions': self.boundary_conditions[i]['tracer'],
            'solver_parameters': self.op.adjoint_solver_parameters['tracer'],
        }
        if self.op.timestepper == 'CrankNicolson':
            kwargs['semi_implicit'] = self.op.use_semi_implicit_linearisation
            kwargs['theta'] = self.op.implicitness_theta
        self.timesteppers[i].adjoint_tracer = integrator(*args, **kwargs)

    # --- Solvers

    # TODO: Turbine setup
    def setup_solver_forward(self, i):
        """Setup forward solver on mesh `i`."""
        op = self.op
        op.print_debug(op.indent + "SETUP: Creating forward equations on mesh {:d}...".format(i))
        self.create_forward_equations(i)
        op.print_debug(op.indent + "SETUP: Creating forward timesteppers on mesh {:d}...".format(i))
        self.create_timesteppers(i)
        bcs = self.boundary_conditions[i]
        if op.solve_swe:
            ts = self.timesteppers[i]['shallow_water']
            dbcs = []
            if op.family == 'cg-cg':
                op.print_debug(op.indent + "SETUP: Applying DirichletBCs on mesh {:d}...".format(i))
                for j in bcs['shallow_water']:
                    if 'elev' in bcs['shallow_water'][j]:
                        dbcs.append(DirichletBC(self.V[i].sub(1), bcs['shallow_water'][j]['elev'], j))
            prob = NonlinearVariationalProblem(ts.F, ts.solution, bcs=dbcs)
            ts.solver = NonlinearVariationalSolver(prob, solver_parameters=ts.solver_parameters, options_prefix="forward")
        if op.solve_tracer:
            ts = self.timesteppers[i]['tracer']
            dbcs = []
            if op.tracer_family == 'cg':
                op.print_debug(op.indent + "SETUP: Applying tracer DirichletBCs on mesh {:d}...".format(i))
                for j in bcs['tracer']:
                    if 'value' in bcs['tracer'][j]:
                        dbcs.append(DirichletBC(self.Q[i], bcs['tracer'][j]['value'], j))
            prob = NonlinearVariationalProblem(ts.F, ts.solution, bcs=dbcs)
            ts.solver = NonlinearVariationalSolver(prob, solver_parameters=ts.solver_parameters, options_prefix="forward_tracer")
            op.print_debug(op.indent + "SETUP: Adding callbacks on mesh {:d}...".format(i))
        self.add_callbacks(i)

    def solve_forward_step(self, i, update_forcings=None, export_func=None, plot_pvd=True):
        """
        Solve forward PDE on mesh `i`.

        :kwarg update_forcings: a function which takes simulation time as an argument and is
            evaluated at the start of every timestep.
        :kwarg export_func: a function with no arguments which is evaluated at every export step.
        """
        op = self.op
        plot_pvd &= op.plot_pvd

        # Callbacks
        update_forcings = update_forcings or self.op.get_update_forcings(self, i)
        export_func = export_func or self.op.get_export_func(self, i)
        print_output(80*'=')
        # if i == 0:
        if export_func is not None:
            export_func()
        self.callbacks[i].evaluate(mode='export')

        # We need to project to P1 for vtk outputs
        if op.solve_swe and plot_pvd:
            proj_u = Function(self.P1_vec[i], name="Projected velocity")
            proj_eta = Function(self.P1[i], name="Projected elevation")
            self.solution_file._topology = None
            if i == 0:
                u, eta = self.fwd_solutions[i].split()
                proj_u.project(u)
                proj_eta.project(eta)
                self.solution_file.write(proj_u, proj_eta)
        if op.solve_tracer and plot_pvd:
            proj_tracer = Function(self.P1[i], name="Projected tracer")
            self.tracer_file._topology = None
            if i == 0:
                proj_tracer.project(self.fwd_solutions_tracer[i])
                self.tracer_file.write(proj_tracer)

        t_epsilon = 1.0e-05
        iteration = 0
        start_time = i*op.dt*self.dt_per_mesh
        end_time = (i+1)*op.dt*self.dt_per_mesh
        try:
            assert np.allclose(self.simulation_time, start_time)
        except AssertionError:
            msg = "Mismatching start time: {:.2f} vs {:.2f}"
            raise ValueError(msg.format(self.simulation_time, start_time))
        update_forcings(self.simulation_time)
        op.print_debug("SOLVE: Entering forward timeloop on mesh {:d}...".format(i))
        msg = "{:2d} {:s} FORWARD SOLVE mesh {:2d}/{:2d}  time {:8.2f}"
        print_output(msg.format(self.outer_iteration, '  '*i, i+1, self.num_meshes, self.simulation_time))
        ts = self.timesteppers[i]
        coords = self.meshes[i].coordinates
        while self.simulation_time <= end_time - t_epsilon:

            # Get mesh velocity
            if self.mesh_movers[i] is not None:  # TODO: generalise
                self.mesh_movers[i].adapt()
                self.mesh_velocities[i].assign((self.mesh_movers[i].x - coords)/op.dt)
                if iteration % op.dt_per_export == 0:
                    self.mesh_velocity_file.write(self.mesh_velocities[i])

            # Solve equations
            if op.solve_swe:
                ts.shallow_water.advance(self.simulation_time, update_forcings)
            if op.solve_tracer:
                ts.tracer.advance(self.simulation_time, update_forcings)
                if self.tracer_options[i].use_limiter_for_tracers:
                    self.tracer_limiters[i].apply(self.fwd_solutions_tracer[i])

            # Move mesh
            self.move_mesh(i)
            # if self.mesh_movers[i] is not None:  # TODO: generalise
            #     self.meshes[i].coordinates.assign(self.mesh_movers[i].x)

            # Outputs
            iteration += 1
            self.simulation_time += op.dt
            self.callbacks[i].evaluate(mode='timestep')
            if iteration % op.dt_per_export == 0:
                print_output(msg.format(self.outer_iteration, '  '*i, i+1, self.num_meshes, self.simulation_time))
                if op.solve_swe and plot_pvd:
                    u, eta = self.fwd_solutions[i].split()
                    proj_u.project(u)
                    proj_eta.project(eta)
                    self.solution_file.write(proj_u, proj_eta)
                if op.solve_tracer and plot_pvd:
                    proj_tracer.project(self.fwd_solutions_tracer[i])
                    self.tracer_file.write(proj_tracer)
                if export_func is not None:
                    export_func()
                self.callbacks[i].evaluate(mode='export')
        op.print_debug("Done!")
        print_output(80*'=')

    def setup_solver_adjoint(self, i):
        """Setup forward solver on mesh `i`."""
        op = self.op
        op.print_debug(op.indent + "SETUP: Creating adjoint equations on mesh {:d}...".format(i))
        self.create_adjoint_equations(i)
        op.print_debug(op.indent + "SETUP: Creating adjoint timesteppers on mesh {:d}...".format(i))
        self.create_adjoint_timesteppers(i)
        bcs = self.boundary_conditions[i]
        if op.solve_swe:
            dbcs = []
            ts = self.timesteppers[i]['adjoint_shallow_water']
            if op.family == 'cg-cg':
                op.print_debug(op.indent + "SETUP: Applying adjoint DirichletBCs on mesh {:d}...".format(i))
                for j in bcs['shallow_water']:
                    if 'un' not in bcs['shallow_water'][j]:
                        dbcs.append(DirichletBC(self.V[i].sub(1), 0, j))
            prob = NonlinearVariationalProblem(ts.F, ts.solution, bcs=dbcs)
            ts.solver = NonlinearVariationalSolver(prob, solver_parameters=ts.solver_parameters, options_prefix="adjoint")
        if op.solve_tracer:
            ts = self.timesteppers[i]['adjoint_tracer']
            dbcs = []
            if op.family == 'cg':
                op.print_debug(op.indent + "SETUP: Applying adjoint tracer DirichletBCs on mesh {:d}...".format(i))
                raise NotImplementedError  # TODO
            prob = NonlinearVariationalProblem(ts.F, ts.solution, bcs=dbcs)
            ts.solver = NonlinearVariationalSolver(prob, solver_parameters=ts.solver_parameters, options_prefix="adjoint_tracer")

    def solve_adjoint_step(self, i, update_forcings=None, export_func=None, plot_pvd=True):
        """
        Solve adjoint PDE on mesh `i` *backwards in time*.

        :kwarg update_forcings: a function which takes simulation time as an argument and is
            evaluated at the start of every timestep.
        :kwarg export_func: a function with no arguments which is evaluated at every export step.
        """
        op = self.op
        plot_pvd &= op.plot_pvd

        # Callbacks
        update_forcings = update_forcings or self.op.get_update_forcings(self, i)
        export_func = export_func or self.op.get_export_func(self, i)
        print_output(80*'=')
        # if i == self.num_meshes-1:
        if export_func is not None:
            export_func()

        # We need to project to P1 for vtk outputs
        if op.solve_swe and plot_pvd:
            proj_z = Function(self.P1_vec[i], name="Projected continuous adjoint velocity")
            proj_zeta = Function(self.P1[i], name="Projected continuous adjoint elevation")
            self.adjoint_solution_file._topology = None
            if i == self.num_meshes-1:
                z, zeta = self.adj_solutions[i].split()
                proj_z.project(z)
                proj_zeta.project(zeta)
                self.adjoint_solution_file.write(proj_z, proj_zeta)
        if op.solve_tracer and plot_pvd:
            proj_tracer = Function(self.P1[i], name="Projected continuous adjoint tracer")
            self.adjoint_tracer_file._topology = None
            if i == self.num_meshes-1:
                proj_tracer.project(self.adj_solutions_tracer[i])
                self.adjoint_tracer_file.write(proj_tracer)

        t_epsilon = 1.0e-05
        iteration = 0
        start_time = (i+1)*op.dt*self.dt_per_mesh
        end_time = i*op.dt*self.dt_per_mesh
        try:
            assert np.allclose(self.simulation_time, start_time)
        except AssertionError:
            msg = "Mismatching start time: {:f} vs {:f}"
            raise ValueError(msg.format(self.simulation_time, start_time))
        update_forcings(self.simulation_time)
        op.print_debug("SOLVE: Entering forward timeloop on mesh {:d}...".format(i))
        msg = "{:2d} {:s}  ADJOINT SOLVE mesh {:2d}/{:2d}  time {:8.2f}"
        print_output(msg.format(self.outer_iteration, '  '*i, i+1, self.num_meshes, self.simulation_time))
        ts = self.timesteppers[i]
        while self.simulation_time >= end_time + t_epsilon:
            if op.solve_swe:
                ts.adjoint_shallow_water.advance(self.simulation_time, update_forcings)
            if op.solve_tracer:
                ts.adjoint_tracer.advance(self.simulation_time, update_forcings)
                if self.tracer_options[i].use_limiter_for_tracers:
                    self.tracer_limiters[i].apply(self.adj_solutions_tracer[i])
            iteration += 1
            self.simulation_time -= op.dt
            self.callbacks[i].evaluate(mode='timestep')
            if iteration % op.dt_per_export == 0:
                print_output(msg.format(self.outer_iteration, '  '*i, i+1, self.num_meshes, self.simulation_time))
                if op.solve_swe and plot_pvd:
                    z, zeta = self.adj_solutions[i].split()
                    proj_z.project(z)
                    proj_zeta.project(zeta)
                    self.adjoint_solution_file.write(proj_z, proj_zeta)
                if op.solve_tracer and plot_pvd:
                    proj_tracer.project(self.adj_solutions_tracer[i])
                    self.adjoint_tracer_file.write(proj_tracer)
                if export_func is not None:
                    export_func()
        op.print_debug("Done!")
        print_output(80*'=')

    # --- Metric

    def get_hessian_metric(self, adjoint=False, **kwargs):
        kwargs.setdefault('normalise', True)
        kwargs['op'] = self.op
        self.metrics = []
        solutions = self.adj_solutions if adjoint else self.fwd_solutions
        for i, sol in enumerate(solutions):
            fields = {'bathymetry': self.bathymetry[i], 'inflow': self.inflow[i]}
            self.metrics.append(get_hessian_metric(sol, fields=fields, **kwargs))

    # --- Run scripts

    # TODO: Tracer
    def run_hessian_based(self, **kwargs):
        """
        Adaptation loop for Hessian based approach.

        Field for adaptation is specified by `op.adapt_field`.

        Multiple fields can be combined using double-understrokes and either 'avg' for metric
        average or 'int' for metric intersection. We assume distributivity of intersection over
        averaging.

        For example, `adapt_field = 'elevation__avg__velocity_x__int__bathymetry'` would imply
        first intersecting the Hessians recovered from the x-component of velocity and bathymetry
        and then averaging the result with the Hessian recovered from the elevation.

        Stopping criteria:
          * iteration count > self.op.num_adapt;
          * relative change in element count < self.op.element_rtol;
          * relative change in quantity of interest < self.op.qoi_rtol.
        """
        op = self.op
        if op.adapt_field in ('all_avg', 'all_int'):
            c = op.adapt_field[-3:]
            op.adapt_field = "velocity_x__{:s}__velocity_y__{:s}__elevation".format(c, c)
        adapt_fields = ('__int__'.join(op.adapt_field.split('__avg__'))).split('__int__')
        if op.hessian_time_combination not in ('integrate', 'intersect'):
            msg = "Hessian time combination method '{:s}' not recognised."
            raise ValueError(msg.format(op.hessian_time_combination))

        for n in range(op.num_adapt):
            self.outer_iteration = n

            # Arrays to hold Hessians for each field on each window
            H_windows = [[Function(P1_ten) for P1_ten in self.P1_ten] for f in adapt_fields]

            if hasattr(self, 'hessian_func'):
                delattr(self, 'hessian_func')
            update_forcings = None
            export_func = None
            for i in range(self.num_meshes):

                # Transfer the solution from the previous mesh / apply initial condition
                self.transfer_forward_solution(i)

                if n < op.num_adapt-1:

                    # Create double L2 projection operator which will be repeatedly used
                    kwargs = {
                        'enforce_constraints': False,
                        'normalise': False,
                        'noscale': True,
                    }
                    recoverer = ShallowWaterHessianRecoverer(
                        self.V[i], op=op,
                        constant_fields={'bathymetry': self.bathymetry[i]}, **kwargs,
                    )

                    def hessian(sol, adapt_field):
                        fields = {'adapt_field': adapt_field, 'fields': self.fields[i]}
                        return recoverer.get_hessian_metric(sol, **fields, **kwargs)

                    # Array to hold time-integrated Hessian UFL expression
                    H_window = [0 for f in adapt_fields]

                    def update_forcings(t):  # TODO: Other timesteppers
                        """Time-integrate Hessian using Trapezium Rule."""
                        iteration = int(self.simulation_time/op.dt)
                        if iteration % op.hessian_timestep_lag != 0:
                            iteration += 1
                            return
                        first_ts = iteration == i*self.dt_per_mesh
                        final_ts = iteration == (i+1)*self.dt_per_mesh
                        dt = op.dt*op.hessian_timestep_lag
                        for j, f in enumerate(adapt_fields):
                            H = hessian(self.fwd_solutions[i], f)
                            if f == 'bathymetry':
                                H_window[j] = H
                            elif op.hessian_time_combination == 'integrate':
                                H_window[j] += (0.5 if first_ts or final_ts else 1.0)*dt*H
                            else:
                                H_window[j] = H if first_ts else metric_intersection(H, H_window[j])

                    def export_func():
                        """
                        Extract time-averaged Hessian.

                        NOTE: We only care about the final export in each mesh iteration
                        """
                        if np.allclose(self.simulation_time, (i+1)*op.dt*self.dt_per_mesh):
                            for j, H in enumerate(H_window):
                                if op.hessian_time_combination == 'intersect':
                                    H_window[j] *= op.dt*self.dt_per_mesh
                                H_windows[j][i].interpolate(H_window[j])

                # Solve step for current mesh iteration
                self.setup_solver_forward(i)
                self.solve_forward_step(i, export_func=export_func, update_forcings=update_forcings, plot_pvd=op.plot_pvd)

                # Delete objects to free memory
                if n < op.num_adapt-1:
                    del H_window
                    del recoverer

            # --- Convergence criteria

            # Check QoI convergence
            qoi = self.quantity_of_interest()
            print_output("Quantity of interest {:d}: {:.4e}".format(n+1, qoi))
            self.qois.append(qoi)
            if len(self.qois) > 1:
                if np.abs(self.qois[-1] - self.qois[-2]) < op.qoi_rtol*self.qois[-2]:
                    print_output("Converged quantity of interest!")
                    break

            # Check maximum number of iterations
            if n == op.num_adapt - 1:
                break

            # --- Time normalise metrics

            for j in range(len(adapt_fields)):
                space_time_normalise(H_windows[j], op=op)

            # Combine metrics (if appropriate)
            metrics = [Function(P1_ten, name="Hessian metric") for P1_ten in self.P1_ten]
            for i in range(self.num_meshes):
                H_window = [H_windows[j][i] for j in range(len(adapt_fields))]
                if 'int' in op.adapt_field:
                    if 'avg' in op.adapt_field:
                        raise NotImplementedError  # TODO: mixed case
                    metrics[i].assign(metric_intersection(*H_window))
                elif 'avg' in op.adapt_field:
                    metrics[i].assign(metric_average(*H_window))
                else:
                    try:
                        assert len(adapt_fields) == 1
                    except AssertionError:
                        msg = "Field for adaptation '{:s}' not recognised"
                        raise ValueError(msg.format(op.adapt_field))
                    metrics[i].assign(H_window[0])
            del H_windows

            # metric_file = File(os.path.join(self.di, 'metric.pvd'))
            complexities = []
            for i, M in enumerate(metrics):
                # metric_file.write(M)
                complexities.append(metric_complexity(M))
            self.st_complexities.append(sum(complexities)*op.end_time/op.dt)

            # --- Adapt meshes

            print_output("\nStarting mesh adaptation for iteration {:d}...".format(n+1))
            for i, M in enumerate(metrics):
                print_output("Adapting mesh {:d}/{:d}...".format(i+1, self.num_meshes))
                self.meshes[i] = pragmatic_adapt(self.meshes[i], M, op=op)
            del metrics
            self.num_cells.append([mesh.num_cells() for mesh in self.meshes])
            self.num_vertices.append([mesh.num_vertices() for mesh in self.meshes])
            print_output("Done!")

            # ---  Setup for next run / logging

            self.setup_all(self.meshes)
            self.dofs.append([np.array(V.dof_count).sum() for V in self.V])

            print_output("\nResulting meshes")
            msg = "  {:2d}: complexity {:8.1f} vertices {:7d} elements {:7d}"
            for i, c in enumerate(complexities):
                print_output(msg.format(i, c, self.num_vertices[n+1][i], self.num_cells[n+1][i]))
            msg = "  total:            {:8.1f}          {:7d}          {:7d}\n"
            print_output(msg.format(
                self.st_complexities[-1],
                sum(self.num_vertices[n+1])*self.dt_per_mesh,
                sum(self.num_cells[n+1])*self.dt_per_mesh,
            ))

            # Check convergence of *all* element counts
            converged = True
            for i, num_cells_ in enumerate(self.num_cells[n-1]):
                if np.abs(self.num_cells[n][i] - num_cells_) > op.element_rtol*num_cells_:
                    converged = False
            if converged:
                print_output("Converged number of mesh elements!")
                break

    # TODO: Tracer?
    # TODO: Modify indicator for time interval
    def run_dwp(self, **kwargs):
        r"""
        The "dual weighted primal" approach, first used (not under this name) in [1]. For shallow
        water tsunami propagation problems with a quantity of interest of the form

      ..math::
            J(u, \eta) = \int_{t_0}^{t_f} \int_R \eta \;\mathrm dx\;\mathrm dt,

        where :math:`eta` is free surface displacement and :math:`R\subset\Omega` is a spatial
        region of interest, it can be shown [1] that

      ..math::
            \int_R q(x, t=t_0) \cdot \hat q(x, t=t_0) \;\mathrm dx = \int_R q(x, t=t_f) \cdot \hat q(x, t=t_f) \;\mathrm dx

        under certain boundary condition assumptions. Here :math:`q=(u,\eta)` and :math:`\hat q`
        denotes the adjoint solution. Note that the choice of :math:`[t_0, t_f] \subseteq [0, T]`
        is arbitrary, so the above holds at all time levels.

        This motivates using error indicators of the form :math:`|q \cdot \hat q|`.

        [1] B. Davis & R. LeVeque, "Adjoint Methods for Guiding Adaptive Mesh Refinement in
            Tsunami Modelling", Pure and Applied Geophysics, 173, Springer International
            Publishing (2016), p.4055--4074, DOI 10.1007/s00024-016-1412-y.
        """
        op = self.op
        for n in range(op.num_adapt):
            self.outer_iteration = n

            # --- Solve forward to get checkpoints

            # self.get_checkpoints()
            self.solve_forward()

            # --- Convergence criteria

            # Check QoI convergence
            qoi = self.quantity_of_interest()
            print_output("Quantity of interest {:d}: {:.4e}".format(n+1, qoi))
            self.qois.append(qoi)
            if len(self.qois) > 1:
                if np.abs(self.qois[-1] - self.qois[-2]) < op.qoi_rtol*self.qois[-2]:
                    print_output("Converged quantity of interest!")
                    break

            # Check maximum number of iterations
            if n == op.num_adapt - 1:
                break

            # --- Loop over mesh windows *in reverse*

            for i, P1 in enumerate(self.P1):
                self.indicators[i]['dwp'] = Function(P1, name="DWP indicator")
            metrics = [Function(P1_ten, name="Metric") for P1_ten in self.P1_ten]
            for i in range(self.num_meshes-1, -1, -1):
                fwd_solutions_step = []
                adj_solutions_step = []

                # --- Solve forward on current window

                def export_func():
                    fwd_solutions_step.append(self.fwd_solutions[i].copy(deepcopy=True))

                self.simulation_time = i*op.dt*self.dt_per_mesh
                self.transfer_forward_solution(i)
                self.setup_solver_forward(i)
                self.solve_forward_step(i, export_func=export_func, plot_pvd=False)

                # --- Solve adjoint on current window

                def export_func():
                    adj_solutions_step.append(self.adj_solutions[i].copy(deepcopy=True))

                self.transfer_adjoint_solution(i)
                self.setup_solver_adjoint(i)
                self.solve_adjoint_step(i, export_func=export_func, plot_pvd=False)

                # --- Assemble indicators and metrics

                n_fwd = len(fwd_solutions_step)
                n_adj = len(adj_solutions_step)
                if n_fwd != n_adj:
                    msg = "Mismatching number of indicators ({:d} vs {:d})"
                    raise ValueError(msg.format(n_fwd, n_adj))
                I = 0
                op.print_debug("DWP indicators on mesh {:2d}".format(i))
                for j, solutions in enumerate(zip(fwd_solutions_step, reversed(adj_solutions_step))):
                    scaling = 0.5 if j in (0, n_fwd-1) else 1.0  # Trapezium rule  # TODO: Other integrators
                    fwd_dot_adj = abs(inner(*solutions))
                    op.print_debug("    ||<q, q*>||_L2 = {:.4e}".format(assemble(fwd_dot_adj*fwd_dot_adj*dx)))
                    I += op.dt*self.dt_per_mesh*scaling*fwd_dot_adj
                self.indicators[i]['dwp'].interpolate(I)
                metrics[i].assign(isotropic_metric(self.indicators[i]['dwp'], normalise=False))

            # --- Normalise metrics

            space_time_normalise(metrics, op=op)

            # Output to .pvd and .vtu
            # metric_file = File(os.path.join(self.di, 'metric.pvd'))
            complexities = []
            for i, M in enumerate(metrics):
                self.indicator_file._topology = None
                self.indicator_file.write(self.indicators[i]['dwp'])
                # metric_file._topology = None
                # metric_file.write(M)
                complexities.append(metric_complexity(M))
            self.st_complexities.append(sum(complexities)*op.end_time/op.dt)

            # --- Adapt meshes

            print_output("\nStarting mesh adaptation for iteration {:d}...".format(n+1))
            for i, M in enumerate(metrics):
                print_output("Adapting mesh {:d}/{:d}...".format(i+1, self.num_meshes))
                self.meshes[i] = pragmatic_adapt(self.meshes[i], M, op=op)
            del metrics
            self.num_cells.append([mesh.num_cells() for mesh in self.meshes])
            self.num_vertices.append([mesh.num_vertices() for mesh in self.meshes])
            print_output("Done!")

            # ---  Setup for next run / logging

            self.setup_all(self.meshes)
            self.dofs.append([np.array(V.dof_count).sum() for V in self.V])

            print_output("\nResulting meshes")
            msg = "  {:2d}: complexity {:8.1f} vertices {:7d} elements {:7d}"
            for i, c in enumerate(complexities):
                print_output(msg.format(i, c, self.num_vertices[n+1][i], self.num_cells[n+1][i]))
            msg = "  total:            {:8.1f}          {:7d}          {:7d}\n"
            print_output(msg.format(
                self.st_complexities[-1],
                sum(self.num_vertices[n+1])*self.dt_per_mesh,
                sum(self.num_cells[n+1])*self.dt_per_mesh,
            ))

            # Check convergence of *all* element counts
            converged = True
            for i, num_cells_ in enumerate(self.num_cells[n-1]):
                if np.abs(self.num_cells[n][i] - num_cells_) > op.element_rtol*num_cells_:
                    converged = False
            if converged:
                print_output("Converged number of mesh elements!")
                break

    # TODO: Tracer
    # TODO: Enable move to base class
    def run_dwr(self, **kwargs):
        # TODO: doc
        op = self.op
        for n in range(op.num_adapt):
            self.outer_iteration = n

            # --- Solve forward to get checkpoints

            # self.get_checkpoints()
            self.solve_forward()

            # --- Convergence criteria

            # Check QoI convergence
            qoi = self.quantity_of_interest()
            print_output("Quantity of interest {:d}: {:.4e}".format(n+1, qoi))
            self.qois.append(qoi)
            if len(self.qois) > 1:
                if np.abs(self.qois[-1] - self.qois[-2]) < op.qoi_rtol*self.qois[-2]:
                    print_output("Converged quantity of interest!")
                    break

            # Check maximum number of iterations
            if n == op.num_adapt - 1:
                break

            # --- Setup problem on enriched space

            same_mesh = True
            for mesh in self.meshes:
                if mesh != self.meshes[0]:
                    same_mesh = False
                    break
            if same_mesh:
                print_output("All meshes are identical so we use an identical hierarchy.")
                hierarchy = MeshHierarchy(self.meshes[0], 1)
                refined_meshes = [hierarchy[1] for mesh in self.meshes]
            else:
                print_output("Meshes differ so we create separate hierarchies.")
                hierarchies = [MeshHierarchy(mesh, 1) for mesh in self.meshes]
                refined_meshes = [hierarchy[1] for hierarchy in hierarchies]
            ep = type(self)(
                op,
                meshes=refined_meshes,
                nonlinear=self.nonlinear,
            )
            ep.outer_iteration = n

            # --- Loop over mesh windows *in reverse*

            for i, P1 in enumerate(self.P1):
                self.indicators[i]['dwr'] = Function(P1, name="DWR indicator")
            metrics = [Function(P1_ten, name="Metric") for P1_ten in self.P1_ten]
            for i in range(self.num_meshes-1, -1, -1):
                fwd_solutions_step = []
                fwd_solutions_step_old = []
                adj_solutions_step = []
                enriched_adj_solutions_step = []
                tm = dmhooks.get_transfer_manager(self.meshes[i]._plex)

                # --- Setup forward solver for enriched problem

                # TODO: Need to transfer fwd sol in nonlinear case
                ep.create_error_estimators(i)  # These get passed to the timesteppers under the hood
                ep.setup_solver_forward(i)
                ets = ep.timesteppers[i]['shallow_water']  # TODO: Tracer option

                # --- Solve forward on current window

                ts = self.timesteppers[i]['shallow_water']  # TODO: Tracer option

                def export_func():
                    fwd_solutions_step.append(ts.solution.copy(deepcopy=True))
                    fwd_solutions_step_old.append(ts.solution_old.copy(deepcopy=True))
                    # TODO: Also need store fields at each export (in general case)

                self.simulation_time = i*op.dt*self.dt_per_mesh
                self.transfer_forward_solution(i)
                self.setup_solver_forward(i)
                self.solve_forward_step(i, export_func=export_func, plot_pvd=False)

                # --- Solve adjoint on current window

                def export_func():
                    adj_solutions_step.append(self.adj_solutions[i].copy(deepcopy=True))

                self.transfer_adjoint_solution(i)
                self.setup_solver_adjoint(i)
                self.solve_adjoint_step(i, export_func=export_func, plot_pvd=False)

                # --- Solve adjoint on current window in enriched space

                def export_func():
                    enriched_adj_solutions_step.append(ep.adj_solutions[i].copy(deepcopy=True))

                ep.simulation_time = (i+1)*op.dt*self.dt_per_mesh  # TODO: Shouldn't be needed
                ep.transfer_adjoint_solution(i)
                ep.setup_solver_adjoint(i)
                ep.solve_adjoint_step(i, export_func=export_func, plot_pvd=False)

                # --- Assemble indicators and metrics

                n_fwd = len(fwd_solutions_step)
                n_adj = len(adj_solutions_step)
                if n_fwd != n_adj:
                    msg = "Mismatching number of indicators ({:d} vs {:d})"
                    raise ValueError(msg.format(n_fwd, n_adj))
                adj_solutions_step = list(reversed(adj_solutions_step))
                enriched_adj_solutions_step = list(reversed(enriched_adj_solutions_step))
                I = 0
                op.print_debug("DWR indicators on mesh {:2d}".format(i))
                indicator_enriched = Function(ep.P0[i])
                fwd_proj = Function(ep.V[i])
                fwd_old_proj = Function(ep.V[i])
                adj_error = Function(ep.V[i])
                bcs = self.boundary_conditions[i]['shallow_water']  # TODO: Tracer option
                ets.setup_error_estimator(fwd_proj, fwd_old_proj, adj_error, bcs)

                # Loop over exported timesteps
                for j in range(len(fwd_solutions_step)):
                    scaling = 0.5 if j in (0, n_fwd-1) else 1.0  # Trapezium rule  # TODO: Other integrators

                    # Prolong forward solution at current and previous timestep
                    tm.prolong(fwd_solutions_step[j], fwd_proj)
                    tm.prolong(fwd_solutions_step_old[j], fwd_old_proj)

                    # Approximate adjoint error in enriched space
                    tm.prolong(adj_solutions_step[j], adj_error)
                    adj_error *= -1
                    adj_error += enriched_adj_solutions_step[j]

                    # Compute dual weighted residual
                    indicator_enriched.interpolate(abs(ets.error_estimator.weighted_residual()))

                    # Time-integrate
                    I += op.dt*self.dt_per_mesh*scaling*indicator_enriched
                indicator_enriched_cts = interpolate(I, ep.P1[i])
                tm.inject(indicator_enriched_cts, self.indicators[i]['dwr'])
                metrics[i].assign(isotropic_metric(self.indicators[i]['dwr'], normalise=False))

            del indicator_enriched_cts
            del adj_error
            del indicator_enriched
            del ep
            del refined_meshes
            if same_mesh:
                del hierarchy
            else:
                del hierarchies

            # --- Normalise metrics

            space_time_normalise(metrics, op=op)

            # Output to .pvd and .vtu
            # metric_file = File(os.path.join(self.di, 'metric.pvd'))
            complexities = []
            for i, M in enumerate(metrics):
                self.indicator_file._topology = None
                self.indicator_file.write(self.indicators[i]['dwr'])
                # metric_file._topology = None
                # metric_file.write(M)
                complexities.append(metric_complexity(M))
            self.st_complexities.append(sum(complexities)*op.end_time/op.dt)

            # --- Adapt meshes

            print_output("\nStarting mesh adaptation for iteration {:d}...".format(n+1))
            for i, M in enumerate(metrics):
                print_output("Adapting mesh {:d}/{:d}...".format(i+1, self.num_meshes))
                self.meshes[i] = pragmatic_adapt(self.meshes[i], M, op=op)
            del metrics
            self.num_cells.append([mesh.num_cells() for mesh in self.meshes])
            self.num_vertices.append([mesh.num_vertices() for mesh in self.meshes])
            print_output("Done!")

            # ---  Setup for next run / logging

            self.setup_all(self.meshes)
            self.dofs.append([np.array(V.dof_count).sum() for V in self.V])

            print_output("\nResulting meshes")
            msg = "  {:2d}: complexity {:8.1f} vertices {:7d} elements {:7d}"
            for i, c in enumerate(complexities):
                print_output(msg.format(i, c, self.num_vertices[n+1][i], self.num_cells[n+1][i]))
            msg = "  total:            {:8.1f}          {:7d}          {:7d}\n"
            print_output(msg.format(
                self.st_complexities[-1],
                sum(self.num_vertices[n+1])*self.dt_per_mesh,
                sum(self.num_cells[n+1])*self.dt_per_mesh,
            ))

            # Check convergence of *all* element counts
            converged = True
            for i, num_cells_ in enumerate(self.num_cells[n-1]):
                if np.abs(self.num_cells[n][i] - num_cells_) > op.element_rtol*num_cells_:
                    converged = False
            if converged:
                print_output("Converged number of mesh elements!")
                break
