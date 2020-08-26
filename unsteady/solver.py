from __future__ import absolute_import

from thetis import *
from thetis.limiter import VertexBasedP1DGLimiter

import numpy as np
import os
from time import perf_counter

from ..adapt.adaptation import pragmatic_adapt
from ..adapt.metric import *
from .base import AdaptiveProblemBase
from .callback import *
from .swe.utils import *


__all__ = ["AdaptiveProblem"]


# TODO:
#  * Mesh movement ALE formulation
#  * CG tracers (plus SU and SUPG stabilisation)
#  * Multiple tracers
#  * Checkpointing to disk
#  * Allow mesh dependent Lax-Friedrichs parameter(s)

class AdaptiveProblem(AdaptiveProblemBase):
    """
    A class describing 2D models which may involve coupling of any of:

        * shallow water equations (hydrodynamics);
        * tracer transport;
        * sediment transport;
        * Exner equation.

    Mesh adaptation is supported both using the metric-based approach of Pragmatic, as well as
    mesh movement methods. Mesh movement can either be driven by a prescribed velocity or by a
    monitor function.

    In addition to enabling the solution of these equations, the class provides means to solve
    the associated continuous form adjoint equations, evaluate goal-oriented error estimators and
    indicators, as well as various other utilities.
    """
    def __init__(self, op, nonlinear=True, **kwargs):
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
            'mesh_velocity': None,
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
        self.sediment_options = [AttrDict() for i in range(op.num_meshes)]
        self.exner_options = [AttrDict() for i in range(op.num_meshes)]
        static_options = {
            'use_automatic_sipg_parameter': op.use_automatic_sipg_parameter,
            # 'check_tracer_conservation': True,  # TODO
            'use_lax_friedrichs_tracer': op.stabilisation == 'lax_friedrichs',
            'use_limiter_for_tracers': op.use_limiter_for_tracers and op.tracer_family == 'dg',
            'sipg_parameter': None,
            'use_tracer_conservative_form': op.use_tracer_conservative_form,
        }
        if op.use_tracer_conservative_form and op.approach == 'lagrangian':
            raise NotImplementedError  # TODO
        self.tracer_limiters = [None for i in range(op.num_meshes)]
        for i, to in enumerate(self.tracer_options):
            to.update(static_options)
            if hasattr(op, 'sipg_parameter_tracer') and op.sipg_parameter_tracer is not None:
                swo['sipg_parameter_tracer'] = op.sipg_parameter_tracer
        for i, to in enumerate(self.sediment_options):
            to.update(static_options)
            if hasattr(op, 'sipg_parameter_sediment') and op.sipg_parameter_sediment is not None:
                swo['sipg_parameter_sediment'] = op.sipg_parameter_sediment
        super(AdaptiveProblem, self).__init__(op, nonlinear=nonlinear, **kwargs)

    def create_outfiles(self):
        if self.op.solve_swe:
            self.solution_file = File(os.path.join(self.di, 'solution.pvd'))
            self.adjoint_solution_file = File(os.path.join(self.di, 'adjoint_solution.pvd'))
        if self.op.solve_tracer:
            self.tracer_file = File(os.path.join(self.di, 'tracer.pvd'))
            self.adjoint_tracer_file = File(os.path.join(self.di, 'adjoint_tracer.pvd'))
        if self.op.solve_sediment:
            self.sediment_file = File(os.path.join(self.di, 'sediment.pvd'))
        if self.op.plot_bathymetry or self.op.solve_exner:
            self.exner_file = File(os.path.join(self.di, 'modified_bathymetry.pvd'))

    def set_finite_elements(self):
        """
        There are three options for the shallow water mixed finite element pair:
          * Taylor-Hood (continuous Galerkin)   P2-P1      'cg-cg';
          * equal order discontinuous Galerkin  PpDG-PpDG  'dg-dg';
          * mixed continuous-discontinuous      P1DG-P2    'dg-cg'.

        There are two options for the tracer finite element:
          * Continuous Galerkin     Pp    'cg';
          * Discontinuous Galerkin  PpDG  'dg'.

        For the sediment and Exner models, you are required to use DG and CG,
        respectively.
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

        if self.op.solve_sediment:
            p = self.op.degree_sediment
            family = self.op.sediment_family
            if family == 'dg':
                self.finite_element_sediment = FiniteElement("DG", triangle, p)
            else:
                raise NotImplementedError("Cannot build order {:d} {:s} element".format(p, family))

        if self.op.solve_exner:
            p = self.op.degree_bathymetry
            family = self.op.bathymetry_family
            if family == 'cg':
                self.finite_element_bathymetry = FiniteElement("CG", triangle, p)
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

        # Tracer space(s)
        if self.op.solve_tracer:
            self.Q = [FunctionSpace(mesh, self.finite_element_tracer) for mesh in self.meshes]
        if self.op.solve_sediment:
            self.Q = [FunctionSpace(mesh, self.finite_element_sediment) for mesh in self.meshes]

        # Bathymetry space
        if self.op.solve_exner:
            self.W = [FunctionSpace(mesh, self.finite_element_bathymetry) for mesh in self.meshes]

    def create_intermediary_spaces(self):
        super(AdaptiveProblem, self).create_intermediary_spaces()
        if self.op.approach != 'monge_ampere':
            return
        mesh_copies = self.intermediary_meshes
        if self.op.solve_tracer:
            spaces = [FunctionSpace(mesh, self.finite_element_tracer) for mesh in mesh_copies]
            self.intermediary_solutions_tracer = [Function(space) for space in spaces]
        if self.op.solve_sediment:
            spaces = [FunctionSpace(mesh, self.finite_element_sediment) for mesh in mesh_copies]
            self.intermediary_solutions_sediment = [Function(space) for space in spaces]
        if self.op.solve_exner:
            spaces = [FunctionSpace(mesh, self.finite_element_bathymetry) for mesh in mesh_copies]
            self.intermediary_solutions_bathymetry = [Function(space) for space in spaces]
        if hasattr(self.op, 'sediment_model'):
            space_uv_cg = [FunctionSpace(mesh, self.op.sediment_model.uv_cg.function_space().ufl_element()) for mesh in mesh_copies]
            self.intermediary_solutions_uv_cg = [Function(space) for space in space_uv_cg]
            spaces = [FunctionSpace(mesh, self.finite_element_bathymetry) for mesh in mesh_copies]
            self.intermediary_solutions_old_bathymetry = [Function(space) for space in spaces]
            self.intermediary_solutions_TOB = [Function(space) for space in spaces]
            self.intermediary_solutions_depth = [Function(space) for space in spaces]

            if self.op.suspended:
                if self.op.convective_vel_flag:
                    self.intermediary_corr_vel_factor = [Function(space) for space in spaces]
                space_dg = [FunctionSpace(mesh, self.op.sediment_model.coeff.function_space().ufl_element()) for mesh in mesh_copies]
                self.intermediary_coeff = [Function(space) for space in space_dg]
                self.intermediary_ceq = [Function(space) for space in space_dg]
                self.intermediary_equiltracer = [Function(space) for space in space_dg]
                self.intermediary_ero = [Function(space) for space in space_dg]
                self.intermediary_ero_term = [Function(space) for space in space_dg]
                self.intermediary_depo_term = [Function(space) for space in space_dg]

    def create_solutions(self):
        """
        Set up `Function`s in the prognostic spaces to hold forward and adjoint solutions.
        """
        self.fwd_solutions = [None for mesh in self.meshes]
        self.adj_solutions = [None for mesh in self.meshes]
        self.fwd_solutions_tracer = [None for mesh in self.meshes]
        self.fwd_solutions_sediment = [None for mesh in self.meshes]
        self.fwd_solutions_bathymetry = [None for mesh in self.meshes]
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
        if self.op.solve_sediment:
            self.fwd_solutions_sediment = [Function(Q, name="Forward sediment solution") for Q in self.Q]
            # self.adj_solutions_sediment = [Function(Q, name="Adjoint sediment solution") for Q in self.Q]
        if self.op.solve_exner:
            self.fwd_solutions_bathymetry = [Function(W, name="Forward bathymetry solution") for W in self.W]
            # self.adj_solutions_bathymetry = [Function(W, name="Adjoint bathymetry solution") for W in self.W]

    def set_fields(self, init=False):
        """
        Set various fields *on each mesh*, including:

            * viscosity;
            * diffusivity;
            * the Coriolis parameter;
            * drag coefficients;
            * bed roughness.

        The bathymetry is defined via a modified version of the `DepthExpression` found Thetis.
        """

        # Bathymetry
        if self.op.solve_exner:
            if init:
                for i, bathymetry in enumerate(self.fwd_solutions_bathymetry):
                    bathymetry.project(self.op.set_bathymetry(self.P1[i]))
                    self.op.create_sediment_model(self.P1[i].mesh(), bathymetry)
                self.depth = [None for bathymetry in self.fwd_solutions_bathymetry]
            for i, bathymetry in enumerate(self.fwd_solutions_bathymetry):
                self.depth[i] = DepthExpression(
                    bathymetry,
                    use_nonlinear_equations=self.shallow_water_options[i].use_nonlinear_equations,
                    use_wetting_and_drying=self.shallow_water_options[i].use_wetting_and_drying,
                    wetting_and_drying_alpha=self.shallow_water_options[i].wetting_and_drying_alpha,
                )
        elif self.op.solve_sediment:
            self.bathymetry = [self.op.set_bathymetry(P1) for P1 in self.P1]
            for i, bathymetry in enumerate(self.bathymetry):
                if init:
                    self.op.create_sediment_model(self.P1[i].mesh(), bathymetry)
                    self.depth = [None for bathymetry in self.bathymetry]
            for i, bathymetry in enumerate(self.bathymetry):
                self.depth[i] = DepthExpression(
                    bathymetry,
                    use_nonlinear_equations=self.shallow_water_options[i].use_nonlinear_equations,
                    use_wetting_and_drying=self.shallow_water_options[i].use_wetting_and_drying,
                    wetting_and_drying_alpha=self.shallow_water_options[i].wetting_and_drying_alpha,
                )
        else:
            self.bathymetry = [self.op.set_bathymetry(P1) for P1 in self.P1]
            self.depth = [None for bathymetry in self.bathymetry]
            for i, bathymetry in enumerate(self.bathymetry):
                self.depth[i] = DepthExpression(
                    bathymetry,
                    use_nonlinear_equations=self.shallow_water_options[i].use_nonlinear_equations,
                    use_wetting_and_drying=self.shallow_water_options[i].use_wetting_and_drying,
                    wetting_and_drying_alpha=self.shallow_water_options[i].wetting_and_drying_alpha,
                )

        self.fields = [AttrDict() for P1 in self.P1]
        for i, P1 in enumerate(self.P1):
            self.fields[i].update({
                'horizontal_viscosity': self.op.set_viscosity(P1),
                'horizontal_diffusivity': self.op.set_diffusivity(P1),
                'coriolis_frequency': self.op.set_coriolis(P1),
                'nikuradse_bed_roughness': self.op.ksp,
                'quadratic_drag_coefficient': self.op.set_quadratic_drag_coefficient(P1),
                'manning_drag_coefficient': self.op.set_manning_drag_coefficient(P1),
                'tracer_advective_velocity_factor': self.op.set_advective_velocity_factor(P1)
            })
        if self.op.solve_tracer:
            for i, P1DG in enumerate(self.P1DG):
                self.fields[i].update({
                    'tracer_source_2d': self.op.set_tracer_source(P1DG),
                })
        if self.op.solve_sediment or self.op.solve_exner:
            for i, P1DG in enumerate(self.P1DG):
                self.fields[i].update({
                    'sediment_source_2d': self.op.set_sediment_source(P1DG),
                    'sediment_depth_integ_source': self.op.set_sediment_depth_integ_source(P1DG),
                    'sediment_sink_2d': self.op.set_sediment_sink(P1DG),
                    'sediment_depth_integ_sink': self.op.set_sediment_depth_integ_sink(P1DG)
                })
        self.inflow = [self.op.set_inflow(P1_vec) for P1_vec in self.P1_vec]

        # Check CFL criterion
        if self.op.debug and hasattr(self.op, 'check_cfl_criterion'):
            self.op.check_cfl_criterion(self)

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
            self._set_tracer_stabilisation_step(i, sediment=False)
        if self.op.solve_sediment:
            self._set_tracer_stabilisation_step(i, sediment=True)

    def _set_shallow_water_stabilisation_step(self, i):
        op = self.op

        # Symmetric Interior Penalty Galerkin (SIPG) method
        sipg = None
        if op.family != 'cg-cg':
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
            assert op.family != 'cg-cg'
            assert hasattr(op, 'lax_friedrichs_velocity_scaling_factor')
            self.shallow_water_options[i]['lax_friedrichs_velocity_scaling_factor'] = op.lax_friedrichs_velocity_scaling_factor  # TODO: Allow mesh dependent
        else:
            msg = "Stabilisation method {:s} not recognised for {:s}"
            raise ValueError(msg.format(self.stabilisation, self.__class__.__name__))

    def _set_tracer_stabilisation_step(self, i, sediment=False):
        op = self.op
        eq_options = self.sediment_options if sediment else self.tracer_options

        # Symmetric Interior Penalty Galerkin (SIPG) method
        family = op.sediment_family if sediment else op.tracer_family
        sipg = None
        if family == 'dg':
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
            assert family == 'dg'
            eq_options[i]['lax_friedrichs_tracer_scaling_factor'] = op.lax_friedrichs_tracer_scaling_factor  # TODO: Allow mesh dependent
        elif self.stabilisation == 'su':
            assert family == 'cg'
            raise NotImplementedError  # TODO
        elif self.stabilisation == 'supg':
            assert family == 'cg'
            raise NotImplementedError  # TODO
        else:
            msg = "Stabilisation method {:s} not recognised for {:s}"
            raise ValueError(msg.format(self.stabilisation, self.__class__.__name__))

    # --- Solution initialisation and transfer

    def set_initial_condition(self, **kwargs):
        """Apply initial condition(s) for forward solution(s) on first mesh."""
        self.op.set_initial_condition(self, **kwargs)
        if self.op.solve_tracer:
            self.op.set_initial_condition_tracer(self)
        if self.op.solve_sediment:
            self.op.set_initial_condition_sediment(self)
        if self.op.solve_exner:
            self.op.set_initial_condition_bathymetry(self)

    def set_terminal_condition(self, **kwargs):
        """Apply terminal condition(s) for adjoint solution(s) on terminal mesh."""
        self.op.set_terminal_condition(self, **kwargs)
        if self.op.solve_tracer:
            self.op.set_terminal_condition_tracer(self, **kwargs)
        if self.op.solve_sediment:
            self.op.set_terminal_condition_sediment(self, **kwargs)
        if self.op.solve_exner:
            self.op.set_terminal_condition_exner(self, **kwargs)

    def project_forward_solution(self, i, j, **kwargs):
        """
        Project forward solution(s) from mesh `i` to mesh `j`.

        If the shallow water equations are not solved then the fluid velocity
        and surface elevation are set via the initial condition.
        """
        if self.op.solve_swe:
            self.project(self.fwd_solutions, i, j)
        else:
            self.op.set_initial_condition(self, **kwargs)
        if self.op.solve_tracer:
            self.project(self.fwd_solutions_tracer, i, j)
        if self.op.solve_sediment:
            self.project(self.fwd_solutions_sediment, i, j)
        if self.op.solve_exner:
            self.project(self.fwd_solutions_bathymetry, i, j)

    def project_adjoint_solution(self, i, j, **kwargs):
        """
        Project adjoint solution(s) from mesh `i` to mesh `j`.

        If the adjoint shallow water equations are not solved then the adjoint
        fluid velocity and surface elevation are set via the terminal condition.
        """
        if self.op.solve_swe:
            self.project(self.adj_solutions, i, j)
        else:
            self.op.set_terminal_condition(self, **kwargs)
        if self.op.solve_tracer:
            self.project(self.adj_solutions_tracer, i, j)
        if self.op.solve_sediment:
            self.project(self.adj_solutions_sediment, i, j)
        if self.op.solve_exner:
            self.project(self.adj_solutions_bathymetry, i, j)

    def project_to_intermediary_mesh(self, i):
        super(AdaptiveProblem, self).project_to_intermediary_mesh(i)
        if self.op.solve_tracer:
            self.intermediary_solutions_tracer[i].project(self.fwd_solutions_tracer[i])
        if self.op.solve_sediment:
            self.intermediary_solutions_sediment[i].project(self.fwd_solutions_sediment[i])
        if self.op.solve_exner:
            self.intermediary_solutions_bathymetry[i].project(self.fwd_solutions_bathymetry[i])
        if hasattr(self.op, 'sediment_model'):
            self.intermediary_solutions_old_bathymetry[i].project(self.op.sediment_model.old_bathymetry_2d)
            self.intermediary_solutions_uv_cg[i].project(self.op.sediment_model.uv_cg)
            self.intermediary_solutions_TOB[i].project(self.op.sediment_model.TOB)
            self.intermediary_solutions_depth[i].project(self.op.sediment_model.depth)

            if self.op.suspended:
                if self.op.convective_vel_flag:
                    self.intermediary_corr_vel_factor[i].project(self.op.sediment_model.corr_factor_model.corr_vel_factor)
                self.intermediary_coeff[i].project(self.op.sediment_model.coeff)
                self.intermediary_ceq[i].project(self.op.sediment_model.ceq)
                self.intermediary_equiltracer[i].project(self.op.sediment_model.equiltracer)
                self.intermediary_ero[i].project(self.op.sediment_model.ero)
                self.intermediary_ero_term[i].project(self.op.sediment_model.ero_term)
                self.intermediary_depo_term[i].project(self.op.sediment_model.depo_term)

        def debug(a, b, name):
            if np.allclose(a, b):
                print_output("WARNING: Is the intermediary {:s} solution just copied?".format(name))

        if self.op.debug:
            debug(self.fwd_solutions[i].dat.data[0],
                  self.intermediary_solutions[i].dat.data[0],
                  "velocity")
            debug(self.fwd_solutions[i].dat.data[1],
                  self.intermediary_solutions[i].dat.data[1],
                  "elevation")
            if self.op.solve_tracer:
                debug(self.fwd_solutions_tracer[i].dat.data,
                      self.intermediary_solutions_tracer[i].dat.data,
                      "tracer")
            if self.op.solve_sediment:
                debug(self.fwd_solutions_sediment[i].dat.data,
                      self.intermediary_solutions_sediment[i].dat.data,
                      "sediment")
            if self.op.solve_exner:
                debug(self.fwd_solutions_bathymetry[i].dat.data,
                      self.intermediary_solutions_bathymetry[i].dat.data,
                      "bathymetry")
            if hasattr(self.op, 'sediment_model'):
                debug(self.op.sediment_model.old_bathymetry_2d.dat.data,
                      self.intermediary_solutions_old_bathymetry[i].dat.data,
                      "old_bathymetry")
                debug(self.op.sediment_model.uv_cg.dat.data,
                      self.intermediary_solutions_uv_cg[i].dat.data,
                      "uv_cg")
                debug(self.op.sediment_model.TOB.dat.data,
                      self.intermediary_solutions_TOB[i].dat.data,
                      "TOB")
                debug(self.op.sediment_model.depth.dat.data,
                      self.intermediary_solutions_depth[i].dat.data,
                      "depth")
                if self.op.suspended:
                    if self.op.convective_vel_flag:
                        debug(self.op.sediment_model.corr_factor_model.corr_vel_factor.dat.data,
                              self.intermediary_corr_vel_factor[i].dat.data,
                              "corr_vel_factor")
                    debug(self.op.sediment_model.coeff.dat.data,
                          self.intermediary_coeff[i].dat.data,
                          "coeff")
                    debug(self.op.sediment_model.ceq.dat.data,
                          self.intermediary_ceq[i].dat.data,
                          "ceq")
                    debug(self.op.sediment_model.equiltracer.dat.data,
                          self.intermediary_equiltracer[i].dat.data,
                          "equiltracer")
                    debug(self.op.sediment_model.ero.dat.data,
                          self.intermediary_ero[i].dat.data,
                          "ero")
                    debug(self.op.sediment_model.ero_term.dat.data,
                          self.intermediary_ero_term[i].dat.data,
                          "ero_term")
                    debug(self.op.sediment_model.depo_term.dat.data,
                          self.intermediary_depo_term[i].dat.data,
                          "depo_term")

    def copy_data_from_intermediary_mesh(self, i):
        super(AdaptiveProblem, self).copy_data_from_intermediary_mesh(i)
        if self.op.solve_tracer:
            self.fwd_solutions_tracer[i].dat.data[:] = self.intermediary_solutions_tracer[i].dat.data
        if self.op.solve_sediment:
            self.fwd_solutions_sediment[i].dat.data[:] = self.intermediary_solutions_sediment[i].dat.data
        if self.op.solve_exner:
            self.fwd_solutions_bathymetry[i].dat.data[:] = self.intermediary_solutions_bathymetry[i].dat.data

        if hasattr(self.op, 'sediment_model'):
            self.op.sediment_model.old_bathymetry_2d.dat.data[:] = self.intermediary_solutions_old_bathymetry[i].dat.data
            self.op.sediment_model.uv_cg.dat.data[:] = self.intermediary_solutions_uv_cg[i].dat.data
            self.op.sediment_model.TOB.dat.data[:] = self.intermediary_solutions_TOB[i].dat.data
            self.op.sediment_model.depth.dat.data[:] = self.intermediary_solutions_depth[i].dat.data
            if self.op.suspended:
                if self.op.convective_vel_flag:
                    self.op.sediment_model.corr_factor_model.corr_vel_factor.dat.data[:] = self.intermediary_corr_vel_factor[i].dat.data
                self.op.sediment_model.coeff.dat.data[:] = self.intermediary_coeff[i].dat.data
                self.op.sediment_model.ceq.dat.data[:] = self.intermediary_ceq[i].dat.data
                self.op.sediment_model.equiltracer.dat.data[:] = self.intermediary_equiltracer[i].dat.data
                self.op.sediment_model.ero.dat.data[:] = self.intermediary_ero[i].dat.data
                self.op.sediment_model.ero_term.dat.data[:] = self.intermediary_ero_term[i].dat.data
                self.op.sediment_model.depo_term.dat.data[:] = self.intermediary_depo_term[i].dat.data
    # --- Equations

    def create_forward_equations(self, i):
        if self.op.solve_swe:
            self._create_forward_shallow_water_equations(i)
        if self.op.solve_tracer:
            self._create_forward_tracer_equation(i)
        if self.op.solve_sediment:
            self._create_forward_sediment_equation(i)
        if self.op.solve_exner:
            self._create_forward_exner_equation(i)

    def _create_forward_shallow_water_equations(self, i):
        from .swe.equation import ShallowWaterEquations

        if self.mesh_velocities[i] is not None:
            self.shallow_water_options[i]['mesh_velocity'] = self.mesh_velocities[i]
        self.equations[i].shallow_water = ShallowWaterEquations(
            self.V[i],
            self.depth[i],
            self.shallow_water_options[i],
        )
        self.equations[i].shallow_water.bnd_functions = self.boundary_conditions[i]['shallow_water']

    def _create_forward_tracer_equation(self, i):
        from .tracer.equation import TracerEquation2D, ConservativeTracerEquation2D

        op = self.tracer_options[i]
        conservative = op.use_tracer_conservative_form
        model = ConservativeTracerEquation2D if conservative else TracerEquation2D
        self.equations[i].tracer = model(
            self.Q[i],
            self.depth[i],
            use_lax_friedrichs=self.tracer_options[i].use_lax_friedrichs_tracer,
            sipg_parameter=self.tracer_options[i].sipg_parameter,
        )
        if op.use_limiter_for_tracers and self.Q[i].ufl_element().degree() > 0:
            self.tracer_limiters[i] = VertexBasedP1DGLimiter(self.Q[i])
        self.equations[i].tracer.bnd_functions = self.boundary_conditions[i]['tracer']

    def _create_forward_sediment_equation(self, i):
        from .sediment.equation import SedimentEquation2D

        op = self.sediment_options[i]
        model = SedimentEquation2D
        self.equations[i].sediment = model(
            self.Q[i],
            # self.op.sediment_model.depth_expr,
            self.depth[i],
            use_lax_friedrichs=self.tracer_options[i].use_lax_friedrichs_tracer,
            sipg_parameter=self.tracer_options[i].sipg_parameter,
            conservative=self.op.use_tracer_conservative_form,
        )
        if op.use_limiter_for_tracers and self.Q[i].ufl_element().degree() > 0:
            self.tracer_limiters[i] = VertexBasedP1DGLimiter(self.Q[i])
        self.equations[i].sediment.bnd_functions = self.boundary_conditions[i]['sediment']

    def _create_forward_exner_equation(self, i):
        from .sediment.exner_eq import ExnerEquation

        model = ExnerEquation
        self.equations[i].exner = model(
            self.W[i],
            self.depth[i],
            conservative=self.op.use_tracer_conservative_form,
            sed_model=self.op.sediment_model,
        )

    def create_adjoint_equations(self, i):
        if self.op.solve_swe:
            self._create_adjoint_shallow_water_equations(i)
        if self.op.solve_tracer:
            self._create_adjoint_tracer_equation(i)
        if self.op.solve_sediment:
            self._create_adjoint_sediment_equation(i)
        if self.op.solve_exner:
            self._create_adjoint_exner_equation(i)

    def _create_adjoint_shallow_water_equations(self, i):
        from .swe.adjoint import AdjointShallowWaterEquations

        self.equations[i].adjoint_shallow_water = AdjointShallowWaterEquations(
            self.V[i],
            self.depth[i],
            self.shallow_water_options[i],
        )
        self.equations[i].adjoint_shallow_water.bnd_functions = self.boundary_conditions[i]['shallow_water']

    def _create_adjoint_tracer_equation(self, i):
        from .tracer.adjoint import AdjointTracerEquation2D, AdjointConservativeTracerEquation2D

        op = self.tracer_options[i]
        conservative = op.use_tracer_conservative_form
        model = AdjointConservativeTracerEquation2D if conservative else AdjointTracerEquation2D
        self.equations[i].adjoint_tracer = model(
            self.Q[i],
            self.depth[i],
            use_lax_friedrichs=self.tracer_options[i].use_lax_friedrichs_tracer,
            sipg_parameter=self.tracer_options[i].sipg_parameter,
        )
        if op.use_limiter_for_tracers and self.Q[i].ufl_element().degree() > 0:
            self.tracer_limiters[i] = VertexBasedP1DGLimiter(self.Q[i])
        self.equations[i].adjoint_tracer.bnd_functions = self.boundary_conditions[i]['tracer']

    def _create_adjoint_sediment_equation(self, i):
        raise NotImplementedError("Continuous adjoint sediment equation not implemented")

    def _create_adjoint_exner_equation(self, i):
        raise NotImplementedError("Continuous adjoint Exner equation not implemented")

    # --- Error estimators

    def create_error_estimators(self, i):
        if self.op.solve_swe:
            self._create_shallow_water_error_estimator(i)
        if self.op.solve_tracer:
            self._create_tracer_error_estimator(i)
        if self.op.solve_sediment:
            self._create_sediment_error_estimator(i)
        if self.op.solve_exner:
            self._create_exner_error_estimator(i)

    def _create_shallow_water_error_estimator(self, i):
        from .swe.error_estimation import ShallowWaterGOErrorEstimator

        self.error_estimators[i].shallow_water = ShallowWaterGOErrorEstimator(
            self.V[i],
            self.depth[i],
            self.shallow_water_options[i],
        )

    def _create_tracer_error_estimator(self, i):
        from .tracer.error_estimation import TracerGOErrorEstimator

        if self.tracer_options[i].use_tracer_conservative_form:
            raise NotImplementedError("Error estimation for conservative tracers not implemented.")
        else:
            estimator = TracerGOErrorEstimator
        self.error_estimators[i].tracer = estimator(
            self.Q[i],
            self.depth[i],
            use_lax_friedrichs=self.tracer_options[i].use_lax_friedrichs_tracer,
            sipg_parameter=self.tracer_options[i].sipg_parameter,
        )

    def _create_sediment_error_estimator(self, i):
        raise NotImplementedError("Error estimators for sediment not implemented.")

    def _create_exner_error_estimator(self, i):
        raise NotImplementedError("Error estimators for Exner not implemented.")

    # --- Timestepping

    def create_forward_timesteppers(self, i):
        if i == 0:
            self.simulation_time = 0.0
        if self.op.solve_swe:
            self._create_forward_shallow_water_timestepper(i, self.integrator)
        if self.op.solve_tracer:
            self._create_forward_tracer_timestepper(i, self.integrator)
        if self.op.solve_sediment:
            self._create_forward_sediment_timestepper(i, self.integrator)
        if self.op.solve_exner:
            self._create_forward_exner_timestepper(i, self.integrator)

    def _get_fields_for_shallow_water_timestepper(self, i):
        fields = AttrDict({
            'linear_drag_coefficient': None,
            'quadratic_drag_coefficient': self.fields[i].quadratic_drag_coefficient,
            'manning_drag_coefficient': self.fields[i].manning_drag_coefficient,
            'nikuradse_bed_roughness': self.fields[i].nikuradse_bed_roughness,
            'viscosity_h': self.fields[i].horizontal_viscosity,
            'coriolis': self.fields[i].coriolis_frequency,
            'wind_stress': None,
            'atmospheric_pressure': None,
            'momentum_source': None,
            'volume_source': None,
        })
        if self.op.approach == 'lagrangian':
            raise NotImplementedError  # TODO
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
            'tracer_advective_velocity_factor': self.fields[i].tracer_advective_velocity_factor,
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

    def _get_fields_for_sediment_timestepper(self, i):
        u, eta = self.fwd_solutions[i].split()
        fields = AttrDict({
            'elev_2d': eta,
            'uv_2d': u,
            'diffusivity_h': self.fields[i].horizontal_diffusivity,
            'source': self.fields[i].sediment_source_2d,
            'depth_integrated_source': self.fields[i].sediment_depth_integ_source,
            'sink': self.fields[i].sediment_sink_2d,
            'depth_integrated_sink': self.fields[i].sediment_depth_integ_sink,
            'tracer_advective_velocity_factor': self.fields[i].tracer_advective_velocity_factor,
            'lax_friedrichs_tracer_scaling_factor': self.sediment_options[i].lax_friedrichs_tracer_scaling_factor,
            'mesh_velocity': None,
        })
        if self.mesh_velocities[i] is not None:
            fields['mesh_velocity'] = self.mesh_velocities[i]
        if self.op.approach == 'lagrangian':
            self.mesh_velocities[i] = u
            fields['uv_2d'] = Constant(as_vector([0.0, 0.0]))
        if self.stabilisation == 'lax_friedrichs':
            fields['lax_friedrichs_tracer_scaling_factor'] = self.sediment_options[i].lax_friedrichs_tracer_scaling_factor
        return fields

    def _get_fields_for_exner_timestepper(self, i):
        u, eta = self.fwd_solutions[i].split()
        fields = AttrDict({
            'elev_2d': eta,
            'source': self.fields[i].sediment_source_2d,
            'depth_integrated_source': self.fields[i].sediment_depth_integ_source,
            'sink': self.fields[i].sediment_sink_2d,
            'depth_integrated_sink': self.fields[i].sediment_depth_integ_sink,
            'sediment': self.fwd_solutions_sediment[i],
            'morfac': self.op.morphological_acceleration_factor,
            'porosity': self.op.porosity,
        })
        if self.mesh_velocities[i] is not None:
            fields['mesh_velocity'] = self.mesh_velocities[i]
        if self.op.approach == 'lagrangian':
            self.mesh_velocities[i] = u
            fields['uv_2d'] = Constant(as_vector([0.0, 0.0]))
        return fields

    def _create_forward_shallow_water_timestepper(self, i, integrator):
        fields = self._get_fields_for_shallow_water_timestepper(i)
        bcs = self.boundary_conditions[i]['shallow_water']
        kwargs = {'bnd_conditions': bcs}
        if self.op.timestepper == 'PressureProjectionPicard':
            from .swe.equation import ShallowWaterMomentumEquation

            self.equations[i].shallow_water_momentum = ShallowWaterMomentumEquation(
                TestFunction(self.V[i].sub(0)),
                self.V[i].sub(0),
                self.V[i].sub(1),
                self.depth[i],
                self.shallow_water_options[i],
            )
            self.equations[i].shallow_water_momentum.bnd_functions = bcs
            args = (self.equations[i].shallow_water, self.equations[i].shallow_water_momentum,
                    self.fwd_solutions[i], fields, self.op.dt, )
            kwargs['solver_parameters'] = self.op.solver_parameters_pressure
            kwargs['solver_parameters_mom'] = self.op.solver_parameters_momentum
            kwargs['iterations'] = self.op.picard_iterations
        else:
            args = (self.equations[i].shallow_water, self.fwd_solutions[i], fields, self.op.dt, )
            kwargs['solver_parameters'] = self.op.solver_parameters['shallow_water']
        if self.op.timestepper in ('CrankNicolson', 'PressureProjectionPicard'):
            kwargs['semi_implicit'] = self.op.use_semi_implicit_linearisation
            kwargs['theta'] = self.op.implicitness_theta
        if 'shallow_water' in self.error_estimators[i]:
            kwargs['error_estimator'] = self.error_estimators[i].shallow_water
        self.timesteppers[i].shallow_water = integrator(*args, **kwargs)

    def _create_forward_tracer_timestepper(self, i, integrator):
        fields = self._get_fields_for_tracer_timestepper(i)
        args = (self.equations[i].tracer, self.fwd_solutions_tracer[i], fields, self.op.dt, )
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

    def _create_forward_sediment_timestepper(self, i, integrator):
        fields = self._get_fields_for_sediment_timestepper(i)
        dt = self.op.dt
        args = (self.equations[i].sediment, self.fwd_solutions_sediment[i], fields, dt, )
        kwargs = {
            'bnd_conditions': self.boundary_conditions[i]['sediment'],
            'solver_parameters': self.op.solver_parameters['sediment'],
        }
        if self.op.timestepper == 'CrankNicolson':
            kwargs['semi_implicit'] = self.op.use_semi_implicit_linearisation
            kwargs['theta'] = self.op.implicitness_theta
        if 'sediment' in self.error_estimators[i]:
            kwargs['error_estimator'] = self.error_estimators[i].sediment
        self.timesteppers[i].sediment = integrator(*args, **kwargs)

    def _create_forward_exner_timestepper(self, i, integrator):
        fields = self._get_fields_for_exner_timestepper(i)
        dt = self.op.dt
        args = (self.equations[i].exner, self.fwd_solutions_bathymetry[i], fields, dt, )
        kwargs = {
            'solver_parameters': self.op.solver_parameters['exner'],
        }
        if self.op.timestepper == 'CrankNicolson':
            kwargs['semi_implicit'] = self.op.use_semi_implicit_linearisation
            kwargs['theta'] = self.op.implicitness_theta
        if 'exner' in self.error_estimators[i]:
            raise NotImplementedError
        self.timesteppers[i].exner = integrator(*args, **kwargs)

    def create_adjoint_timesteppers(self, i):
        if i == self.num_meshes-1:
            self.simulation_time = self.op.end_time
        if self.op.solve_swe:
            self._create_adjoint_shallow_water_timestepper(i, self.integrator)
        if self.op.solve_tracer:
            self._create_adjoint_tracer_timestepper(i, self.integrator)
        if self.op.solve_sediment:
            self._create_adjoint_sediment_timestepper(i, self.integrator)
        if self.op.solve_exner:
            self._create_adjoint_exner_timestepper(i, self.integrator)

    def _create_adjoint_shallow_water_timestepper(self, i, integrator):
        fields = self._get_fields_for_shallow_water_timestepper(i)
        fields['uv_2d'], fields['elev_2d'] = self.fwd_solutions[i].split()

        # Account for dJdq
        self.op.set_qoi_kernel(self, i)
        dJdu, dJdeta = self.kernels[i].split()
        self.time_kernel = Constant(1.0 if self.simulation_time >= self.op.start_time else 0.0)
        fields['dJdu'] = self.time_kernel*dJdu
        fields['dJdeta'] = self.time_kernel*dJdeta

        # Construct time integrator
        args = (self.equations[i].adjoint_shallow_water, self.adj_solutions[i], fields, self.op.dt, )
        kwargs = {
            'bnd_conditions': self.boundary_conditions[i]['shallow_water'],
            'solver_parameters': self.op.adjoint_solver_parameters['shallow_water'],
            'adjoint': True,  # Makes sure fields are updated according to appropriate timesteps
        }
        if self.op.timestepper == 'PressureProjectionPicard':
            raise NotImplementedError  # TODO
        if self.op.timestepper in ('CrankNicolson', 'PressureProjectionPicard'):
            kwargs['semi_implicit'] = self.op.use_semi_implicit_linearisation
            kwargs['theta'] = self.op.implicitness_theta
        self.timesteppers[i].adjoint_shallow_water = integrator(*args, **kwargs)

    def _create_adjoint_tracer_timestepper(self, i, integrator):
        fields = self._get_fields_for_tracer_timestepper(i)

        # fields.uv_2d = - fields.uv_2d

        # Account for dJdc
        dJdc = self.op.set_qoi_kernel_tracer(self, i)  # TODO: Store this kernel somewhere
        self.time_kernel = Constant(1.0 if self.simulation_time >= self.op.start_time else 0.0)
        fields['source'] = self.time_kernel*dJdc

        # Construct time integrator
        args = (self.equations[i].adjoint_tracer, self.adj_solutions_tracer[i], fields, self.op.dt, )
        kwargs = {
            'bnd_conditions': self.boundary_conditions[i]['tracer'],
            'solver_parameters': self.op.adjoint_solver_parameters['tracer'],
            'adjoint': True,  # Makes sure fields are updated according to appropriate timesteps
        }
        if self.op.timestepper == 'CrankNicolson':
            kwargs['semi_implicit'] = self.op.use_semi_implicit_linearisation
            kwargs['theta'] = self.op.implicitness_theta
        self.timesteppers[i].adjoint_tracer = integrator(*args, **kwargs)

    def _create_adjoint_sediment_timestepper(self, i, integrator):
        raise NotImplementedError("Continuous adjoint sediment timestepping not implemented")

    def _create_adjoint_exner_timestepper(self, i, integrator):
        raise NotImplementedError("Continuous adjoint Exner timestepping not implemented")

    # --- Solvers

    def add_callbacks(self, i):
        if self.op.solve_swe:
            self.callbacks[i].add(VelocityNormCallback(self, i), 'export')
            self.callbacks[i].add(ElevationNormCallback(self, i), 'export')
        if self.op.solve_tracer:
            self.callbacks[i].add(TracerNormCallback(self, i), 'export')
        if self.op.solve_sediment:
            self.callbacks[i].add(SedimentNormCallback(self, i), 'export')
        if self.op.solve_exner:
            self.callbacks[i].add(ExnerNormCallback(self, i), 'export')

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
        if op.solve_sediment:
            ts = self.timesteppers[i]['sediment']
            dbcs = []
            prob = NonlinearVariationalProblem(ts.F, ts.solution, bcs=dbcs)
            ts.solver = NonlinearVariationalSolver(prob, solver_parameters=ts.solver_parameters, options_prefix="forward_sediment")
            op.print_debug(op.indent + "SETUP: Adding callbacks on mesh {:d}...".format(i))
        if op.solve_exner:
            ts = self.timesteppers[i]['exner']
            dbcs = []
            prob = NonlinearVariationalProblem(ts.F, ts.solution, bcs=dbcs)
            ts.solver = NonlinearVariationalSolver(prob, solver_parameters=ts.solver_parameters, options_prefix="forward_exner")
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

        # Initialise counters
        t_epsilon = 1.0e-05
        self.iteration = 0
        start_time = i*op.dt*self.dt_per_mesh
        end_time = (i+1)*op.dt*self.dt_per_mesh
        try:
            assert np.allclose(self.simulation_time, start_time)
        except AssertionError:
            msg = "Mismatching start time: {:.2f} vs {:.2f}"
            raise ValueError(msg.format(self.simulation_time, start_time))
        # update_forcings(self.simulation_time)
        self.print(80*'=')
        op.print_debug("SOLVE: Entering forward timeloop on mesh {:d}...".format(i))
        if self.num_meshes == 1:
            msg = "FORWARD SOLVE  time {:8.2f}  ({:6.2f}) seconds"
            self.print(msg.format(self.simulation_time, 0.0))
        else:
            msg = "{:2d} {:s} FORWARD SOLVE mesh {:2d}/{:2d}  time {:8.2f}  ({:6.2f}) seconds"
            self.print(msg.format(self.outer_iteration, '  '*i, i+1,
                                  self.num_meshes, self.simulation_time, 0.0))
        cpu_timestamp = perf_counter()

        # Callbacks
        update_forcings = update_forcings or self.op.get_update_forcings(self, i, adjoint=False)
        export_func = export_func or self.op.get_export_func(self, i)
        # if i == 0:
        if export_func is not None:
            export_func()
        self.callbacks[i].evaluate(mode='export')
        self.callbacks[i].evaluate(mode='timestep')

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
        if op.solve_sediment and plot_pvd:
            proj_sediment = Function(self.P1[i], name="Projected sediment")
            self.sediment_file._topology = None
            if i == 0:
                proj_sediment.project(self.fwd_solutions_sediment[i])
                self.sediment_file.write(proj_sediment)
        if (op.plot_bathymetry or op.solve_exner) and plot_pvd:
            proj_bath = Function(self.P1[i], name="Projected bathymetry")
            self.exner_file._topology = None
            b = self.fwd_solutions_bathymetry[i] if op.solve_exner else self.bathymetry[i]
            if i == 0:
                proj_bath.project(b)
                self.exner_file.write(proj_bath)

        # Time integrate
        ts = self.timesteppers[i]
        while self.simulation_time <= end_time - t_epsilon:

            # Mesh movement
            if self.iteration % op.dt_per_mesh_movement == 0:
                self.move_mesh(i)

            # TODO: Update mesh velocity

            # Solve PDE(s)
            if op.solve_swe:
                ts.shallow_water.advance(self.simulation_time, update_forcings)
            if op.solve_tracer:
                ts.tracer.advance(self.simulation_time, update_forcings)
                if self.tracer_options[i].use_limiter_for_tracers:
                    self.tracer_limiters[i].apply(self.fwd_solutions_tracer[i])
            if op.solve_sediment:
                if op.solve_exner:
                    self.op.sediment_model.update(ts.shallow_water.solution, self.fwd_solutions_bathymetry[i])
                else:
                    self.op.sediment_model.update(ts.shallow_water.solution, self.bathymetry[i])
                ts.sediment.advance(self.simulation_time, update_forcings)
                if self.sediment_options[i].use_limiter_for_tracers:
                    self.tracer_limiters[i].apply(self.fwd_solutions_sediment[i])
            if op.solve_exner:
                if not op.solve_sediment:
                    self.op.sediment_model.update(ts.shallow_water.solution, self.fwd_solutions_bathymetry[i])
                ts.exner.advance(self.simulation_time, update_forcings)

            # Save to checkpoint
            if self.checkpointing:
                if op.solve_swe:
                    self.save_to_checkpoint(self.fwd_solutions[i])
                if op.solve_tracer:
                    self.save_to_checkpoint(self.fwd_solutions_tracer[i])
                if op.solve_sediment:
                    self.save_to_checkpoint(self.fwd_solutions_sediment[i])
                if op.solve_exner:
                    self.save_to_checkpoint(self.fwd_solutions_bathymetry[i])
                # TODO: Checkpoint mesh if moving

            # Export
            self.iteration += 1
            self.simulation_time += op.dt
            if self.iteration % op.dt_per_export == 0:
                cpu_time = perf_counter() - cpu_timestamp
                if self.num_meshes == 1:
                    self.print(msg.format(self.simulation_time, cpu_time))
                else:
                    self.print(msg.format(self.outer_iteration, '  '*i, i+1,
                                          self.num_meshes, self.simulation_time, cpu_time))
                cpu_timestamp = perf_counter()
                if op.solve_swe and plot_pvd:
                    u, eta = self.fwd_solutions[i].split()
                    proj_u.project(u)
                    proj_eta.project(eta)
                    self.solution_file.write(proj_u, proj_eta)
                if op.solve_tracer and plot_pvd:
                    proj_tracer.project(self.fwd_solutions_tracer[i])
                    self.tracer_file.write(proj_tracer)
                if op.solve_sediment and plot_pvd:
                    proj_sediment.project(self.fwd_solutions_sediment[i])
                    self.sediment_file.write(proj_sediment)
                if (op.plot_bathymetry or op.solve_exner) and plot_pvd:
                    b = self.fwd_solutions_bathymetry[i] if op.solve_exner else self.bathymetry[i]
                    proj_bath.project(b)
                    self.exner_file.write(proj_bath)
                if export_func is not None:
                    export_func()
                self.callbacks[i].evaluate(mode='export')
            self.callbacks[i].evaluate(mode='timestep')
        update_forcings(self.simulation_time + op.dt)
        self.print(80*'=')

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
            if op.family in ('cg-cg', 'dg-cg'):  # NOTE: This is inconsistent with forward
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

        # Initialise counters
        t_epsilon = 1.0e-05
        self.iteration = (i+1)*self.dt_per_mesh
        start_time = (i+1)*op.dt*self.dt_per_mesh
        end_time = i*op.dt*self.dt_per_mesh
        try:
            assert np.allclose(self.simulation_time, start_time)
        except AssertionError:
            msg = "Mismatching start time: {:f} vs {:f}"
            raise ValueError(msg.format(self.simulation_time, start_time))
        # update_forcings(self.simulation_time)
        self.print(80*'=')
        op.print_debug("SOLVE: Entering forward timeloop on mesh {:d}...".format(i))
        if self.num_meshes == 1:
            msg = "ADJOINT SOLVE time {:8.2f}  ({:6.2f} seconds)"
            self.print(msg.format(self.simulation_time, 0.0))
        else:
            msg = "{:2d} {:s}  ADJOINT SOLVE mesh {:2d}/{:2d}  time {:8.2f}  ({:6.2f} seconds)"
            self.print(msg.format(self.outer_iteration, '  '*i, i+1,
                                  self.num_meshes, self.simulation_time, 0.0))
        cpu_timestamp = perf_counter()

        # Callbacks
        update_forcings = update_forcings or self.op.get_update_forcings(self, i, adjoint=True)
        export_func = export_func or self.op.get_export_func(self, i)
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

        # Time integrate
        ts = self.timesteppers[i]
        while self.simulation_time >= end_time + t_epsilon:
            self.time_kernel.assign(1.0 if self.simulation_time >= self.op.start_time else 0.0)

            # Collect forward solution from checkpoint and free associated memory
            #   NOTE: We need collect the checkpoints from the stack in reverse order
            if self.checkpointing:
                if op.solve_exner:
                    self.fwd_solutions_bathymetry[i].assign(self.collect_from_checkpoint())
                if op.solve_sediment:
                    self.fwd_solutions_sediment[i].assign(self.collect_from_checkpoint())
                if op.solve_tracer:
                    self.fwd_solutions_tracer[i].assign(self.collect_from_checkpoint())
                if op.solve_swe:
                    self.fwd_solutions[i].assign(self.collect_from_checkpoint())

            # Solve adjoint PDE(s)
            if op.solve_swe:
                ts.adjoint_shallow_water.advance(self.simulation_time, update_forcings)
            if op.solve_tracer:
                ts.adjoint_tracer.advance(self.simulation_time, update_forcings)
                if self.tracer_options[i].use_limiter_for_tracers:
                    self.tracer_limiters[i].apply(self.adj_solutions_tracer[i])

            # Increment counters
            self.iteration -= 1
            self.simulation_time -= op.dt
            self.callbacks[i].evaluate(mode='timestep')

            # Export
            if self.iteration % op.dt_per_export == 0:
                cpu_time = perf_counter() - cpu_timestamp
                cpu_timestamp = perf_counter()
                if self.num_meshes == 1:
                    self.print(msg.format(self.simulation_time, cpu_time))
                else:
                    self.print(msg.format(self.outer_iteration, '  '*i, i+1,
                                          self.num_meshes, self.simulation_time, cpu_time))
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
        self.time_kernel.assign(1.0 if self.simulation_time >= self.op.start_time else 0.0)
        update_forcings(self.simulation_time - op.dt)
        self.print(80*'=')

    # --- Metric

    # TODO: Allow Hessian metric for tracer / sediment / Exner
    def get_hessian_metric(self, adjoint=False, **kwargs):
        kwargs.setdefault('normalise', True)
        kwargs['op'] = self.op
        self.metrics = []
        solutions = self.adj_solutions if adjoint else self.fwd_solutions
        for i, sol in enumerate(solutions):
            fields = {'bathymetry': self.bathymetry[i], 'inflow': self.inflow[i]}
            self.metrics.append(get_hessian_metric(sol, fields=fields, **kwargs))

    # --- Run scripts

    # TODO: Allow adaptation to tracer / sediment / Exner
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
            self.print("Quantity of interest {:d}: {:.4e}".format(n+1, qoi))
            self.qois.append(qoi)
            if len(self.qois) > 1:
                if np.abs(self.qois[-1] - self.qois[-2]) < op.qoi_rtol*self.qois[-2]:
                    self.print("Converged quantity of interest!")
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

            self.print("\nStarting mesh adaptation for iteration {:d}...".format(n+1))
            for i, M in enumerate(metrics):
                self.print("Adapting mesh {:d}/{:d}...".format(i+1, self.num_meshes))
                self.meshes[i] = pragmatic_adapt(self.meshes[i], M, op=op)
            del metrics
            self.num_cells.append([mesh.num_cells() for mesh in self.meshes])
            self.num_vertices.append([mesh.num_vertices() for mesh in self.meshes])

            # ---  Setup for next run / logging

            self.setup_all(self.meshes)
            self.dofs.append([np.array(V.dof_count).sum() for V in self.V])

            self.print("\nResulting meshes")
            msg = "  {:2d}: complexity {:8.1f} vertices {:7d} elements {:7d}"
            for i, c in enumerate(complexities):
                self.print(msg.format(i, c, self.num_vertices[n+1][i], self.num_cells[n+1][i]))
            msg = "  total:            {:8.1f}          {:7d}          {:7d}\n"
            self.print(msg.format(
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
                self.print("Converged number of mesh elements!")
                break

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
        self.indicator_file = File(os.path.join(self.di, 'indicator.pvd'))
        for n in range(op.num_adapt):
            self.outer_iteration = n

            # --- Solve forward to get checkpoints

            # self.get_checkpoints()
            self.solve_forward()

            # --- Convergence criteria

            # Check QoI convergence
            qoi = self.quantity_of_interest()
            self.print("Quantity of interest {:d}: {:.4e}".format(n+1, qoi))
            self.qois.append(qoi)
            if len(self.qois) > 1:
                if np.abs(self.qois[-1] - self.qois[-2]) < op.qoi_rtol*self.qois[-2]:
                    self.print("Converged quantity of interest!")
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

            self.print("\nStarting mesh adaptation for iteration {:d}...".format(n+1))
            for i, M in enumerate(metrics):
                self.print("Adapting mesh {:d}/{:d}...".format(i+1, self.num_meshes))
                self.meshes[i] = pragmatic_adapt(self.meshes[i], M, op=op)
            del metrics
            self.num_cells.append([mesh.num_cells() for mesh in self.meshes])
            self.num_vertices.append([mesh.num_vertices() for mesh in self.meshes])

            # ---  Setup for next run / logging

            self.setup_all(self.meshes)
            self.dofs.append([np.array(V.dof_count).sum() for V in self.V])

            self.print("\nResulting meshes")
            msg = "  {:2d}: complexity {:8.1f} vertices {:7d} elements {:7d}"
            for i, c in enumerate(complexities):
                self.print(msg.format(i, c, self.num_vertices[n+1][i], self.num_cells[n+1][i]))
            msg = "  total:            {:8.1f}          {:7d}          {:7d}\n"
            self.print(msg.format(
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
                self.print("Converged number of mesh elements!")
                break

    # TODO: Allow adaptation to tracer / sediment / Exner
    # TODO: Enable move to base class
    def run_dwr(self, **kwargs):
        # TODO: doc
        op = self.op
        self.indicator_file = File(os.path.join(self.di, 'indicator.pvd'))
        for n in range(op.num_adapt):
            self.outer_iteration = n

            # --- Solve forward to get checkpoints

            # self.get_checkpoints()
            self.solve_forward()

            # --- Convergence criteria

            # Check QoI convergence
            qoi = self.quantity_of_interest()
            self.print("Quantity of interest {:d}: {:.4e}".format(n+1, qoi))
            self.qois.append(qoi)
            if len(self.qois) > 1:
                if np.abs(self.qois[-1] - self.qois[-2]) < op.qoi_rtol*self.qois[-2]:
                    self.print("Converged quantity of interest!")
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
                self.print("All meshes are identical so we use an identical hierarchy.")
                hierarchy = MeshHierarchy(self.meshes[0], 1)
                refined_meshes = [hierarchy[1] for mesh in self.meshes]
            else:
                self.print("Meshes differ so we create separate hierarchies.")
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

            self.print("\nStarting mesh adaptation for iteration {:d}...".format(n+1))
            for i, M in enumerate(metrics):
                self.print("Adapting mesh {:d}/{:d}...".format(i+1, self.num_meshes))
                self.meshes[i] = pragmatic_adapt(self.meshes[i], M, op=op)
            del metrics
            self.num_cells.append([mesh.num_cells() for mesh in self.meshes])
            self.num_vertices.append([mesh.num_vertices() for mesh in self.meshes])

            # ---  Setup for next run / logging

            self.setup_all(self.meshes)
            self.dofs.append([np.array(V.dof_count).sum() for V in self.V])

            self.print("\nResulting meshes")
            msg = "  {:2d}: complexity {:8.1f} vertices {:7d} elements {:7d}"
            for i, c in enumerate(complexities):
                self.print(msg.format(i, c, self.num_vertices[n+1][i], self.num_cells[n+1][i]))
            msg = "  total:            {:8.1f}          {:7d}          {:7d}\n"
            self.print(msg.format(
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
                self.print("Converged number of mesh elements!")
                break
