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
from ..io import *
from .options import ReynoldsNumberArray
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
            swo.tidal_turbine_farms = {}
            if hasattr(op, 'sipg_parameter') and op.sipg_parameter is not None:
                swo['sipg_parameter'] = op.sipg_parameter
        if not nonlinear:
            for model in op.solver_parameters:
                op.solver_parameters[model]['snes_type'] = 'ksponly'
                op.adjoint_solver_parameters[model]['snes_type'] = 'ksponly'
        if op.debug:
            for model in op.solver_parameters:
                op.solver_parameters[model]['ksp_converged_reason'] = None
                op.solver_parameters[model]['snes_converged_reason'] = None
                op.adjoint_solver_parameters[model]['ksp_converged_reason'] = None
                op.adjoint_solver_parameters[model]['snes_converged_reason'] = None
                if op.debug_mode == 'full':
                    op.solver_parameters[model]['ksp_monitor'] = None
                    op.solver_parameters[model]['snes_monitor'] = None
                    op.adjoint_solver_parameters[model]['ksp_monitor'] = None
                    op.adjoint_solver_parameters[model]['snes_monitor'] = None
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
            to.anisotropic_stabilisation = op.anisotropic_stabilisation
        for i, to in enumerate(self.sediment_options):
            to.update(static_options)
            if hasattr(op, 'sipg_parameter_sediment') and op.sipg_parameter_sediment is not None:
                swo['sipg_parameter_sediment'] = op.sipg_parameter_sediment
            to.anisotropic_stabilisation = op.anisotropic_stabilisation

        # Lists to be populated
        self.fwd_solutions = [None for i in range(op.num_meshes)]
        self.adj_solutions = [None for i in range(op.num_meshes)]
        self.fwd_solutions_tracer = [None for i in range(op.num_meshes)]
        self.fwd_solutions_sediment = [None for i in range(op.num_meshes)]
        self.fwd_solutions_bathymetry = [None for i in range(op.num_meshes)]
        self.adj_solutions_tracer = [None for i in range(op.num_meshes)]
        self.fields = [AttrDict() for i in range(op.num_meshes)]
        self.depth = [None for i in range(op.num_meshes)]
        self.bathymetry = [None for i in range(op.num_meshes)]
        self.inflow = [None for i in range(op.num_meshes)]
        self.minimum_angles = [None for i in range(op.num_meshes)]

        super(AdaptiveProblem, self).__init__(op, nonlinear=nonlinear, **kwargs)

        # Custom arrays
        self.reynolds_number = ReynoldsNumberArray(self.meshes, op)

    @property
    def mesh(self):
        return self.meshes[0]

    @property
    def fwd_solution(self):
        return self.fwd_solutions[0]

    @property
    def adj_solution(self):
        return self.adj_solutions[0]

    @property
    def fwd_solution_tracer(self):
        return self.fwd_solutions_tracer[0]

    @property
    def adj_solution_tracer(self):
        return self.adj_solutions_tracer[0]

    def create_outfiles(self):
        if not self.op.plot_pvd:
            return
        if self.op.solve_swe:
            super(AdaptiveProblem, self).create_outfiles()
        if self.op.solve_tracer:
            self.tracer_file = File(os.path.join(self.di, 'tracer.pvd'))
            self.adjoint_tracer_file = File(os.path.join(self.di, 'adjoint_tracer.pvd'))
        if self.op.solve_sediment:
            self.sediment_file = File(os.path.join(self.di, 'sediment.pvd'))
        if self.op.recover_vorticity:
            self.vorticity_file = File(os.path.join(self.di, 'vorticity.pvd'))
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
        self.op.print_debug("SETUP: Creating finite elements...")
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
        Build finite element spaces for the prognostic solutions of each model, along with various
        other useful spaces.

        The shallow water space is denoted `V`, the tracer and sediment space is denoted `Q` and the
        bathymetry space is denoted `W`.
        """
        super(AdaptiveProblem, self).create_function_spaces()

        # Shallow water space
        self.V = [FunctionSpace(mesh, self.finite_element) for mesh in self.meshes]

        # Tracer space(s)
        if self.op.solve_tracer:
            self.Q = [FunctionSpace(mesh, self.finite_element_tracer) for mesh in self.meshes]
        elif self.op.solve_sediment:  # TODO: What if we want both, in different spaces?
            self.Q = [FunctionSpace(mesh, self.finite_element_sediment) for mesh in self.meshes]
        else:
            self.Q = [None for mesh in self.meshes]

        # Bathymetry space
        if self.op.solve_exner:
            self.W = [FunctionSpace(mesh, self.finite_element_bathymetry) for mesh in self.meshes]
        else:
            self.W = [None for mesh in self.meshes]

        # Record DOFs
        self.dofs = [[np.array(V.dof_count).sum() for V in self.V], ]  # TODO: other function spaces

    def get_function_space(self, field):
        space = {'shallow_water': 'V', 'tracer': 'Q', 'sediment': 'Q', 'bathymetry': 'W'}[field]
        return self.__getattribute__(space)

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

    def create_solutions_step(self, i):
        super(AdaptiveProblem, self).create_solutions_step(i)
        u, eta = self.fwd_solutions[i].split()
        u.rename("Fluid velocity")
        eta.rename("Elevation")
        z, zeta = self.adj_solutions[i].split()
        z.rename("Adjoint fluid velocity")
        zeta.rename("Adjoint elevation")
        if self.op.solve_tracer:
            self.fwd_solutions_tracer[i] = Function(self.Q[i], name="Forward tracer solution")
            self.adj_solutions_tracer[i] = Function(self.Q[i], name="Adjoint tracer solution")
        if self.op.solve_sediment:
            self.fwd_solutions_sediment[i] = Function(self.Q[i], name="Forward sediment solution")
            # self.adj_solutions_sediment[i] = Function(self.Q[i], name="Adjoint sediment solution")
        if self.op.solve_exner:
            self.fwd_solutions_bathymetry[i] = Function(self.W[i], name="Forward bathymetry solution")
            # self.adj_solutions_bathymetry[i] = Function(self.W[i], name="Adjoint bathymetry solution")

    def get_solutions(self, field, adjoint=False):
        name = 'adj_solutions' if adjoint else 'fwd_solutions'
        fields = ('tracer', 'sediment', 'bathymetry')
        if field in fields:
            name = '_'.join([name, field])
        return self.__getattribute__(name)

    def free_solutions_step(self, i):
        super(AdaptiveProblem, self).free_solutions_step(i)
        if self.op.solve_tracer:
            self.fwd_solutions_tracer[i] = None
            self.adj_solutions_tracer[i] = None
        if self.op.solve_sediment:
            self.fwd_solutions_sediment[i] = None
            self.adj_solutions_sediment[i] = None
        if self.op.solve_exner:
            self.fwd_solutions_bathymetry[i] = None
            self.adj_solutions_bathymetry[i] = None

    def set_fields_step(self, i, init=False):

        # Bathymetry
        if self.op.solve_exner:
            if init:
                self.fwd_solutions_bathymetry[i].project(self.op.set_bathymetry(self.P1[i]))
                self.op.create_sediment_model(self.P1[i].mesh(), self.fwd_solutions_bathymetry[i])
            self.depth[i] = DepthExpression(
                self.fwd_solutions_bathymetry[i],
                use_nonlinear_equations=self.shallow_water_options[i].use_nonlinear_equations,
                use_wetting_and_drying=self.shallow_water_options[i].use_wetting_and_drying,
                wetting_and_drying_alpha=self.shallow_water_options[i].wetting_and_drying_alpha,
            )
        else:
            self.bathymetry[i] = self.op.set_bathymetry(self.P1[i])
            if self.op.solve_sediment and init:
                self.op.create_sediment_model(self.P1[i].mesh(), self.bathymetry[i])
            self.depth[i] = DepthExpression(
                self.bathymetry[i],
                use_nonlinear_equations=self.shallow_water_options[i].use_nonlinear_equations,
                use_wetting_and_drying=self.shallow_water_options[i].use_wetting_and_drying,
                wetting_and_drying_alpha=self.shallow_water_options[i].wetting_and_drying_alpha,
            )

        self.fields[i].update({
            'horizontal_viscosity': self.op.set_viscosity(self.P1[i]),
            'horizontal_diffusivity': self.op.set_diffusivity(self.P1[i]),
            'coriolis_frequency': self.op.set_coriolis(self.P1[i]),
            'nikuradse_bed_roughness': self.op.ksp,
            'quadratic_drag_coefficient': self.op.set_quadratic_drag_coefficient(self.P1[i]),
            'manning_drag_coefficient': self.op.set_manning_drag_coefficient(self.P1[i]),
            'tracer_advective_velocity_factor': self.op.set_advective_velocity_factor(self.P1[i]),
        })
        if self.op.solve_tracer:
            self.fields[i].update({
                'tracer_source_2d': self.op.set_tracer_source(self.P1DG[i]),
            })
        if self.op.solve_sediment or self.op.solve_exner:
            self.fields[i].update({
                'sediment_source_2d': self.op.set_sediment_source(self.P1DG[i]),
                'sediment_depth_integ_source': self.op.set_sediment_depth_integ_source(self.P1DG[i]),
                'sediment_sink_2d': self.op.set_sediment_sink(self.P1DG[i]),
                'sediment_depth_integ_sink': self.op.set_sediment_depth_integ_sink(self.P1DG[i]),
            })
        self.inflow = [self.op.set_inflow(P1_vec) for P1_vec in self.P1_vec]

    def free_fields_step(self, i):
        super(AdaptiveProblem, self).free_fields_step(i)
        self.bathymetry[i] = None
        self.depth[i] = None
        self.inflow[i] = None

    # --- Stabilisation

    def set_stabilisation_step(self, i):
        """ Set stabilisation mode and corresponding parameter on the ith mesh."""
        if self.op.use_automatic_sipg_parameter:
            self.minimum_angles[i] = get_minimum_angles_2d(self.meshes[i])
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
                cot_theta = 1.0/tan(self.minimum_angles[i])

                # Penalty parameter for shallow water
                nu = self.fields[i].horizontal_viscosity
                if nu is not None:
                    p = self.V[i].sub(0).ufl_element().degree()
                    alpha = Constant(5.0*p*(p+1)) if p != 0 else 1.5
                    alpha = alpha*get_sipg_ratio(nu)*cot_theta
                    sipg = interpolate(alpha, self.P0[i])

            # Set parameter and print to screen
            self.shallow_water_options[i].sipg_parameter = sipg
            if sipg is None:
                pass
            elif isinstance(sipg, Constant):
                msg = "SETUP: constant shallow water SIPG parameter on mesh {:d}: {:.4e}"
                op.print_debug(msg.format(i, sipg.dat.data[0]))
            else:
                msg = "SETUP: variable shallow water SIPG parameter on mesh {:d}: min {:.4e} max {:.4e}"
                with sipg.dat.vec_ro as v:
                    op.print_debug(msg.format(i, v.min()[1], v.max()[1]))

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
                sipg = op.sipg_parameter_tracer if not sediment else None
            if hasattr(op, 'sipg_parameter_sediment'):
                sipg = op.sipg_parameter_sediment if sediment else None
            if eq_options[i].use_automatic_sipg_parameter:
                cot_theta = 1.0/tan(self.minimum_angles[i])

                # Penalty parameter for tracers
                nu = self.fields[i].horizontal_diffusivity
                if nu is not None:
                    p = self.Q[i].ufl_element().degree()
                    alpha = Constant(5.0*p*(p+1)) if p != 0 else 1.5
                    alpha = alpha*get_sipg_ratio(nu)*cot_theta
                    sipg = interpolate(alpha, self.P0[i])

            # Set parameter and print to screen
            eq_options[i].sipg_parameter = sipg
            model = 'sediment' if sediment else 'tracer'
            if sipg is None:
                pass
            elif isinstance(sipg, Constant):
                msg = "SETUP: constant {:s} SIPG parameter on mesh {:d}: {:.4e}"
                op.print_debug(msg.format(model, i, sipg.dat.data[0]))
            else:
                msg = "SETUP: variable {:s} SIPG parameter on mesh {:d}: min {:.4e} max {:.4e}"
                with sipg.dat.vec_ro as v:
                    op.print_debug(msg.format(model, i, v.min()[1], v.max()[1]))

        # Stabilisation
        eq_options[i]['lax_friedrichs_tracer_scaling_factor'] = None
        if self.stabilisation is None:
            return
        elif self.stabilisation == 'lax_friedrichs':
            assert hasattr(op, 'lax_friedrichs_tracer_scaling_factor')
            assert family == 'dg'
            eq_options[i]['lax_friedrichs_tracer_scaling_factor'] = op.lax_friedrichs_tracer_scaling_factor  # TODO: Allow mesh dependent
        elif self.stabilisation == 'su':
            assert family == 'cg'
            eq_options[i]['su_stabilisation'] = True
        elif self.stabilisation == 'supg':
            assert family == 'cg'
            assert self.op.timestepper == 'SteadyState'  # TODO
            eq_options[i]['supg_stabilisation'] = True
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

    def compute_mesh_reynolds_number(self, i):
        # u, eta = self.fwd_solutions[i].split()
        u = self.op.characteristic_velocity
        nu = self.fields[i].horizontal_viscosity
        self.reynolds_number[i] = (u, nu)
        if self.op.plot_pvd:
            if not hasattr(self, 'reynolds_number_file'):
                self.reynolds_number_file = File(os.path.join(self.di, 'reynolds_number.pvd'))
            self.reynolds_number_file._topology = None
            self.reynolds_number_file.write(self.reynolds_number[i])

    def plot_mesh_reynolds_number(self, i, axes=None, **kwargs):
        import matplotlib.pyplot as plt

        if axes is None:
            fig, axes = plt.subplots()
        if self.reynolds_number[i] is None:
            self.compute_mesh_reynolds_number(i)
        Re = self.reynolds_number[i]
        Re_vec = Re.vector().gather()
        kwargs.setdefault('levels', np.linspace(0.99*Re_vec.min(), 1.01*Re_vec.max(), 50))
        kwargs.setdefault('cmap', 'coolwarm')
        return tricontourf(Re, axes=axes, **kwargs)

    def transfer_forward_solution(self, i, **kwargs):
        super(AdaptiveProblem, self).transfer_forward_solution(i, **kwargs)

        # Check Reynolds and CFL numbers
        if self.op.debug and self.op.solve_swe:
            self.compute_mesh_reynolds_number(i)
            if hasattr(self.op, 'check_cfl_criterion'):
                self.op.check_cfl_criterion(self, i, error_factor=None)
                # TODO: parameter for error_factor, defaulted by timestepper choice
                # TODO: allow t-adaptation on subinterval

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

        # TODO: Stash the below as metadata
        fns = (self.fwd_solutions_tracer, self.fwd_solutions_sediment, self.fwd_solutions_bathymetry)
        flgs = (self.op.solve_tracer, self.op.solve_sediment, self.op.solve_exner)
        names = ("tracer", "sediment", "bathymetry")
        spaces = (self.Q, self.Q, self.W)

        # Project between spaces, constructing if necessary
        for flg, f, name, space in zip(flgs, fns, names, spaces):
            if flg:
                if f[i] is None:
                    raise ValueError("Nothing to project.")
                elif f[j] is None:
                    f[j] = Function(space[j], name="Forward {:s} solution".format(name))
                self.project(f, i, j)

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

        # TODO: Stash the below as metadata
        fns = (self.adj_solutions_tracer, self.adj_solutions_sediment, self.adj_solutions_bathymetry)
        flgs = (self.op.solve_tracer, self.op.solve_sediment, self.op.solve_exner)
        names = ("tracer", "sediment", "bathymetry")
        spaces = (self.Q, self.Q, self.W)

        # Project between spaces, constructing if necessary
        for flg, f, name, space in zip(flgs, fns, names, spaces):
            if flg:
                if f[i] is None:
                    raise ValueError("Nothing to project.")
                elif f[j] is None:
                    f[j] = Function(space[j], name="Adjoint {:s} solution".format(name))
                self.project(f, i, j)

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
            if not self.op.debug:
                return
            if np.allclose(a, b):
                print_output("WARNING: Is the intermediary {:s} solution just copied?".format(name))

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

    # TODO: Use par_loop
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

    # --- I/O

    def export_state(self, i, fpath, plexname=None, **kwargs):
        op = self.op
        kwargs['plexname'] = plexname
        kwargs['op'] = op
        if op.solve_swe:
            op.print_debug("I/O: Exporting hydrodynamics to {:s}...".format(fpath))
            export_hydrodynamics(*self.fwd_solutions[i].split(), fpath, **kwargs)
        if op.solve_tracer:
            name = 'tracer'
            op.print_debug("I/O: Exporting {:s} to {:s}...".format(name, fpath))
            export_field(self.fwd_solutions_tracer[i], name, name, fpath, **kwargs)
        if op.solve_sediment:
            name = 'sediment'
            op.print_debug("I/O: Exporting {:s} to {:s}...".format(name, fpath))
            export_field(self.fwd_solutions_sediment[i], name, name, fpath, **kwargs)
        if op.solve_exner:
            name = 'bathymetry'
            op.print_debug("I/O: Exporting {:s} to {:s}...".format(name, fpath))
            export_field(self.fwd_solutions_bathymetry[i], name, name, fpath, **kwargs)

    def load_state(self, i, fpath, plexname=None, **kwargs):
        op = self.op
        kwargs['outputdir'] = self.di
        kwargs['op'] = op
        if op.solve_swe:
            kwargs['plexname'] = plexname
            op.print_debug("I/O: Loading hydrodynamics from {:s}...".format(fpath))
            u_init, eta_init = initialise_hydrodynamics(fpath, **kwargs)
            u, eta = self.fwd_solutions[i].split()
            u.project(u_init)
            eta.project(eta_init)
        if op.solve_tracer:
            name = 'tracer'
            op.print_debug("I/O: Loading {:s} from {:s}...".format(name, fpath))
            args = (self.Q[i], name, name, fpath)
            self.fwd_solutions_tracer[i].project(initialise_field(*args, **kwargs))
        if op.solve_sediment:
            name = 'sediment'
            op.print_debug("I/O: Loading {:s} from {:s}...".format(name, fpath))
            args = (self.Q[i], name, name, fpath)
            self.fwd_solutions_sediment[i].project(initialise_field(*args, **kwargs))
        if op.solve_exner:
            name = 'bathymetry'
            op.print_debug("I/O: Loading {:s} from {:s}...".format(name, fpath))
            args = (self.W[i], name, name, fpath)
            self.fwd_solutions_bathymetry[i].project(initialise_field(*args, **kwargs))

    # --- Equations

    def create_forward_equations_step(self, i):
        if self.op.solve_swe:
            self.create_forward_shallow_water_equations_step(i)
        if self.op.solve_tracer:
            self.create_forward_tracer_equation_step(i)
        if self.op.solve_sediment:
            self.create_forward_sediment_equation_step(i)
        if self.op.solve_exner:
            self.create_forward_exner_equation_step(i)

    def create_forward_shallow_water_equations_step(self, i):
        from .swe.equation import ShallowWaterEquations

        if self.mesh_velocities[i] is not None:
            self.shallow_water_options[i]['mesh_velocity'] = self.mesh_velocities[i]
        self.equations[i].shallow_water = ShallowWaterEquations(
            self.V[i],
            self.depth[i],
            self.shallow_water_options[i],
        )
        self.equations[i].shallow_water.bnd_functions = self.boundary_conditions[i]['shallow_water']

    def create_forward_tracer_equation_step(self, i):
        from .tracer.equation import TracerEquation2D, ConservativeTracerEquation2D

        op = self.tracer_options[i]
        conservative = op.use_tracer_conservative_form
        model = ConservativeTracerEquation2D if conservative else TracerEquation2D
        self.equations[i].tracer = model(
            self.Q[i],
            self.depth[i],
            use_lax_friedrichs=self.tracer_options[i].use_lax_friedrichs_tracer,
            sipg_parameter=self.tracer_options[i].sipg_parameter,
            anisotropic=op.anisotropic_stabilisation,
        )
        if op.use_limiter_for_tracers and self.Q[i].ufl_element().degree() > 0:
            self.tracer_limiters[i] = VertexBasedP1DGLimiter(self.Q[i])
        self.equations[i].tracer.bnd_functions = self.boundary_conditions[i]['tracer']

    def create_forward_sediment_equation_step(self, i):
        from .sediment.equation import SedimentEquation2D

        op = self.sediment_options[i]
        model = SedimentEquation2D
        self.equations[i].sediment = model(
            self.Q[i],
            # self.op.sediment_model.depth_expr,
            self.depth[i],
            use_lax_friedrichs=self.sediment_options[i].use_lax_friedrichs_tracer,
            sipg_parameter=self.sediment_options[i].sipg_parameter,
            conservative=self.op.use_tracer_conservative_form,
        )
        if op.use_limiter_for_tracers and self.Q[i].ufl_element().degree() > 0:
            self.tracer_limiters[i] = VertexBasedP1DGLimiter(self.Q[i])
        self.equations[i].sediment.bnd_functions = self.boundary_conditions[i]['sediment']

    def create_forward_exner_equation_step(self, i):
        from .sediment.exner_eq import ExnerEquation

        model = ExnerEquation
        self.equations[i].exner = model(
            self.W[i],
            self.depth[i],
            conservative=self.op.use_tracer_conservative_form,
            sed_model=self.op.sediment_model,
        )

    def free_forward_equations_step(self, i):
        if self.op.solve_swe:
            self.free_forward_shallow_water_equations_step(i)
        if self.op.solve_tracer:
            self.free_forward_tracer_equation_step(i)
        if self.op.solve_sediment:
            self.free_forward_sediment_equation_step(i)
        if self.op.solve_exner:
            self.free_forward_exner_equation_step(i)

    def free_forward_shallow_water_equations_step(self, i):
        delattr(self.equations[i], 'shallow_water')

    def free_forward_tracer_equation_step(self, i):
        delattr(self.equations[i], 'tracer')

    def free_forward_sediment_equation_step(self, i):
        delattr(self.equations[i], 'sediment')

    def free_forward_exner_equation_step(self, i):
        delattr(self.equations[i], 'exner')

    def create_adjoint_equations_step(self, i):
        if self.op.solve_swe:
            self.create_adjoint_shallow_water_equations_step(i)
        if self.op.solve_tracer:
            self.create_adjoint_tracer_equation_step(i)
        if self.op.solve_sediment:
            self.create_adjoint_sediment_equation_step(i)
        if self.op.solve_exner:
            self.create_adjoint_exner_equation_step(i)

    def create_adjoint_shallow_water_equations_step(self, i):
        from .swe.adjoint import AdjointShallowWaterEquations

        self.equations[i].adjoint_shallow_water = AdjointShallowWaterEquations(
            self.V[i],
            self.depth[i],
            self.shallow_water_options[i],
        )
        self.equations[i].adjoint_shallow_water.bnd_functions = self.boundary_conditions[i]['shallow_water']

    def create_adjoint_tracer_equation_step(self, i):
        from .tracer.equation import TracerEquation2D, ConservativeTracerEquation2D

        op = self.tracer_options[i]
        conservative = op.use_tracer_conservative_form
        model = TracerEquation2D if conservative else ConservativeTracerEquation2D
        self.equations[i].adjoint_tracer = model(
            self.Q[i],
            self.depth[i],
            use_lax_friedrichs=self.tracer_options[i].use_lax_friedrichs_tracer,
            sipg_parameter=self.tracer_options[i].sipg_parameter,
            anisotropic=op.anisotropic_stabilisation,
        )
        if op.use_limiter_for_tracers and self.Q[i].ufl_element().degree() > 0:
            self.tracer_limiters[i] = VertexBasedP1DGLimiter(self.Q[i])
        self.equations[i].adjoint_tracer.bnd_functions = self.boundary_conditions[i]['tracer']

    def create_adjoint_sediment_equation_step(self, i):
        raise NotImplementedError("Continuous adjoint sediment equation not implemented")

    def create_adjoint_exner_equation_step(self, i):
        raise NotImplementedError("Continuous adjoint Exner equation not implemented")

    def free_adjoint_equations_step(self, i):
        if self.op.solve_swe:
            self.free_adjoint_shallow_water_equations_step(i)
        if self.op.solve_tracer:
            self.free_adjoint_tracer_equation_step(i)
        if self.op.solve_sediment:
            self.free_adjoint_sediment_equation_step(i)
        if self.op.solve_exner:
            self.free_adjoint_exner_equation_step(i)

    def free_adjoint_shallow_water_equations_step(self, i):
        delattr(self.equations[i], 'adjoint_shallow_water')

    def free_adjoint_tracer_equation_step(self, i):
        delattr(self.equations[i], 'adjoint_tracer')

    def free_adjoint_sediment_equation_step(self, i):
        raise NotImplementedError("Continuous adjoint sediment equation not implemented")

    def free_adjoint_exner_equation_step(self, i):
        raise NotImplementedError("Continuous adjoint Exner equation not implemented")

    # --- Error estimators

    def create_error_estimators_step(self, i):
        if self.op.solve_swe:
            self.create_shallow_water_error_estimator_step(i)
        if self.op.solve_tracer:
            self.create_tracer_error_estimator_step(i)
        if self.op.solve_sediment:
            self.create_sediment_error_estimator_step(i)
        if self.op.solve_exner:
            self.create_exner_error_estimator_step(i)

    def create_shallow_water_error_estimator_step(self, i):
        from .swe.error_estimation import ShallowWaterGOErrorEstimator

        self.error_estimators[i].shallow_water = ShallowWaterGOErrorEstimator(
            self.V[i],
            self.depth[i],
            self.shallow_water_options[i],
        )

    def create_tracer_error_estimator_step(self, i):
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

    def create_sediment_error_estimator_step(self, i):
        raise NotImplementedError("Error estimators for sediment not implemented.")

    def create_exner_error_estimator_step(self, i):
        raise NotImplementedError("Error estimators for Exner not implemented.")

    # --- Timestepping

    def create_forward_timesteppers_step(self, i):
        if i == 0:
            self.simulation_time = 0.0
        if self.op.solve_swe:
            self.create_forward_shallow_water_timestepper_step(i, self.integrator)
        if self.op.solve_tracer:
            self.create_forward_tracer_timestepper_step(i, self.integrator)
        if self.op.solve_sediment:
            self.create_forward_sediment_timestepper_step(i, self.integrator)
        if self.op.solve_exner:
            self.create_forward_exner_timestepper_step(i, self.integrator)

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
        if self.op.solve_swe:
            # u, eta = self.fwd_solutions[i].split()  # FIXME: Not fully annotated
            u, eta = split(self.fwd_solutions[i])  # FIXME: Not fully annotated
        else:
            # u = Constant(as_vector(self.op.base_velocity))  # FIXME: Pyadjoint doesn't like this
            u = interpolate(as_vector(self.op.base_velocity), self.P1_vec[i])
            eta = Constant(0.0)
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
        elif self.stabilisation == 'su':
            fields['su_stabilisation'] = True
        elif self.stabilisation == 'supg':
            fields['supg_stabilisation'] = True
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

    def create_forward_shallow_water_timestepper_step(self, i, integrator):
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

    def create_forward_tracer_timestepper_step(self, i, integrator):
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

    def create_forward_sediment_timestepper_step(self, i, integrator):
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

    def create_forward_exner_timestepper_step(self, i, integrator):
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

    def free_forward_timesteppers_step(self, i):
        if self.op.solve_swe:
            self.free_forward_shallow_water_timestepper_step(i)
        if self.op.solve_tracer:
            self.free_forward_tracer_timestepper_step(i)
        if self.op.solve_sediment:
            self.free_forward_sediment_timestepper_step(i)
        if self.op.solve_exner:
            self.free_forward_exner_timestepper_step(i)

    def free_forward_shallow_water_timestepper_step(self, i):
        delattr(self.timesteppers[i], 'shallow_water')

    def free_forward_tracer_timestepper_step(self, i):
        delattr(self.timesteppers[i], 'tracer')

    def free_forward_sediment_timestepper_step(self, i):
        delattr(self.timesteppers[i], 'sediment')

    def free_forward_exner_timestepper_step(self, i):
        delattr(self.timesteppers[i], 'exner')

    def create_adjoint_timesteppers_step(self, i):
        if i == self.num_meshes-1:
            self.simulation_time = self.op.end_time
        if self.op.solve_swe:
            self.create_adjoint_shallow_water_timestepper_step(i, self.integrator)
        if self.op.solve_tracer:
            self.create_adjoint_tracer_timestepper_step(i, self.integrator)
        if self.op.solve_sediment:
            self.create_adjoint_sediment_timestepper_step(i, self.integrator)
        if self.op.solve_exner:
            self.create_adjoint_exner_timestepper_step(i, self.integrator)

    def create_adjoint_shallow_water_timestepper_step(self, i, integrator):
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

    def create_adjoint_tracer_timestepper_step(self, i, integrator):
        fields = self._get_fields_for_tracer_timestepper(i)

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

    def create_adjoint_sediment_timestepper_step(self, i, integrator):
        raise NotImplementedError("Continuous adjoint sediment timestepping not implemented")

    def create_adjoint_exner_timestepper_step(self, i, integrator):
        raise NotImplementedError("Continuous adjoint Exner timestepping not implemented")

    def free_adjoint_timesteppers_step(self, i):
        if self.op.solve_swe:
            self.free_adjoint_shallow_water_timestepper_step(i)
        if self.op.solve_tracer:
            self.free_adjoint_tracer_timestepper_step(i)
        if self.op.solve_sediment:
            self.free_adjoint_sediment_timestepper_step(i)
        if self.op.solve_exner:
            self.free_adjoint_exner_timestepper_step(i)

    def free_adjoint_shallow_water_timestepper_step(self, i):
        delattr(self.timesteppers[i], 'adjoint_shallow_water')

    def free_adjoint_tracer_timestepper_step(self, i):
        delattr(self.timesteppers[i], 'adjoint_tracer')

    def free_adjoint_sediment_timestepper_step(self, i):
        raise NotImplementedError("Continuous adjoint sediment timestepping not implemented")

    def free_adjoint_exner_timestepper_step(self, i):
        raise NotImplementedError("Continuous adjoint Exner timestepping not implemented")

    # --- Solvers

    def add_callbacks(self, i):
        from thetis.callback import CallbackManager

        # Create a new CallbackManager object on every mesh
        #   NOTE: This overwrites any pre-existing CallbackManagers
        self.op.print_debug("SETUP: Creating CallbackManagers...")
        self.callbacks[i] = CallbackManager()

        # Add default callbacks
        if self.op.solve_swe:
            self.callbacks[i].add(VelocityNormCallback(self, i), 'export')
            self.callbacks[i].add(ElevationNormCallback(self, i), 'export')
        if self.op.solve_tracer:
            self.callbacks[i].add(TracerNormCallback(self, i), 'export')
        if self.op.solve_sediment:
            self.callbacks[i].add(SedimentNormCallback(self, i), 'export')
        if self.op.solve_exner:
            self.callbacks[i].add(ExnerNormCallback(self, i), 'export')
        if self.op.recover_vorticity:
            if not hasattr(self, 'vorticity'):
                self.vorticity = [None for mesh in self.meshes]
            self.callbacks[i].add(VorticityNormCallback(self, i), 'export')

    def setup_solver_forward_step(self, i):
        """
        Setup forward solver on mesh `i`.
        """
        op = self.op
        op.print_debug("SETUP: Creating forward equations on mesh {:d}...".format(i))
        self.create_forward_equations_step(i)
        op.print_debug("SETUP: Creating forward timesteppers on mesh {:d}...".format(i))
        self.create_forward_timesteppers_step(i)
        bcs = self.boundary_conditions[i]
        if op.solve_swe:
            ts = self.timesteppers[i]['shallow_water']
            dbcs = []
            if op.family == 'cg-cg':
                op.print_debug("SETUP: Applying DirichletBCs on mesh {:d}...".format(i))
                for j in bcs['shallow_water']:
                    if 'elev' in bcs['shallow_water'][j]:
                        dbcs.append(DirichletBC(self.V[i].sub(1), bcs['shallow_water'][j]['elev'], j))
            prob = NonlinearVariationalProblem(ts.F, ts.solution, bcs=dbcs)
            ts.solver = NonlinearVariationalSolver(prob, solver_parameters=ts.solver_parameters, options_prefix="forward")
        if op.solve_tracer:
            ts = self.timesteppers[i]['tracer']
            dbcs = []
            if op.tracer_family == 'cg':
                op.print_debug("SETUP: Applying tracer DirichletBCs on mesh {:d}...".format(i))
                for j in bcs['tracer']:
                    if 'value' in bcs['tracer'][j]:
                        dbcs.append(DirichletBC(self.Q[i], bcs['tracer'][j]['value'], j))
            prob = NonlinearVariationalProblem(ts.F, ts.solution, bcs=dbcs)
            ts.solver = NonlinearVariationalSolver(prob, solver_parameters=ts.solver_parameters, options_prefix="forward_tracer")
        if op.solve_sediment:
            ts = self.timesteppers[i]['sediment']
            dbcs = []
            prob = NonlinearVariationalProblem(ts.F, ts.solution, bcs=dbcs)
            ts.solver = NonlinearVariationalSolver(prob, solver_parameters=ts.solver_parameters, options_prefix="forward_sediment")
        if op.solve_exner:
            ts = self.timesteppers[i]['exner']
            dbcs = []
            prob = NonlinearVariationalProblem(ts.F, ts.solution, bcs=dbcs)
            ts.solver = NonlinearVariationalSolver(prob, solver_parameters=ts.solver_parameters, options_prefix="forward_exner")
        op.print_debug("SETUP: Adding callbacks on mesh {:d}...".format(i))
        self.add_callbacks(i)

    def free_solver_forward_step(self, i):
        op = self.op
        op.print_debug("FREE: Removing forward timesteppers on mesh {:d}...".format(i))
        self.free_forward_timesteppers_step(i)
        op.print_debug("FREE: Removing forward equations on mesh {:d}...".format(i))
        self.free_forward_equations_step(i)

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
            indent = '' if op.debug else '  '*i
            self.print(msg.format(self.outer_iteration, indent, i+1, self.num_meshes,
                                  self.simulation_time, 0.0))
        cpu_timestamp = perf_counter()

        # Callbacks
        update_forcings = update_forcings or self.op.get_update_forcings(self, i, adjoint=False)
        export_func = export_func or self.op.get_export_func(self, i)
        if i == 0:
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
                    indent = '' if op.debug else '  '*i
                    self.print(msg.format(self.outer_iteration, indent, i+1, self.num_meshes,
                                          self.simulation_time, cpu_time))
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
        if update_forcings is not None:
            update_forcings(self.simulation_time + op.dt)
        self.print(80*'=')

    def setup_solver_adjoint_step(self, i):
        """Setup forward solver on mesh `i`."""
        op = self.op
        op.print_debug("SETUP: Creating adjoint equations on mesh {:d}...".format(i))
        self.create_adjoint_equations_step(i)
        op.print_debug("SETUP: Creating adjoint timesteppers on mesh {:d}...".format(i))
        self.create_adjoint_timesteppers_step(i)
        bcs = self.boundary_conditions[i]
        if op.solve_swe:
            dbcs = []
            ts = self.timesteppers[i]['adjoint_shallow_water']
            if op.family in ('cg-cg', 'dg-cg'):  # NOTE: This is inconsistent with forward
                op.print_debug("SETUP: Applying adjoint DirichletBCs on mesh {:d}...".format(i))
                for j in bcs['shallow_water']:
                    if 'un' not in bcs['shallow_water'][j]:
                        dbcs.append(DirichletBC(self.V[i].sub(1), 0, j))
            prob = NonlinearVariationalProblem(ts.F, ts.solution, bcs=dbcs)
            ts.solver = NonlinearVariationalSolver(prob, solver_parameters=ts.solver_parameters, options_prefix="adjoint")
        if op.solve_tracer:
            ts = self.timesteppers[i]['adjoint_tracer']
            dbcs = []
            if op.tracer_family == 'cg':
                op.print_debug("SETUP: Applying adjoint tracer DirichletBCs on mesh {:d}...".format(i))
                for j in bcs['tracer']:
                    if 'diff_flux' not in bcs['tracer'][j]:
                        dbcs.append(DirichletBC(self.Q[i], 0, j))
            prob = NonlinearVariationalProblem(ts.F, ts.solution, bcs=dbcs)
            ts.solver = NonlinearVariationalSolver(prob, solver_parameters=ts.solver_parameters, options_prefix="adjoint_tracer")
        if op.solve_sediment:
            raise NotImplementedError
        if op.solve_exner:
            raise NotImplementedError

    def free_solver_adjoint_step(self, i):
        op = self.op
        op.print_debug("FREE: Removing adjoint timesteppers on mesh {:d}...".format(i))
        self.free_adjoint_timesteppers_step(i)
        op.print_debug("FREE: Removing adjoint equations on mesh {:d}...".format(i))
        self.free_adjoint_equations_step(i)

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
            indent = '' if op.debug else '  '*i
            msg = "{:2d} {:s}  ADJOINT SOLVE mesh {:2d}/{:2d}  time {:8.2f}  ({:6.2f} seconds)"
            self.print(msg.format(self.outer_iteration, indent, i+1, self.num_meshes,
                                  self.simulation_time, 0.0))
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

            # Export
            if self.iteration % op.dt_per_export == 0:
                cpu_time = perf_counter() - cpu_timestamp
                cpu_timestamp = perf_counter()
                if self.num_meshes == 1:
                    self.print(msg.format(self.simulation_time, cpu_time))
                else:
                    indent = '' if op.debug else '  '*i
                    self.print(msg.format(self.outer_iteration, indent, i+1, self.num_meshes,
                                          self.simulation_time, cpu_time))
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

    def recover_hessian_metric(self, adjoint=False, **kwargs):
        kwargs.setdefault('normalise', True)
        kwargs['op'] = self.op
        self.metrics = []
        solutions = self.get_solutions(self.op.adapt_field, adjoint=adjoint)
        if self.op.adapt_field in ('tracer', 'sediment', 'bathymetry'):
            for i, sol in enumerate(solutions):
                self.metrics.append(steady_metric(sol, **kwargs))
        else:
            for i, sol in enumerate(solutions):
                fields = {'bathymetry': self.bathymetry[i], 'inflow': self.inflow[i]}
                self.metrics.append(recover_hessian_metric(sol, fields=fields, **kwargs))

    def get_recovery(self, i, **kwargs):
        op = self.op
        if (not op.solve_swe) or op.solve_tracer or op.solve_sediment or op.solve_exner:
            raise NotImplementedError  # TODO: allow Hessians of tracer fields, etc.
        if op.approach == 'vorticity':  # TODO: Use recoverer stashed in callback
            recoverer = L2ProjectorVorticity(self.V[i], op=op)
        else:
            recoverer = ShallowWaterHessianRecoverer(
                self.V[i], op=op,
                constant_fields={'bathymetry': self.bathymetry[i]}, **kwargs,
            )
        return recoverer


    # --- Run scripts

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
        for n in range(op.max_adapt):
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
            if n == op.max_adapt - 1:
                break

            # --- Loop over mesh windows *in reverse*

            for i, P1 in enumerate(self.P1):
                self.indicators[i]['dwp'] = Function(P1, name="DWP indicator")
            metrics = [Function(P1_ten, name="Metric") for P1_ten in self.P1_ten]
            for i in reversed(range(self.num_meshes)):
                fwd_solutions_step = []
                adj_solutions_step = []

                # --- Solve forward on current window

                def export_func():
                    fwd_solutions_step.append(self.fwd_solutions[i].copy(deepcopy=True))

                self.simulation_time = i*op.dt*self.dt_per_mesh
                self.transfer_forward_solution(i)
                self.setup_solver_forward_step(i)
                self.solve_forward_step(i, export_func=export_func, plot_pvd=False)

                # --- Solve adjoint on current window

                def export_func():
                    adj_solutions_step.append(self.adj_solutions[i].copy(deepcopy=True))

                self.transfer_adjoint_solution(i)
                self.setup_solver_adjoint_step(i)
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

            if op.debug and op.plot_pvd:
                metric_file = File(os.path.join(self.di, 'metric_before_normalisation.pvd'))
                for i, M in enumerate(metrics):
                    metric_file._topology = None
                    metric_file.write(M)

            space_time_normalise(metrics, op=op)

            # Output to .pvd and .vtu
            if op.plot_pvd:
                self.indicator_file = File(os.path.join(self.di, 'indicator.pvd'))
                metric_file = File(os.path.join(self.di, 'metric.pvd'))
            complexities = []
            for i, M in enumerate(metrics):
                if op.plot_pvd:
                    self.indicator_file._topology = None
                    self.indicator_file.write(self.indicators[i]['dwp'])
                    metric_file._topology = None
                    metric_file.write(M)
                complexities.append(metric_complexity(M))
            self.st_complexities.append(sum(complexities)*op.end_time/op.dt)

            # --- Adapt meshes

            self.print("\nStarting mesh adaptation for iteration {:d}...".format(n+1))
            for i, M in enumerate(metrics):
                self.print("Adapting mesh {:d}/{:d}...".format(i+1, self.num_meshes))
                self.meshes[i] = pragmatic_adapt(self.meshes[i], M)
            del metrics

            # ---  Setup for next run / logging

            self.set_meshes(self.meshes)
            self.setup_all()
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

    def run(self, **kwargs):
        """
        Run simulation using mesh adaptation approach specified by `self.approach`.

        For metric-based approaches, a fixed point iteration loop is used.
        """
        run_scripts = {

            # Non-adaptive
            'fixed_mesh': self.solve_forward,

            # Metric-based, no adjoint
            'hessian': self.run_hessian_based,
            'vorticity': self.run_hessian_based,  # TODO: Change name and update docs

            # Metric-based with adjoint
            'dwp': self.run_dwp,
            'dwr': self.run_dwr,
            'a_posteriori': self.run_dwr,
            'a_priori': self.run_dwr,
        }
        try:
            run_scripts[self.approach](**kwargs)
        except KeyError:
            raise ValueError("Approach '{:s}' not recognised".format(self.approach))

    # TODO: Enable move to base class
    def run_dwr(self, **kwargs):
        # TODO: doc
        op = self.op
        adapt_field = op.adapt_field
        if adapt_field not in ('tracer', 'sediment', 'bathymetry'):
            adapt_field = 'shallow_water'
        for n in range(op.max_adapt):
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
            if n == op.max_adapt - 1:
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
            enriched_space = ep.get_function_space(adapt_field)

            # --- Loop over mesh windows *in reverse*

            for i, P1 in enumerate(self.P1):
                self.indicators[i]['dwr'] = Function(P1, name="DWR indicator")
            metrics = [Function(P1_ten, name="Metric") for P1_ten in self.P1_ten]
            for i in reversed(range(self.num_meshes)):
                fwd_solutions_step = []
                fwd_solutions_step_old = []
                adj_solutions_step = []
                enriched_adj_solutions_step = []
                tm = dmhooks.get_transfer_manager(self.get_plex(i))

                # --- Setup forward solver for enriched problem

                # TODO: Need to transfer fwd sol in nonlinear case
                ep.create_error_estimators_step(i)  # These get passed to the timesteppers under the hood
                ep.setup_solver_forward_step(i)
                ets = ep.timesteppers[i][adapt_field]

                # --- Solve forward on current window

                ts = self.timesteppers[i][adapt_field]

                def export_func():
                    fwd_solutions_step.append(ts.solution.copy(deepcopy=True))
                    fwd_solutions_step_old.append(ts.solution_old.copy(deepcopy=True))
                    # TODO: Also need store fields at each export (in general case)

                self.simulation_time = i*op.dt*self.dt_per_mesh
                self.transfer_forward_solution(i)
                self.setup_solver_forward_step(i)
                self.solve_forward_step(i, export_func=export_func, plot_pvd=False)

                # --- Solve adjoint on current window

                def export_func():
                    adj_solutions_step.append(self.adj_solutions[i].copy(deepcopy=True))

                self.transfer_adjoint_solution(i)
                self.setup_solver_adjoint_step(i)
                self.solve_adjoint_step(i, export_func=export_func, plot_pvd=False)

                # --- Solve adjoint on current window in enriched space

                def export_func():
                    enriched_adj_solutions_step.append(ep.adj_solutions[i].copy(deepcopy=True))

                ep.simulation_time = (i+1)*op.dt*self.dt_per_mesh  # TODO: Shouldn't be needed
                ep.transfer_adjoint_solution(i)
                ep.setup_solver_adjoint_step(i)
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
                fwd_proj = Function(enriched_space[i])
                fwd_old_proj = Function(enriched_space[i])
                adj_error = Function(enriched_space[i])
                bcs = self.boundary_conditions[i][adapt_field]
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
                indicator_enriched_cts = interpolate(I, ep.P1[i])  # TODO: Project?
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

            if op.debug and op.plot_pvd:
                metric_file = File(os.path.join(self.di, 'metric_before_normalisation.pvd'))
                for i, M in enumerate(metrics):
                    metric_file._topology = None
                    metric_file.write(M)

            space_time_normalise(metrics, op=op)

            # Output to .pvd and .vtu
            if op.plot_pvd:
                self.indicator_file = File(os.path.join(self.di, 'indicator.pvd'))
                metric_file = File(os.path.join(self.di, 'metric.pvd'))
            complexities = []
            for i, M in enumerate(metrics):
                if op.plot_pvd:
                    self.indicator_file._topology = None
                    self.indicator_file.write(self.indicators[i]['dwr'])
                    metric_file._topology = None
                    metric_file.write(M)
                complexities.append(metric_complexity(M))
            self.st_complexities.append(sum(complexities)*op.end_time/op.dt)

            # --- Adapt meshes

            self.print("\nStarting mesh adaptation for iteration {:d}...".format(n+1))
            for i, M in enumerate(metrics):
                self.print("Adapting mesh {:d}/{:d}...".format(i+1, self.num_meshes))
                self.meshes[i] = pragmatic_adapt(self.meshes[i], M)
            del metrics

            # ---  Setup for next run / logging

            self.set_meshes(self.meshes)
            self.setup_all()
            base_space = self.get_function_space(adapt_field)
            self.dofs.append([np.sum(fs.dof_count) for fs in base_space])

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
