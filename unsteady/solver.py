from __future__ import absolute_import

from thetis import *
from thetis.limiter import VertexBasedP1DGLimiter

import numpy as np
import os
from time import perf_counter

from ..adapt.metric import *
from .base import AdaptiveProblemBase
from .callback import *
from ..io import *
from ..mesh import anisotropic_cell_size
from ..options import ReynoldsNumberArray
from ..swe.utils import *


__all__ = ["AdaptiveProblem"]


# TODO:
#  * Mesh movement ALE formulation
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
        self.stabilisation_tracer = op.stabilisation_tracer
        self.sediment_options = [AttrDict() for i in range(op.num_meshes)]
        self.stabilisation_sediment = op.stabilisation_sediment
        self.exner_options = [AttrDict() for i in range(op.num_meshes)]
        static_options = {
            'use_automatic_sipg_parameter': op.use_automatic_sipg_parameter,
            # 'check_tracer_conservation': True,  # TODO
            'use_lax_friedrichs_tracer': op.stabilisation_tracer == 'lax_friedrichs',
            'use_limiter_for_tracers': op.use_limiter_for_tracers and op.tracer_family == 'dg',
            'sipg_parameter': None,
            'use_tracer_conservative_form': op.use_tracer_conservative_form,
        }
        if op.use_tracer_conservative_form and op.approach in ('lagrangian', 'hybrid'):
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
        self.kernels_tracer = [None for i in range(op.num_meshes)]

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

    def create_outfiles(self, restarted=False):
        if not self.op.plot_pvd:
            return
        if self.op.solve_swe:
            super(AdaptiveProblem, self).create_outfiles(restarted=restarted)
        if self.op.solve_tracer:
            if restarted:
                self.tracer_file._topology = None
                self.adjoint_tracer_file._topology = None
            else:
                self.tracer_file = File(os.path.join(self.di, 'tracer.pvd'))
                self.adjoint_tracer_file = File(os.path.join(self.di, 'adjoint_tracer.pvd'))
        if self.op.solve_sediment:
            if restarted:
                self.sediment_file._topology = None
            else:
                self.sediment_file = File(os.path.join(self.di, 'sediment.pvd'))
        if self.op.recover_vorticity:
            if restarted:
                self.vorticity_file._topology = None
            else:
                self.vorticity_file = File(os.path.join(self.di, 'vorticity.pvd'))
        if self.op.plot_bathymetry or self.op.solve_exner:
            if restarted:
                self.exner_file._topology = None
            else:
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
            assert p in (1, 2)
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
        op = self.op

        # Shallow water space
        self.V = [FunctionSpace(mesh, self.finite_element) for mesh in self.meshes]

        # Tracer space(s)
        if op.solve_tracer:
            assert not op.solve_sediment
            self.Q = [FunctionSpace(mesh, self.finite_element_tracer) for mesh in self.meshes]
        elif op.solve_sediment:
            self.Q = [FunctionSpace(mesh, self.finite_element_sediment) for mesh in self.meshes]
        else:
            self.Q = [None for mesh in self.meshes]

        # Bathymetry space
        if op.solve_exner:
            self.W = [FunctionSpace(mesh, self.finite_element_bathymetry) for mesh in self.meshes]
        else:
            self.W = [None for mesh in self.meshes]

        # Diffusivity space
        self.diffusivity_space = [FunctionSpace(mesh, op.diffusivity_space_family, op.diffusivity_space_degree) for mesh in self.meshes]

        # Record DOFs
        self.dofs = [[np.array(V.dof_count).sum() for V in self.V], ]  # TODO: other function spaces

    def get_function_space(self, field):
        spaces = {'shallow_water': 'V', 'tracer': 'Q', 'sediment': 'Q', 'bathymetry': 'W'}
        space = spaces[field]
        try:
            return self.__getattribute__(space)
        except KeyError:
            return self.V

    def create_intermediary_spaces(self, have_intermediaries=False):
        super(AdaptiveProblem, self).create_intermediary_spaces()
        if have_intermediaries or self.op.approach not in ('monge_ampere', 'hybrid'):
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
            'horizontal_diffusivity': self.op.set_diffusivity(self.diffusivity_space[i]),
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
        """
        Set stabilisation mode and corresponding parameter on the ith mesh.
        """
        dim = self.meshes[i].topological_dimension()
        if self.op.use_automatic_sipg_parameter:
            if dim == 2:
                self.minimum_angles[i] = get_minimum_angles_2d(self.meshes[i])
            else:
                print_output("WARNING: Cannot compute minimum angle in {:d}D.".format(dim))
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
                    alpha = Constant(5.0*p*(p+1) if p != 0 else 1.5)
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
        stabilisation = None if self.stabilisation is None else self.stabilisation.lower()
        if stabilisation is None:
            return
        elif stabilisation == 'lax_friedrichs':
            assert op.family != 'cg-cg'
            assert hasattr(op, 'lax_friedrichs_velocity_scaling_factor')
            self.shallow_water_options[i]['lax_friedrichs_velocity_scaling_factor'] = op.lax_friedrichs_velocity_scaling_factor  # TODO: Allow mesh dependent
        else:
            msg = "Stabilisation method {:s} not recognised for {:s}"
            raise ValueError(msg.format(stabilisation, self.__class__.__name__))

    def _set_tracer_stabilisation_step(self, i, sediment=False):
        op = self.op
        eq_options = self.sediment_options if sediment else self.tracer_options
        stabilisation = self.stabilisation_sediment if sediment else self.stabilisation_tracer
        stabilisation = None if stabilisation is None else stabilisation.lower()

        # Symmetric Interior Penalty Galerkin (SIPG) method
        family = op.sediment_family if sediment else op.tracer_family
        if family == 'dg':
            sipg = None
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
                    alpha = Constant(5.0*p*(p+1) if p != 0 else 1.5)
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
        eq_options[i].lax_friedrichs_tracer_scaling_factor = None
        eq_options[i].su_stabilisation = None
        eq_options[i].supg_stabilisation = None
        if stabilisation is None:
            return
        elif stabilisation == 'lax_friedrichs':
            assert hasattr(op, 'lax_friedrichs_tracer_scaling_factor')
            assert family == 'dg'
            eq_options[i]['lax_friedrichs_tracer_scaling_factor'] = op.lax_friedrichs_tracer_scaling_factor  # TODO: Allow mesh dependent
        elif stabilisation in ('su', 'supg'):
            assert family == 'cg'
            assert op.characteristic_speed is not None
            cell_size_measure = anisotropic_cell_size if op.anisotropic_stabilisation else CellSize
            h = cell_size_measure(self.meshes[i])
            U = op.characteristic_speed
            D = op.characteristic_diffusion
            tau = 0.5*h/U
            if D is not None:
                Pe = 0.5*h*U/D
                tau *= min_value(1, Pe/3)
            if stabilisation == 'su':
                eq_options[i].su_stabilisation = tau
            else:
                eq_options[i].supg_stabilisation = tau
        else:
            msg = "Stabilisation method {:s} not recognised for {:s}"
            raise ValueError(msg.format(stabilisation, self.__class__.__name__))

    # --- Solution initialisation and transfer

    def set_initial_condition(self, **kwargs):
        """
        Apply initial condition(s) for forward solution(s) on first mesh.
        """
        self.op.set_initial_condition(self, **kwargs)
        if self.op.solve_tracer:
            self.op.set_initial_condition_tracer(self)
        if self.op.solve_sediment:
            self.op.set_initial_condition_sediment(self)
        if self.op.solve_exner:
            self.op.set_initial_condition_bathymetry(self)

    def compute_mesh_reynolds_number(self, i):
        # u, eta = self.fwd_solutions[i].split()
        u = self.op.characteristic_velocity or self.fwd_solutions[i].split()[0]
        nu = self.fields[i].horizontal_viscosity
        if nu is None:
            return
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
        """
        Apply terminal condition(s) for adjoint solution(s) on terminal mesh.
        """
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
        op = self.op
        if op.solve_swe:
            self.project(self.fwd_solutions, i, j)
        else:
            op.set_initial_condition(self, **kwargs)

        # Project between spaces, constructing if necessary
        for flg, name in zip(op.solve_flags[1:], op.solve_fields[1:]):
            space = self.get_function_space(name)
            if flg:
                f = self.__getattribute__('fwd_solutions_{:s}'.format(name))
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
        op = self.op
        if op.solve_swe:
            self.project(self.adj_solutions, i, j)
        else:
            op.set_terminal_condition(self, **kwargs)

        # Project between spaces, constructing if necessary
        for flg, name in zip(op.solve_flags[1:], op.solve_fields[1:]):
            space = self.get_function_space(name)
            if flg:
                f = self.__getattribute__('adj_solutions_{:s}'.format(name))
                if f[i] is None:
                    raise ValueError("Nothing to project.")
                elif f[j] is None:
                    f[j] = Function(space[j], name="Adjoint {:s} solution".format(name))
                self.project(f, i, j)

    def project_to_intermediary_mesh(self, i):
        if self.op.solve_swe:
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

    def project_from_intermediary_mesh(self, i):
        if self.op.solve_swe:
            super(AdaptiveProblem, self).project_from_intermediary_mesh(i)
        if self.op.solve_tracer:
            self.fwd_solutions_tracer[i].project(self.intermediary_solutions_tracer[i])
        if self.op.solve_sediment:
            self.fwd_solutions_sediment[i].project(self.intermediary_solutions_sediment[i])
        if self.op.solve_exner:
            self.fwd_solutions_bathymetry[i].project(self.intermediary_solutions_bathymetry[i])
        if hasattr(self.op, 'sediment_model'):
            raise NotImplementedError

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
        from ..swe.equation import ShallowWaterEquations

        if self.mesh_velocities[i] is not None:
            self.shallow_water_options[i]['mesh_velocity'] = self.mesh_velocities[i]
        self.equations[i].shallow_water = ShallowWaterEquations(
            self.V[i],
            self.depth[i],
            self.shallow_water_options[i],
        )
        self.equations[i].shallow_water.bnd_functions = self.boundary_conditions[i]['shallow_water']

    def create_forward_tracer_equation_step(self, i):
        from ..tracer.equation import TracerEquation2D, ConservativeTracerEquation2D

        op = self.tracer_options[i]
        conservative = op.use_tracer_conservative_form
        model = ConservativeTracerEquation2D if conservative else TracerEquation2D
        self.equations[i].tracer = model(
            self.Q[i],
            self.depth[i],
            stabilisation=self.stabilisation_tracer,
            anisotropic=op.anisotropic_stabilisation,
            sipg_parameter=op.sipg_parameter,
            su_stabilisation=op.su_stabilisation,
            supg_stabilisation=op.supg_stabilisation,
        )
        if op.use_limiter_for_tracers and self.Q[i].ufl_element().degree() > 0:
            self.tracer_limiters[i] = VertexBasedP1DGLimiter(self.Q[i])
        self.equations[i].tracer.bnd_functions = self.boundary_conditions[i]['tracer']

    def create_forward_sediment_equation_step(self, i):
        from ..sediment.equation import SedimentEquation2D

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
        from ..sediment.exner_eq import ExnerEquation

        model = ExnerEquation
        self.equations[i].exner = model(
            self.W[i],
            self.depth[i],
            conservative=self.op.use_tracer_conservative_form,
            sed_model=self.op.sediment_model,
        )

    def free_forward_equations_step(self, i):
        if self.op.solve_swe:
            delattr(self.equations[i], 'shallow_water')
        if self.op.solve_tracer:
            delattr(self.equations[i], 'tracer')
        if self.op.solve_sediment:
            delattr(self.equations[i], 'sediment')
        if self.op.solve_exner:
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
        from ..swe.adjoint import AdjointShallowWaterEquations

        self.equations[i].adjoint_shallow_water = AdjointShallowWaterEquations(
            self.V[i],
            self.depth[i],
            self.shallow_water_options[i],
        )
        self.equations[i].adjoint_shallow_water.bnd_functions = self.boundary_conditions[i]['shallow_water']

    def create_adjoint_tracer_equation_step(self, i):
        from ..tracer.equation import TracerEquation2D, ConservativeTracerEquation2D

        op = self.tracer_options[i]
        conservative = op.use_tracer_conservative_form
        model = TracerEquation2D if conservative else ConservativeTracerEquation2D
        self.equations[i].adjoint_tracer = model(
            self.Q[i],
            self.depth[i],
            stabilisation=self.stabilisation_tracer,
            anisotropic=op.anisotropic_stabilisation,
            sipg_parameter=op.sipg_parameter,
            su_stabilisation=op.su_stabilisation,
            supg_stabilisation=op.supg_stabilisation,
        )
        if op.use_limiter_for_tracers and self.Q[i].ufl_element().degree() > 0:
            self.tracer_limiters[i] = VertexBasedP1DGLimiter(self.Q[i])
        adjoint_boundary_conditions = {}
        zero = Constant(0.0)
        for segment in self.boundary_conditions[i]['tracer']:
            adjoint_boundary_conditions[segment] = {}
            if 'diff_flux' not in self.boundary_conditions[i]['tracer'][segment]:
                adjoint_boundary_conditions[segment]['value'] = zero
            if 'value' not in self.boundary_conditions[i]['tracer'][segment]:
                adjoint_boundary_conditions[segment]['diff_flux'] = 'adjoint'
        self.equations[i].adjoint_tracer.bnd_functions = adjoint_boundary_conditions

    def create_adjoint_sediment_equation_step(self, i):
        raise NotImplementedError("Continuous adjoint sediment equation not implemented")

    def create_adjoint_exner_equation_step(self, i):
        raise NotImplementedError("Continuous adjoint Exner equation not implemented")

    def free_adjoint_equations_step(self, i):
        if self.op.solve_swe:
            delattr(self.equations[i], 'adjoint_shallow_water')
        if self.op.solve_tracer:
            delattr(self.equations[i], 'adjoint_tracer')
        if self.op.solve_sediment:
            delattr(self.equations[i], 'adjoint_sediment')
        if self.op.solve_exner:
            delattr(self.equations[i], 'adjoint_exner')

    def get_boundary_conditions(self, i):
        field = self.op.adapt_field
        if field not in ('tracer', 'sediment', 'bathymetry'):
            field = 'shallow_water'
        return self.boundary_conditions[i][field]

    # --- Error estimators

    def create_error_estimators_step(self, i, adjoint=False):
        if adjoint:
            self.create_adjoint_error_estimators_step(i)
        else:
            self.create_forward_error_estimators_step(i)

    def create_forward_error_estimators_step(self, i):
        if self.op.solve_swe:
            self.create_forward_shallow_water_error_estimator_step(i)
        if self.op.solve_tracer:
            self.create_forward_tracer_error_estimator_step(i)
        if self.op.solve_sediment:
            self.create_forward_sediment_error_estimator_step(i)
        if self.op.solve_exner:
            self.create_forward_exner_error_estimator_step(i)

    def create_adjoint_error_estimators_step(self, i):
        if self.op.solve_swe:
            self.create_adjoint_shallow_water_error_estimator_step(i)
        if self.op.solve_tracer:
            self.create_adjoint_tracer_error_estimator_step(i)
        if self.op.solve_sediment:
            self.create_adjoint_sediment_error_estimator_step(i)
        if self.op.solve_exner:
            self.create_adjoint_exner_error_estimator_step(i)

    def create_forward_shallow_water_error_estimator_step(self, i):
        from ..swe.error_estimation import ShallowWaterGOErrorEstimator

        self.error_estimators[i].shallow_water = ShallowWaterGOErrorEstimator(
            self.V[i],
            self.depth[i],
            self.shallow_water_options[i],
        )

    def create_forward_tracer_error_estimator_step(self, i):
        from ..tracer.error_estimation import TracerGOErrorEstimator

        op = self.tracer_options[i]
        self.error_estimators[i].tracer = TracerGOErrorEstimator(
            self.Q[i],
            self.depth[i],
            stabilisation=self.stabilisation_tracer,
            anisotropic=self.op.anisotropic_stabilisation,
            sipg_parameter=op.sipg_parameter,
            su_stabilisation=op.su_stabilisation,
            supg_stabilisation=op.supg_stabilisation,
            conservative=op.use_tracer_conservative_form,
            adjoint=False,
        )

    def create_forward_sediment_error_estimator_step(self, i):
        raise NotImplementedError("Error estimators for sediment not implemented.")

    def create_forward_exner_error_estimator_step(self, i):
        raise NotImplementedError("Error estimators for Exner not implemented.")

    def create_adjoint_shallow_water_error_estimator_step(self, i):
        raise NotImplementedError("Error estimators for adjoint shallow water not implemented.")

    def create_adjoint_tracer_error_estimator_step(self, i):
        from ..tracer.error_estimation import TracerGOErrorEstimator

        op = self.tracer_options[i]
        self.error_estimators[i].adjoint_tracer = TracerGOErrorEstimator(
            self.Q[i],
            self.depth[i],
            stabilisation=self.stabilisation_tracer,
            anisotropic=self.op.anisotropic_stabilisation,
            sipg_parameter=op.sipg_parameter,
            su_stabilisation=op.su_stabilisation,
            supg_stabilisation=op.supg_stabilisation,
            conservative=not op.use_tracer_conservative_form,
            adjoint=True,
        )

    def create_adjoint_sediment_error_estimator_step(self, i):
        raise NotImplementedError("Error estimators for adjoint sediment not implemented.")

    def create_adjoint_exner_error_estimator_step(self, i):
        raise NotImplementedError("Error estimators for adjoint Exner not implemented.")

    # --- Timestepping

    def create_forward_timesteppers_step(self, i, restarted=False):
        if i == 0 and not restarted:
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
        if self.op.approach in ('lagrangian', 'hybrid'):
            raise NotImplementedError  # TODO
        if self.stabilisation == 'lax_friedrichs':
            fields['lax_friedrichs_velocity_scaling_factor'] = self.shallow_water_options[i].lax_friedrichs_velocity_scaling_factor
        return fields

    def _get_fields_for_tracer_timestepper(self, i):
        if self.op.timestepper == 'SteadyState':
            if self.op.solve_swe:
                u, eta = split(self.fwd_solutions[i])  # FIXME: Not fully annotated
            else:
                u = interpolate(as_vector(self.op.base_velocity), self.P1_vec[i])
                eta = Constant(0.0)
        else:
            u, eta = self.fwd_solutions[i].split()  # FIXME: Not fully annotated
        fields = AttrDict({
            'elev_{:d}d'.format(self.dim): eta,
            'uv_{:d}d'.format(self.dim): u,
            'diffusivity_h': self.fields[i].horizontal_diffusivity,
            'source': self.fields[i].tracer_source_2d,
            'tracer_advective_velocity_factor': self.fields[i].tracer_advective_velocity_factor,
            'lax_friedrichs_tracer_scaling_factor': self.tracer_options[i].lax_friedrichs_tracer_scaling_factor,
            'mesh_velocity': None,
        })
        if self.mesh_velocities[i] is not None:
            fields['mesh_velocity'] = self.mesh_velocities[i]
        if self.op.approach in ('lagrangian', 'hybrid'):
            self.mesh_velocities[i] = u
            fields['uv_{:d}d'.format(self.dim)] = Constant(as_vector(np.zeros(self.dim)))
        if self.stabilisation_tracer == 'lax_friedrichs':
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
        if self.op.approach in ('lagrangian', 'hybrid'):
            self.mesh_velocities[i] = u
            fields['uv_2d'] = Constant(as_vector([0.0, 0.0]))
        if self.stabilisation_sediment == 'lax_friedrichs':
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
        if self.op.approach in ('lagrangian', 'hybrid'):
            self.mesh_velocities[i] = u
            fields['uv_2d'] = Constant(as_vector([0.0, 0.0]))
        return fields

    def create_forward_shallow_water_timestepper_step(self, i, integrator):
        fields = self._get_fields_for_shallow_water_timestepper(i)
        bcs = self.boundary_conditions[i]['shallow_water']
        args = (self.equations[i].shallow_water, self.fwd_solutions[i], fields, self.op.dt, )
        kwargs = {
            'bnd_conditions': bcs,
            'solver_parameters': self.op.solver_parameters['shallow_water'],
        }
        if self.op.timestepper == 'CrankNicolson':
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
            delattr(self.timesteppers[i], 'shallow_water')
        if self.op.solve_tracer:
            delattr(self.timesteppers[i], 'tracer')
        if self.op.solve_sediment:
            delattr(self.timesteppers[i], 'sediment')
        if self.op.solve_exner:
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
        uv_str = 'uv_{:d}d'.format(self.dim)
        elev_str = 'elev_{:d}d'.format(self.dim)
        fields[uv_str], fields[elev_str] = self.fwd_solutions[i].split()

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
        if self.op.timestepper == 'CrankNicolson':
            kwargs['semi_implicit'] = self.op.use_semi_implicit_linearisation
            kwargs['theta'] = self.op.implicitness_theta
        if 'adjoint_shallow_water' in self.error_estimators[i]:
            kwargs['error_estimator'] = self.error_estimators[i].adjoint_shallow_water
        self.timesteppers[i].adjoint_shallow_water = integrator(*args, **kwargs)

    def create_adjoint_tracer_timestepper_step(self, i, integrator):
        fields = self._get_fields_for_tracer_timestepper(i)

        # Account for dJdc
        self.kernels_tracer[i] = self.op.set_qoi_kernel_tracer(self, i)
        self.time_kernel = Constant(1.0 if self.simulation_time >= self.op.start_time else 0.0)
        fields['source'] = self.time_kernel*self.kernels_tracer[i]

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
        if 'adjoint_tracer' in self.error_estimators[i]:
            kwargs['error_estimator'] = self.error_estimators[i].adjoint_tracer
        self.timesteppers[i].adjoint_tracer = integrator(*args, **kwargs)

    def create_adjoint_sediment_timestepper_step(self, i, integrator):
        raise NotImplementedError("Continuous adjoint sediment timestepping not implemented")

    def create_adjoint_exner_timestepper_step(self, i, integrator):
        raise NotImplementedError("Continuous adjoint Exner timestepping not implemented")

    def free_adjoint_timesteppers_step(self, i):
        if self.op.solve_swe:
            delattr(self.timesteppers[i], 'adjoint_shallow_water')
        if self.op.solve_tracer:
            delattr(self.timesteppers[i], 'adjoint_tracer')
        if self.op.solve_sediment:
            delattr(self.timesteppers[i], 'adjoint_sediment')
        if self.op.solve_exner:
            delattr(self.timesteppers[i], 'adjoint_exner')

    def get_timestepper(self, i, field, adjoint=False):
        if field not in ('tracer', 'sediment', 'bathymetry'):
            field = 'shallow_water'
        if adjoint:
            field = '_'.join(['adjoint', field])
        return self.timesteppers[i][field]

    # --- Solvers

    def add_callbacks(self, i, **kwargs):
        from thetis.callback import CallbackManager

        # Create a new CallbackManager object on every mesh
        #   NOTE: This overwrites any pre-existing CallbackManagers
        self.op.print_debug("SETUP: Creating CallbackManagers...")
        self.callbacks[i] = CallbackManager()

        # Get label
        mode = 'export'
        adjoint = kwargs.get('adjoint', False)
        if adjoint:
            mode += '_adjoint'

        # Add default callbacks
        if self.op.solve_swe:
            self.callbacks[i].add(VelocityNormCallback(self, i, **kwargs), mode)
            self.callbacks[i].add(ElevationNormCallback(self, i, **kwargs), mode)
        if self.op.solve_tracer:
            self.callbacks[i].add(TracerNormCallback(self, i, **kwargs), mode)
        if self.op.solve_sediment:
            self.callbacks[i].add(SedimentNormCallback(self, i, **kwargs), mode)
        if self.op.solve_exner:
            self.callbacks[i].add(ExnerNormCallback(self, i, **kwargs), mode)
        if self.op.recover_vorticity and not adjoint:
            if not hasattr(self, 'vorticity'):
                self.vorticity = [None for mesh in self.meshes]
            self.callbacks[i].add(VorticityNormCallback(self, i, **kwargs), mode)

    def setup_solver_forward_step(self, i, restarted=False):
        """
        Setup forward solver on mesh `i`.
        """
        op = self.op
        op.print_debug("SETUP: Creating forward equations on mesh {:d}...".format(i))
        self.create_forward_equations_step(i)
        op.print_debug("SETUP: Creating forward timesteppers on mesh {:d}...".format(i))
        self.create_forward_timesteppers_step(i, restarted=restarted)
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
        self.add_callbacks(i, adjoint=False)

    def solve_forward_step(self, i, update_forcings=None, export_func=None, plot_pvd=True, export_initial=False, restarted=False):
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
        if not restarted:
            self.iteration = 0
        start_time = i*op.dt*self.dt_per_mesh
        end_time = (i+1)*op.dt*self.dt_per_mesh
        if not restarted:
            try:
                assert np.allclose(self.simulation_time, start_time)
            except AssertionError:
                msg = "Mismatching start time: {:.2f} vs {:.2f}"
                raise ValueError(msg.format(self.simulation_time, start_time))

        # Exports and callbacks
        self.print(80*'=')
        update_forcings = update_forcings or self.op.get_update_forcings(self, i, adjoint=False)
        export_func = export_func or self.op.get_export_func(self, i)
        if export_initial:
            update_forcings(self.simulation_time)  # TODO: CHECK
            if export_func is not None:
                export_func()
            self.callbacks[i].evaluate(mode='export')
            self.callbacks[i].evaluate(mode='timestep')

        # Print time to screen
        op.print_debug("SOLVE: Entering forward timeloop on mesh {:d}...".format(i))
        if self.num_meshes == 1:
            msg = "FORWARD SOLVE  time {:8.2f}  ({:6.2f}) seconds"
            self.print(msg.format(self.simulation_time, 0.0))
        else:
            msg = "{:2d} FORWARD SOLVE mesh {:2d}/{:2d}  time {:8.2f}  ({:6.2f}) seconds"
            self.print(msg.format(self.outer_iteration, i+1, self.num_meshes,
                                  self.simulation_time, 0.0))
        cpu_timestamp = perf_counter()

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
                inverted = self.move_mesh(i)
                if inverted and op.approach in ('lagrangian', 'hybrid'):
                    self.simulation_time += op.dt
                    self.iteration += 1
                    self.add_callbacks(i)  # TODO: Only normed ones will work
                    self.setup_solver_forward_step(i, restarted=True)
                    self.solve_forward_step(i, update_forcings=update_forcings, export_func=export_func, plot_pvd=plot_pvd, export_initial=True, restarted=True)
                    return

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

            self.iteration += 1
            self.simulation_time += op.dt
            self.callbacks[i].evaluate(mode='timestep')
            if self.iteration % op.dt_per_export == 0:

                # Exports and callbacks
                if export_func is not None:
                    export_func()
                self.callbacks[i].evaluate(mode='export')

                # Print time to screen
                cpu_time = perf_counter() - cpu_timestamp
                if self.num_meshes == 1:
                    self.print(msg.format(self.simulation_time, cpu_time))
                else:
                    self.print(msg.format(self.outer_iteration, i+1, self.num_meshes,
                                          self.simulation_time, cpu_time))
                cpu_timestamp = perf_counter()

                # Plot to .pvd
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
        if update_forcings is not None:
            update_forcings(self.simulation_time + op.dt)
        self.print(80*'=')

    def setup_solver_adjoint_step(self, i):
        """
        Setup forward solver on mesh `i`.
        """
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
        self.add_callbacks(i, adjoint=True)

    def solve_adjoint_step(self, i, update_forcings=None, export_func=None, plot_pvd=True, export_initial=False, **kwargs):
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

        # Exports and callbacks
        self.print(80*'=')
        update_forcings = update_forcings or self.op.get_update_forcings(self, i, adjoint=True)
        export_func = export_func or self.op.get_export_func(self, i)
        if export_initial:
            update_forcings(self.simulation_time)
            export_func()
        self.callbacks[i].evaluate(mode='export_adjoint')

        # Print time to screen
        op.print_debug("SOLVE: Entering forward timeloop on mesh {:d}...".format(i))
        if self.num_meshes == 1:
            msg = "ADJOINT SOLVE time {:8.2f}  ({:6.2f} seconds)"
            self.print(msg.format(self.simulation_time, 0.0))
        else:
            msg = "{:2d}  ADJOINT SOLVE mesh {:2d}/{:2d}  time {:8.2f}  ({:6.2f} seconds)"
            self.print(msg.format(self.outer_iteration, i+1, self.num_meshes,
                                  self.simulation_time, 0.0))
        cpu_timestamp = perf_counter()

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

            self.iteration -= 1
            self.simulation_time -= op.dt
            if self.iteration % op.dt_per_export == 0:

                # Exports and callbacks
                if export_func is not None:
                    export_func()
                self.callbacks[i].evaluate(mode='export_adjoint')

                # Print time to screen
                cpu_time = perf_counter() - cpu_timestamp
                if self.num_meshes == 1:
                    self.print(msg.format(self.simulation_time, cpu_time))
                else:
                    self.print(msg.format(self.outer_iteration, i+1, self.num_meshes,
                                          self.simulation_time, cpu_time))
                cpu_timestamp = perf_counter()

                # Plot to .pvd
                if op.solve_swe and plot_pvd:
                    z, zeta = self.adj_solutions[i].split()
                    proj_z.project(z)
                    proj_zeta.project(zeta)
                    self.adjoint_solution_file.write(proj_z, proj_zeta)
                if op.solve_tracer and plot_pvd:
                    proj_tracer.project(self.adj_solutions_tracer[i])
                    self.adjoint_tracer_file.write(proj_tracer)
            self.time_kernel.assign(1.0 if self.simulation_time >= self.op.start_time else 0.0)
        if update_forcings is not None:
            update_forcings(self.simulation_time - op.dt)
        self.print(80*'=')

    # --- Metric

    def recover_hessian_metrics(self, i, adjoint=False, **kwargs):  # TODO: USEME more
        op = self.op
        kwargs.setdefault('normalise', True)
        kwargs['op'] = op
        sol = self.get_solutions(op.adapt_field, adjoint=adjoint)[i]
        if op.adapt_field in ('tracer', 'sediment', 'bathymetry'):

            # Account for SUPG stabilisation in weighted Hessian metric
            if op.adapt_field in ('tracer', 'sediment') and op.stabilisation_tracer == 'supg':
                if self.approach == 'weighted_hessian':
                    eq_options = self.__getattribute__('{:s}_options'.format(op.adapt_field))
                    u = op.get_velocity(self.simulation_time)
                    sol = sol + eq_options[i].supg_stabilisation*dot(u, grad(sol))

            return [steady_metric(sol, mesh=self.meshes[i], **kwargs)]
        else:
            fields = {'bathymetry': self.bathymetry[i], 'inflow': self.inflow[i]}
            return [
                recover_hessian_metric(sol, adapt_field='velocity_x', fields=fields, **kwargs),
                recover_hessian_metric(sol, adapt_field='velocity_y', fields=fields, **kwargs),
                recover_hessian_metric(sol, adapt_field='elevation', fields=fields, **kwargs),
            ]

    def get_static_hessian_metric(self, adapt_field, i=0, adjoint=False, elementwise=False):
        """
        Compute an appropriate Hessian for the problem at hand. This is inherently
        problem-dependent, since the choice of field for adaptation is not universal.
        """
        hessian_kwargs = dict(normalise=False, enforce_constraints=False, op=self.op)
        if elementwise:
            sol = self.get_solutions(adapt_field, adjoint=adjoint)[i]
            hessian_kwargs['V'] = self.P0_ten[i]
            gradient_kwargs = dict(mesh=self.mesh, op=self.op)
            if adapt_field == 'shallow_water':
                u, eta = sol.split()
                fields = [u[0], u[1], eta]
                gradients = [recover_gradient(f, **gradient_kwargs) for f in fields]
                hessians = [steady_metric(H=grad(g), **hessian_kwargs) for g in gradients]
                return combine_metrics(*hessians, average='avg' in self.op.adapt_field)
            else:
                return steady_metric(H=grad(recover_gradient(sol, op=self.op)), **hessian_kwargs)
        else:
            hessians = self.recover_hessian_metrics(0, adjoint=adjoint, **hessian_kwargs)
            if self.op.adapt_field in ('tracer', 'sediment', 'bathymetry'):
                return hessians[0]
            else:
                return combine_metrics(*hessians, average='avg' in self.op.adapt_field)

    def get_recovery(self, i, **kwargs):
        """
        Create an :class:`L2Projector` object which can repeatedly project fields specified in
        :attr:`op.adapt_field`.
        """
        op = self.op
        if op.adapt_field in ('tracer', 'sediment', 'bathymetry'):
            fs = self.get_function_space(op.adapt_field)[i]
            return HessianMetricRecoverer(fs, op=op)
        elif op.approach == 'vorticity':  # TODO: Use recoverer stashed in callback
            return L2ProjectorVorticity(self.V[i], op=op)
        else:
            return ShallowWaterHessianRecoverer(
                self.V[i], op=op,
                constant_fields={'bathymetry': self.bathymetry[i]}, **kwargs,
            )

    # --- Run scripts

    def run(self, **kwargs):
        """
        Run simulation using mesh adaptation approach specified by `self.approach`.

        For metric-based approaches, a fixed point iteration loop is used.
        """
        run_scripts = {

            # Non-adaptive
            'fixed_mesh': self.solve_forward,

            # Metric-based using forward solution fields
            'hessian': self.run_hessian_based,
            'vorticity': self.run_hessian_based,  # TODO: Change name and update docs

            # Metric-based using forward *and* adjoint solution fields
            'dwp': self.run_dwp,

            # Metric-based goal-oriented using DWR
            'dwr': self.run_dwr,
            'dwr_adjoint': self.run_dwr,
            'dwr_avg': self.run_dwr,
            'dwr_int': self.run_dwr,
            'isotropic_dwr': self.run_dwr,                 # TODO: Unsteady case
            'isotropic_dwr_adjoint': self.run_dwr,         # TODO: Unsteady case
            'isotropic_dwr_avg': self.run_dwr,             # TODO: Unsteady case
            'isotropic_dwr_int': self.run_dwr,             # TODO: Unsteady case
            'anisotropic_dwr': self.run_dwr,               # TODO: Unsteady case
            'anisotropic_dwr_adjoint': self.run_dwr,       # TODO: Unsteady case
            'anisotropic_dwr_avg': self.run_dwr,           # TODO: Unsteady case
            'anisotropic_dwr_int': self.run_dwr,           # TODO: Unsteady case

            # Metric-based goal-oriented *not* using DWR
            'weighted_hessian': self.run_no_dwr,
            'weighted_hessian_adjoint': self.run_no_dwr,   # TODO: Unsteady case
            'weighted_hessian_avg': self.run_no_dwr,       # TODO: Unsteady case
            'weighted_hessian_int': self.run_no_dwr,       # TODO: Unsteady case
            'weighted_gradient': self.run_no_dwr,          # TODO: Unsteady case
            'weighted_gradient_adjoint': self.run_no_dwr,  # TODO: Unsteady case
            'weighted_gradient_avg': self.run_no_dwr,      # TODO: Unsteady case
            'weighted_gradient_int': self.run_no_dwr,      # TODO: Unsteady case
        }
        if self.approach not in run_scripts:
            raise ValueError("Approach '{:s}' not recognised".format(self.approach))
        run_scripts[self.approach](**kwargs)

    def run_dwp(self, **kwargs):  # TODO: Modify indicator for time interval
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

        Convergence criteria:
          * Convergence of quantity of interest (relative tolerance `op.qoi_rtol`);
          * Convergence of mesh element count (relative tolerance `op.element_rtol`);
          * Maximum number of iterations reached (`op.max_adapt`).

        [1] B. Davis & R. LeVeque, "Adjoint Methods for Guiding Adaptive Mesh Refinement in
            Tsunami Modelling", Pure and Applied Geophysics, 173, Springer International
            Publishing (2016), p.4055--4074, DOI 10.1007/s00024-016-1412-y.
        """
        op = self.op
        wq = Constant(1.0)  # Quadrature weight

        # Loop until we hit the maximum number of iterations, max_adapt
        assert op.min_adapt < op.max_adapt
        for n in range(op.max_adapt):
            self.outer_iteration = n

            # Solve forward to get checkpoints
            self.solve_forward()

            # Check convergence
            if (self.qoi_converged or self.maximum_adaptations_met) and self.minimum_adaptations_met:
                break

            # Loop over mesh windows *in reverse*
            self.metrics = [Function(P1_ten, name="Metric") for P1_ten in self.P1_ten]
            for i in reversed(range(self.num_meshes)):
                fwd_solutions_step = []
                adj_solutions_step = []

                # --- Solve forward on current window

                def export_func():
                    fwd_solutions_step.append(self.fwd_solutions[i].copy(deepcopy=True))

                self.simulation_time = i*op.dt*self.dt_per_mesh
                self.transfer_forward_solution(i)
                self.setup_solver_forward_step(i)
                self.solve_forward_step(i, export_func=export_func, plot_pvd=False, export_initial=True)

                # --- Solve adjoint on current window

                def export_func():
                    adj_solutions_step.append(self.adj_solutions[i].copy(deepcopy=True))

                self.transfer_adjoint_solution(i)
                self.setup_solver_adjoint_step(i)
                self.solve_adjoint_step(i, export_func=export_func, plot_pvd=False)

                # Assemble indicator
                n_fwd = len(fwd_solutions_step)
                n_adj = len(adj_solutions_step)
                if n_fwd != n_adj:
                    msg = "Mismatching number of indicators ({:d} vs {:d})"
                    raise ValueError(msg.format(n_fwd, n_adj))
                self.indicators[i]['dwp'] = Function(self.P1[i], name="DWP indicator")
                op.print_debug("DWP indicators on mesh {:2d}".format(i))
                for j, solutions in enumerate(zip(fwd_solutions_step, reversed(adj_solutions_step))):
                    if op.timestepper == 'CrankNicolson':
                        w = 0.5 if j in (0, n_fwd-1) else 1.0  # Trapezium rule
                    else:
                        raise NotImplementedError  # TODO: Other integrators
                    wq.assign(w*op.dt*self.dt_per_mesh)
                    self.indicators[i]['dwp'] += interpolate(wq*abs(inner(*solutions)), self.P1[i])

                # Construct isotropic metric
                self.metrics[i].assign(isotropic_metric(self.indicators[i]['dwp'], normalise=False))

            # Normalise metrics
            self.space_time_normalise()

            # Adapt meshes
            self.adapt_meshes()

            # Check convergence
            if not self.minimum_adaptations_met:
                continue
            if self.elements_converged:
                break

    def run_dwr(self, **kwargs):
        """
        Main script for goal-oriented mesh adaptation routines.

        Both isotropic and anisotropic metrics are considered, as described in
        [Wallwork et al. 2021].

        Convergence criteria:
          * Convergence of quantity of interest (relative tolerance `op.qoi_rtol`);
          * Convergence of mesh element count (relative tolerance `op.element_rtol`);
          * Convergence of error estimator (relative tolerance `op.estimator_rtol`);
          * Maximum number of iterations reached (`op.max_adapt`).

        [Wallwork et al. 2021] J. G. Wallwork, N. Barral, D. A. Ham, M. D. Piggott, "Goal-Oriented
            Error Estimation and Mesh Adaptation for Tracer Transport Problems", to be submitted to
            Computer Aided Design.
        """
        op = self.op
        wq = Constant(1.0)  # Quadrature weight
        assert self.approach in ('dwr', 'anisotropic_dwr')
        if op.dt_per_export > 1:
            raise NotImplementedError  # TODO

        # Loop until we hit the maximum number of iterations, max_adapt
        assert op.min_adapt < op.max_adapt
        self.estimators['dwr'] = []
        for n in range(op.max_adapt):
            self.outer_iteration = n

            # Solve forward to get checkpoints
            self.solve_forward()

            # Check convergence
            if (self.qoi_converged or self.maximum_adaptations_met) and self.minimum_adaptations_met:
                break

            # Setup problem on enriched space
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
                hierarchy = [MeshHierarchy(mesh, 1) for mesh in self.meshes]
                refined_meshes = [h[1] for h in hierarchy]
            ep = type(self)(
                op,
                meshes=refined_meshes,
                nonlinear=self.nonlinear,
            )
            ep.outer_iteration = n
            enriched_space = ep.get_function_space(op.adapt_field)

            # Loop over mesh windows *in reverse*
            for i, P1 in enumerate(self.P1):
                self.indicators[i]['dwr'] = Function(P1, name="DWR indicator")
            self.metrics = [Function(P1_ten, name="Metric") for P1_ten in self.P1_ten]
            self.estimators['dwr'].append(0.0)
            for i in reversed(range(self.num_meshes)):
                fwd_solutions_step = []
                fwd_solutions_step_old = []
                adj_solutions_step = []
                enriched_adj_solutions_step = []
                tm = dmhooks.get_transfer_manager(self.get_plex(i))

                # --- Setup forward solver for enriched problem

                # TODO: Need to transfer fwd sol in nonlinear case
                ep.create_error_estimators_step(i)  # Passed to the timesteppers under the hood
                ep.setup_solver_forward_step(i)
                ets = ep.get_timestepper(i, op.adapt_field)

                # --- Solve forward on current window

                ts = self.get_timestepper(i, op.adapt_field)

                def export_func():
                    fwd_solutions_step.append(ts.solution.copy(deepcopy=True))
                    fwd_solutions_step_old.append(ts.solution_old.copy(deepcopy=True))
                    # TODO: Also need store fields at each export (in general case)

                self.simulation_time = i*op.dt*self.dt_per_mesh
                self.transfer_forward_solution(i)
                self.setup_solver_forward_step(i)
                self.solve_forward_step(i, export_func=export_func, plot_pvd=False, export_initial=False)

                # --- Solve adjoint on current window

                def export_func():
                    adj = self.get_solutions(op.adapt_field, adjoint=True)[i].copy(deepcopy=True)
                    adj_solutions_step.append(adj)

                self.transfer_adjoint_solution(i)
                self.setup_solver_adjoint_step(i)
                self.solve_adjoint_step(i, export_func=export_func, plot_pvd=False, export_initial=True)

                # --- Solve adjoint on current window in enriched space

                def export_func():
                    adj = ep.get_solutions(op.adapt_field, adjoint=True)[i].copy(deepcopy=True)
                    enriched_adj_solutions_step.append(adj)

                ep.simulation_time = (i+1)*op.dt*self.dt_per_mesh  # TODO: Shouldn't be needed
                ep.transfer_adjoint_solution(i)
                ep.setup_solver_adjoint_step(i)
                ep.solve_adjoint_step(i, export_func=export_func, plot_pvd=False, export_initial=True)

                # --- Assemble indicators and metrics

                # Reverse adjoint solution arrays and take pairwise averages
                adj_solutions_step = list(reversed(adj_solutions_step))
                enriched_adj_solutions_step = list(reversed(enriched_adj_solutions_step))
                for j in range(len(adj_solutions_step)):
                    adj_solutions_step[j] *= 0.5
                    enriched_adj_solutions_step[j] *= 0.5
                for j in range(len(adj_solutions_step)-1):
                    for adj1, adj2 in zip(adj_solutions_step[j].split(), adj_solutions_step[j+1].split()):
                        adj1 += adj2
                    for adj1, adj2 in zip(enriched_adj_solutions_step[j].split(), enriched_adj_solutions_step[j+1].split()):
                        adj1 += adj2
                adj_solutions_step = adj_solutions_step[:-1]
                enriched_adj_solutions_step = enriched_adj_solutions_step[:-1]

                # Checks
                n_fwd = len(fwd_solutions_step)
                n_adj = len(adj_solutions_step)
                if n_fwd != n_adj:
                    msg = "Mismatching number of indicators ({:d} vs {:d})"
                    raise ValueError(msg.format(n_fwd, n_adj))
                op.print_debug("GO: Computing DWR indicators on mesh {:2d}".format(i))

                # Various work fields
                indicator_enriched = Function(ep.P0[i])
                indicator_enriched_cts = Function(ep.P1[i])
                tmp = Function(ep.P1[i])
                fwd_proj = Function(enriched_space[i])
                fwd_old_proj = Function(enriched_space[i])
                adj_error = Function(enriched_space[i])

                # Setup error estimator
                bcs = self.get_boundary_conditions(i)
                ets.setup_error_estimator(fwd_proj, fwd_old_proj, adj_error, bcs)

                # Loop over exported timesteps
                for j in range(len(fwd_solutions_step)):
                    if op.timestepper == 'CrankNicolson':
                        w = 0.5 if j in (0, n_fwd-1) else 1.0  # Trapezium rule
                    else:
                        raise NotImplementedError  # TODO: Other integrators
                    wq.assign(w*op.dt*op.dt_per_export)

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
                    tmp.project(indicator_enriched)
                    indicator_enriched_cts += wq*tmp

                # Inject into the base space and construct an isotropic metric
                tm.inject(indicator_enriched_cts, self.indicators[i]['dwr'])
                self.indicators[i]['dwr'].interpolate(abs(self.indicators[i]['dwr']))  # Ensure +ve
                self.estimators['dwr'][-1] += self.indicators[i]['dwr'].vector().gather().sum()
                if self.approach == 'dwr':
                    self.metrics[i].assign(isotropic_metric(self.indicators[i]['dwr'], normalise=False))
                else:
                    raise NotImplementedError  # TODO: anisotropic_dwr

            del adj_error
            del indicator_enriched
            del ep
            del refined_meshes
            del hierarchy

            # Check convergence of error estimator
            if self.estimator_converged:
                break

            # Normalise metrics
            self.space_time_normalise()

            # Adapt meshes
            self.adapt_meshes()

            # Check convergence
            if not self.minimum_adaptations_met:
                continue
            if self.elements_converged:
                break

    def run_no_dwr(self, **kwargs):
        """
        Main script for goal-oriented mesh adaptation routines which do not use the dual-weighted
        residual.

        'Weighted Hessian' and 'weighted gradient' metrics are considered, as described in
        [Wallwork et al. 2021].

        Convergence criteria:
          * Convergence of quantity of interest (relative tolerance `op.qoi_rtol`);
          * Convergence of mesh element count (relative tolerance `op.element_rtol`);
          * Maximum number of iterations reached (`op.max_adapt`).

        [Wallwork et al. 2021] J. G. Wallwork, N. Barral, D. A. Ham, M. D. Piggott, "Goal-Oriented
            Error Estimation and Mesh Adaptation for Tracer Transport Problems", to be submitted to
            Computer Aided Design.
        """
        if self.approach == 'weighted_hessian':
            self._run_weighted_hessian()
        else:
            raise NotImplementedError  # TODO

    def _run_weighted_hessian(self, **kwargs):
        op = self.op
        N = self.export_per_mesh
        assert self.approach == 'weighted_hessian'
        w = op.dt*op.dt_per_export
        if op.hessian_time_combination == 'integrate':
            w *= N-1

        # Process parameters
        if op.adapt_field not in ('tracer', 'sediment', 'bathymetry'):
            if op.adapt_field not in ('all_avg', 'all_int'):
                op.adapt_field = 'all_{:s}'.format('int' if 'int' in op.adapt_field else 'avg')
        if op.adapt_field in ('all_avg', 'all_int'):
            c = op.adapt_field[-3:]
            op.adapt_field = "velocity_x__{:s}__velocity_y__{:s}__elevation".format(c, c)
        adapt_fields = ('__int__'.join(op.adapt_field.split('__avg__'))).split('__int__')
        if op.dt_per_export > 1 or op.hessian_timestep_lag > 1:
            raise NotImplementedError  # TODO

        # Loop until we hit the maximum number of iterations, max_adapt
        assert op.min_adapt < op.max_adapt
        hessian_kwargs = dict(normalise=False, enforce_constraints=False)
        for n in range(op.max_adapt):
            self.outer_iteration = n
            export_func_wrapper = None

            # Arrays to hold Hessians for each field on each window
            self._H_windows = [[[
                self.maximum_metric(i) for j in range(N)]
                for i in range(self.num_meshes)]
                for f in adapt_fields
            ]

            # Solve forward to get checkpoints
            for i in range(self.num_meshes):
                self.create_error_estimators_step(i)  # Passed to the timesteppers under the hood
                self.transfer_forward_solution(i)
                self.setup_solver_forward_step(i)
                self.solve_forward_step(i)

            # Check convergence
            if (self.qoi_converged or self.maximum_adaptations_met) and self.minimum_adaptations_met:
                break

            # --- Loop over mesh windows *in reverse*

            for i, P1 in enumerate(self.P1):
                self.indicators[i]['dwr'] = Function(P1, name="DWR indicator")
            self.metrics = [Function(P1_ten, name="Metric") for P1_ten in self.P1_ten]
            for i in reversed(range(self.num_meshes)):
                strong_residuals_step = []
                adj_solutions = self.get_solutions(op.adapt_field, adjoint=True)
                update_forcings = op.get_update_forcings(self, i, adjoint=False)
                export_func = op.get_export_func(self, i)

                # --- Solve forward on current window

                self.simulation_time = i*op.dt*self.dt_per_mesh
                self.transfer_forward_solution(i)
                self.setup_solver_forward_step(i)

                # Setup Hessian recovery
                if op.adapt_field == 'tracer' and op.stabilisation_tracer == 'SUPG':

                    def hessian(sol, adapt_field):
                        """
                        Cannot repeatedly project because we are not projecting a Function.
                        """
                        assert adapt_field == 'tracer'
                        return self.recover_hessian_metrics(i, adjoint=True, **hessian_kwargs)[0]

                else:
                    recoverer = self.get_recovery(i, **hessian_kwargs)

                    def hessian(sol, adapt_field):
                        fields = {'adapt_field': adapt_field, 'fields': self.fields[i]}
                        return recoverer.construct_metric(sol, **fields, **hessian_kwargs)

                def export_func_wrapper():
                    export_func()
                    strong_residuals_step.append(self.get_strong_residual(i))

                # Solve step for current mesh iteration
                solve_kwargs = {
                    'export_func': export_func_wrapper,
                    'update_forcings': update_forcings,
                    'plot_pvd': False,
                    'export_initial': False,
                }
                self.solve_forward_step(i, **solve_kwargs)
                if len(strong_residuals_step) != N-1:
                    msg = "Mismatching number of exports ({:d} vs {:d})"
                    raise ValueError(msg.format(len(strong_residuals_step), N-1))
                assert len(strong_residuals_step[0]) == len(adapt_fields)

                # --- Solve adjoint on current window

                self.transfer_adjoint_solution(i)
                self.setup_solver_adjoint_step(i)
                self.counter = 0

                def export_func():
                    """
                    Extract Hessians at each export time.
                    """
                    if self.counter < N:
                        for f, field in enumerate(adapt_fields):
                            self._H_windows[f][i][self.counter] = hessian(adj_solutions[i], field)
                    self.counter += 1

                # Solve step for current mesh iteration
                solve_kwargs = {
                    'export_func': export_func,
                    'update_forcings': None,
                    'plot_pvd': op.plot_pvd,
                    'export_initial': True,
                }
                self.solve_adjoint_step(i, **solve_kwargs)

                # Reverse order of Hessians and take pairwise averages
                for f in range(len(adapt_fields)):
                    self._H_windows[f][i] = list(reversed(self._H_windows[f][i]))
                    for j in range(N-1):
                        self._H_windows[f][i][j] = metric_average(*self._H_windows[f][i][j:j+2])
                    self._H_windows[f][i][-1] = None

                # --- Assemble indicators and metrics

                # Weight Hessians
                for j in range(N-1):
                    for f in range(len(adapt_fields)):
                        self._H_windows[f][i][j].interpolate(
                            abs(strong_residuals_step[j][f])*self._H_windows[f][i][j]
                        )

                # Combine metrics over exports for each field
                kwargs = {
                    'average': op.hessian_time_combination == 'integrate',
                    'weights': np.ones(N-1),
                }
                for f in range(len(adapt_fields)):
                    for j in range(N-2):
                        self._H_windows[f][i][j] *= w
                    if op.hessian_time_combination == 'integrate':
                        if op.timestepper == 'CrankNicolson':
                            self._H_windows[f][i][0] *= 0.5
                            self._H_windows[f][i][-2] *= 0.5
                        else:
                            raise NotImplementedError  # TODO: Other timesteppers
                    self._H_windows[f][i] = combine_metrics(*self._H_windows[f][i][:-1], **kwargs)

                # Delete objects to free memory
                self.free_solver_forward_step(i)
                self.free_solver_adjoint_step(i)

            # Normalise metrics
            self.plot_metrics(normalised=False, hessians=True)
            for H_window in self._H_windows:
                space_time_normalise(H_window, op=op)

            # Combine metrics
            self.combine_over_windows(adapt_fields)
            self.plot_metrics(normalised=True)
            self.log_complexities()

            # Adapt meshes
            self.adapt_meshes()

            # Check convergence
            if not self.minimum_adaptations_met:
                continue
            if self.elements_converged:
                break
