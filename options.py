from __future__ import absolute_import
from thetis import *
from thetis.configuration import *

import os
import sys

from . import misc
from .mesh import MeshStats


__all__ = ["Options", "CoupledOptions", "ReynoldsNumberArray"]


class Options(FrozenConfigurable):
    name = 'Common parameters for mesh adaptive simulations'

    # --- Solvers

    # Spatial discretisation
    family = Enum(
        ['dg-dg', 'dg-cg', 'cg-cg'],
        default_value='dg-dg',
        help="""
        Mixed finite element pair to use for the hydrodynamics system. Choose from:
          'cg-cg': Taylor-Hood                    (P2-P1);
          'dg-dg': Equal order DG                 (PpDG-PpDG);
          'dg-cg': Mixed continuous-discontinuous (P1DG-P2),
        where p is the polynomial order specified by :attr:`degree`.""").tag(config=True)
    degree = NonNegativeInteger(1, help="""
        Polynomial order for hydrodynamics finite element pair :attr:`family'.""").tag(config=True)
    degree_increase = NonNegativeInteger(0, help="""
        When defining an enriched hydrodynamics finite element space, how much should the
        polynomial order of the finite element space by incremented? (NOTE: zero is an option)
        """).tag(config=True)
    periodic = Bool(False, help="Is mesh periodic?").tag(config=True)

    # Time discretisation
    timestepper = Enum(
        ['SteadyState', 'CrankNicolson'],
        default_value='CrankNicolson',
        help="Time integration scheme used.").tag(config=True)
    dt = PositiveFloat(0.1, help="Timestep").tag(config=True)
    start_time = NonNegativeFloat(0.0, help="Start of time window of interest.").tag(config=True)
    end_time = PositiveFloat(60.0, help="End of time window of interest.").tag(config=True)
    num_meshes = PositiveInteger(1, help="""
        Number of meshes in :class:`AdaptiveProblem` solver""").tag(config=True)
    dt_per_export = PositiveFloat(10, help="Number of timesteps per export.").tag(config=True)
    dt_per_mesh_movement = PositiveFloat(1, help="Number of timesteps per mesh movement.").tag(config=True)
    use_semi_implicit_linearisation = Bool(False, help="""
        Toggle whether or not to linearise implicit terms. This is generally recommended if Manning
        friction is to be included.""").tag(config=True)
    implicitness_theta = NonNegativeFloat(0.5, help=r"""
        Consider an ODE

      ..math::
            u_t = f(u), u(0) = u_0.

        Crank-Nicolson can be represented as

      ..math::
            \frac{u^{(n+1)} - u^{(n)})}{\Delta t} = \theta f(u^{(n+1)}) + (1-\theta) f(u^{(n)}),

        where usually :math:`\theta=\frac12`. This parameter allows to vary this parameter between
        0 and 1, with 0 corresponding to Forward Euler and 1 corresponding to Backward Euler.
        """).tag(config=True)

    # Boundary conditions
    boundary_conditions = PETScSolverParameters({}, help="""
        Boundary conditions expressed as a dictionary.
        """).tag(config=True)
    adjoint_boundary_conditions = PETScSolverParameters({}, help="""
        Boundary conditions for continuous adjoint problem expressed as a dictionary.
        """).tag(config=True)

    # Stabilisation
    stabilisation = Unicode(None, allow_none=True, help="""
        Stabilisation approach to use for hydrodynamic model.
        """).tag(config=True)
    anisotropic_stabilisation = Bool(False, help="""
        Account for mesh anisotropy by using an alternative cell size measure to `CellSize`.
        """).tag(config=True)
    use_automatic_sipg_parameter = Bool(True, help="""
        Toggle automatic generation of symmetric interior penalty method.""").tag(config=True)

    # Solver parameters
    solver_parameters = PETScSolverParameters({}, help="""
        Solver parameters for the forward model, separated by equation set.""").tag(config=True)
    adjoint_solver_parameters = PETScSolverParameters({}, help="""
        Solver parameters for the adjoint models, separated by equation set.""").tag(config=True)

    # Outputs
    debug = Bool(False, help="Toggle debugging for more verbose screen output.").tag(config=True)
    debug_mode = Enum(
        ['basic', 'full'],
        default_value='basic',
        help="""Choose debugging mode from {'basic', 'full'}.""").tag(config=True)
    plot_pvd = Bool(True, help="Toggle saving fields to .pvd and .vtu.").tag(config=True)
    plot_bathymetry = Bool(False, help="Toggle plotting bathymetry to .pvd and .vtu.").tag(config=True)
    save_hdf5 = Bool(False, help="Toggle saving fields to HDF5.").tag(config=True)

    # --- Adaptation

    approach = Unicode('fixed_mesh', help="Mesh adaptive approach.").tag(config=True)

    # Metric based
    rescaling = PositiveFloat(0.85, help="""
        Scaling parameter for target number of vertices.
        """).tag(config=True)  # TODO: UNUSED?
    convergence_rate = PositiveFloat(6, help="""
        Convergence rate parameter used in approach of [Carpio et al. 2013].
        """).tag(config=True)
    h_min = PositiveFloat(1.0e-10, help="Minimum tolerated element size.").tag(config=True)
    h_max = PositiveFloat(5.0e+00, help="Maximum tolerated element size.").tag(config=True)
    max_anisotropy = PositiveFloat(1.0e+03, help="Maximum tolerated anisotropy.").tag(config=True)
    normalisation = Unicode('complexity', help="""
        Metric normalisation approach, from {'complexity', 'error'}.
        """).tag(config=True)
    target = PositiveFloat(1.0e+2, help="""
        Target complexity / inverse desired error for normalisation, as appropriate.
        """).tag(config=True)
    norm_order = NonNegativeFloat(None, allow_none=True, help="""
        Degree p of Lp norm used in spatial normalisation. Use `None` to specify infinity norm.
        """).tag(config=True)
    intersect_boundary = Bool(False, help="Intersect with initial boundary metric.").tag(config=True)

    # Mesh movement
    pseudo_dt = PositiveFloat(0.1, help="Pseudo-timstep used in r-adaptation.").tag(config=True)
    r_adapt_maxit = PositiveInteger(1000, help="""
        Maximum number of iterations in r-adaptation loop.
        """).tag(config=True)
    r_adapt_rtol = PositiveFloat(1.0e-8, help="""
        Relative tolerance for residual in r-adaptation loop.
        """).tag(config=True)
    nonlinear_method = Enum(['quasi_newton', 'relaxation'], default_value='quasi_newton', help="""
        Method for solving nonlinear system under Monge-Ampere mesh movement.
        """).tag(config=True)

    # Recovery
    gradient_recovery = Enum(['L2', 'ZZ'], default_value='L2', help="""
        Hessian recovery technique, from:
          'L2':    global L2 projection;
          'ZZ':    recovery a la [Zienkiewicz and Zhu 1987].
        """).tag(config=True)
    gradient_solver_parameters = PETScSolverParameters(
        {
            'snes_rtol': 1e8,
            'ksp_rtol': 1e-5,
            'ksp_gmres_restart': 20,
            'pc_type': 'sor',
        },
        help="Solver parameters for gradient recovery.").tag(config=True)
    hessian_recovery = Enum(['parts', 'L2', 'ZZ'], default_value='L2', help="""
        Hessian recovery technique, from:
          'L2':    global double L2 projection;
          'parts': direct application of integration by parts;
          'ZZ':    recovery a la [Zienkiewicz and Zhu 1987].
        """).tag(config=True)
    hessian_solver_parameters = PETScSolverParameters(
        {
            # Integration by parts
            'parts': {

                # GMRES with restarts
                'ksp_type': 'gmres',
                'ksp_gmres_restart': 20,
                'ksp_rtol': 1.0e-05,

                # SOR preconditioning
                'pc_type': 'sor',
            },

            # Double L2 projection
            'L2': {
                'mat_type': 'aij',

                # Use stationary preconditioners in the Schur complement, to get away with applying
                # GMRES to the whole mixed system
                'ksp_type': 'gmres',
                'pc_type': 'fieldsplit',
                'pc_fieldsplit_type': 'schur',

                # We want to eliminate H (field 1) to get an equation for g (field 0)
                'pc_fieldsplit_0_fields': '1',
                'pc_fieldsplit_1_fields': '0',

                # Use a diagonal approximation of the A00 block.
                'pc_fieldsplit_schur_precondition': 'selfp',

                # Use ILU to approximate the inverse of A00, without a KSP solver
                'fieldsplit_0_pc_type': 'ilu',
                'fieldsplit_0_ksp_type': 'preonly',

                # Use GAMG to approximate the inverse of the Schur complement matrix
                'fieldsplit_1_ksp_type': 'preonly',
                'fieldsplit_1_pc_type': 'gamg',
                'ksp_max_it': 20,
            }
        },
        help="Solver parameters for Hessian recovery.").tag(config=True)
    hessian_time_combination = Enum(
        ['integrate', 'intersect'],
        default_value='integrate',
        help="Method used to combine Hessians over timesteps.").tag(config=True)
    hessian_timestep_lag = PositiveFloat(1, help="""
        Allow lagged Hessian computation by setting greater than one.
        """).tag(config=True)

    # Goal-oriented adaptation
    region_of_interest = List(default_value=[], help="""
        Spatial region related to quantity of interest.
        """).tag(config=True)
    solve_enriched_forward = Bool(False, help="""
        Toggle whether the forward problem should be solved in the enriched space or simply
        prolonged from the base space for nonlinear problems.
        """).tag(config=True)
    enrichment_method = Enum(['GE_hp', 'GE_h', 'GE_p', 'PR', 'DQ'], default_value='GE_h', help="""
        Method used to construct an enriched space for the higher order approximation of the
        adjoint and/or forward error term in the DWR residual.

        Options:
          * 'GE_hp': global h-refinement and global p-refinement;
          * 'GE_h' : global h-refinement alone;
          * 'GE_p' : global p-refinement alone;
          * 'PR'   : patch recovery;
          * 'DQ'   : difference quotients.
        """).tag(config=True)

    # Adaptation loop
    min_adapt = NonNegativeInteger(0, help="""
        Minimum number of mesh adaptations in outer loop.
        """).tag(config=True)
    max_adapt = NonNegativeInteger(4, help="""
        Maximum number of mesh adaptations in outer loop.
        """).tag(config=True)
    element_rtol = PositiveFloat(0.005, help="""
        Relative tolerance for convergence in mesh element count
        """).tag(config=True)
    qoi_rtol = PositiveFloat(0.005, allow_none=True, help="""
        Relative tolerance for convergence in quantity of interest.
        """).tag(config=True)
    estimator_rtol = PositiveFloat(0.005, allow_none=True, help="""
        Relative tolerance for convergence in error estimator.
        """).tag(config=True)
    target_base = PositiveFloat(10.0, help="""
        Base for exponential increase/decay of target complexity/error within outer mesh adaptation
        loop.
        """).tag(config=True)
    outer_iterations = PositiveInteger(1, help="""
        Number of iterations in outer adaptation loop.""").tag(config=True)
    indent = Unicode('', help="Indent used in nested print statements.").tag(config=True)

    def __init__(self, mesh=None, fpath=None, **kwargs):
        """
        Upon initialising the class, any kwargs will be added and a output directory path will be
        created as determined by :attr:`approach` as `outputs/<approach>/`.

        :kwarg mesh: a mesh to use as the :attr:`default_mesh` instead of the default one defined in
            the subclass.
        :fpath: optional extension to the usual `outputs/<approach>/` output directory path.
        """
        self.default_mesh = mesh
        self.update(kwargs)
        self.di = os.path.join('outputs', self.approach)
        if fpath is not None:
            self.di = os.path.join(self.di, fpath)
        self.di = create_directory(self.di)
        if self.debug:
            if self.debug_mode == 'basic':
                set_log_level(INFO)
            else:
                set_log_level(DEBUG)

    def set_all_rtols(self, tol):
        """Set all relative tolerances to a single value, `tol`."""
        self.element_rtol = tol
        self.qoi_rtol = tol
        self.estimator_rtol = tol

    # TODO: Collapse indicators to one function and include type in RoI and source specifications

    def box(self, mesh, source=False, **kwargs):
        locs = self.source_loc if source else self.region_of_interest
        return misc.box(locs, mesh, **kwargs)

    def ball(self, mesh, source=False, **kwargs):  # TODO: ellipse
        locs = self.source_loc if source else self.region_of_interest
        return misc.ellipse(locs, mesh, **kwargs)

    def bump(self, mesh, source=False, **kwargs):
        locs = self.source_loc if source else self.region_of_interest
        return misc.bump(locs, mesh, **kwargs)

    def circular_bump(self, mesh, source=False, **kwargs):
        locs = self.source_loc if source else self.region_of_interest
        return misc.circular_bump(locs, mesh, **kwargs)

    def gaussian(self, mesh, source=False, **kwargs):
        locs = self.source_loc if source else self.region_of_interest
        return misc.gaussian(locs, mesh, **kwargs)

    def set_start_condition(self, fs, adjoint=False):
        return self.set_terminal_condition(fs) if adjoint else self.set_initial_condition(fs)

    def set_initial_condition(self, fs):
        raise NotImplementedError("Should be implemented in derived class.")

    def set_terminal_condition(self, fs):
        raise NotImplementedError("Should be implemented in derived class.")

    def set_boundary_conditions(self, fs):
        raise NotImplementedError("Should be implemented in derived class.")

    def set_boundary_surface(self):  # TODO: surely it needs an arg
        raise NotImplementedError("Should be implemented in derived class.")

    def set_source(self, fs):
        raise NotImplementedError("Should be implemented in derived class.")

    def set_qoi_kernel(self, fs):
        raise NotImplementedError("Should be implemented in derived class.")

    def set_qoi_kernel_tracer(self, fs):
        raise NotImplementedError("Should be implemented in derived class.")

    def analytical_solution(self, fs):
        raise NotImplementedError("Should be implemented in derived class.")

    def analytical_qoi(self):
        raise NotImplementedError("Should be implemented in derived class.")

    def get_update_forcings(self, prob, i, **kwargs):
        """Should be implemented in derived class."""
        def update_forcings(t):
            return
        return update_forcings

    def get_export_func(self, prob, i, **kwargs):
        """Should be implemented in derived class."""
        def export_func():
            return
        return export_func

    def print_debug(self, msg, mode='basic'):
        """
        Print a string `msg` only if debugging is on.

        :kwarg mode: if 'full' is specified, the debugging statement will only be printed if
            :attr:`debug_mode` is set to 'full'.
        """
        if not self.debug:
            return
        if mode == 'full' and self.debug_mode == 'basic':
            return
        try:
            print_output(self.indent + msg)
        except TypeError:
            print(msg)

    def copy(self):
        op = self.__class__()
        op.update(self)
        return op


# TODO: Improve doc
class CoupledOptions(Options):
    """
    Parameters for the coupled system. Selection from the four model components may be achieved using
    the following flags:
      * `solve_swe`      - shallow water model (hydrodynamics);
      * `solve_tracer`   - passive tracer transport model;
      * `solve_sediment` - sediment model;
      * `solve_exner`    - Exner equation.
    """
    qoi_quadrature_degree = NonNegativeInteger(12, help="""
        Quadrature degree to use for quantity of interest.
        """).tag(config=True)

    # Physics
    base_viscosity = NonNegativeFloat(0.0, help="""
        Non-negative value providing the default constant viscosity field.
        """).tag(config=True)
    min_viscosity = NonNegativeFloat(0.0, help="""
        Non-negative value providing the minimum tolerated viscosity.
        """).tag(config=True)
    base_diffusivity = NonNegativeFloat(0.0, help="""
        Non-negative value providing the default constant diffusivity field.
        """).tag(config=True)
    base_bathymetry = PositiveFloat(1.0, help="""
        Positive value providing the default bathymetry field.
        """).tag(config=True)
    base_velocity = List([0.0, 0.0], help="""
        Two element list providing the default constant velocity field.
        """).tag(config=True)
    g = FiredrakeScalarExpression(Constant(9.81), help="""
        Non-negative value providing the default constant gravitational acceleration.
        """).tag(config=True)
    friction = Unicode(None, allow_none=True, help="""
        Friction parametrisation for the drag term. Choose from {'nikuradse', 'manning'}.
        """).tag(config=True)
    friction_coeff = NonNegativeFloat(None, allow_none=True, help="""
        Non-negative value providing the default constant drag parameter.
        """).tag(config=True)

    # Shallow water model
    solve_swe = Bool(True, help="Toggle solving the shallow water model.").tag(config=True)
    grad_div_viscosity = Bool(False).tag(config=True)  # TODO: help
    grad_depth_viscosity = Bool(False).tag(config=True)  # TODO: help
    wetting_and_drying = Bool(False, help="""
        Toggle use of Thetis' wetting-and-drying routine.
        """).tag(config=True)
    wetting_and_drying_alpha = FiredrakeScalarExpression(Constant(4.3), help="""
        Scalar parameter used in Thetis' wetting-and-drying routine.
        """).tag(config=True)
    stabilisation = Unicode(None, allow_none=True, help="""
        Stabilisation approach for shallow water model, either 'lax_friedrichs'} or None.
        """).tag(config=True)
    lax_friedrichs_velocity_scaling_factor = FiredrakeConstantTraitlet(Constant(1.0), help="""
        Scaling factor for Lax Friedrichs stabilisation term in horizontal momentum advection.
        """).tag(config=True)
    sipg_parameter = FiredrakeScalarExpression(None, allow_none=True, help="""
        Optional user-provided symemetric interior penalty parameter for the shallow water model.
        Can also be set automatically using :attr:`use_automatic_sipg_parameter`.
        """).tag(config=True)
    recover_vorticity = Bool(False, help="""
        If True, a vorticity field is L2-projected from the hydrodynamics velocity output.
        """).tag(config=True)
    characteristic_velocity = FiredrakeVectorExpression(None, allow_none=True, help="""
        Characteristic velocity value to use in Reynolds and Peclet number calculations. Typically,
        this is set using a `Constant`, but it could also be spatially varying.
        """)
    characteristic_speed = FiredrakeScalarExpression(None, allow_none=True, help="""
        Characteristic speed value used in SU/SUPG stabilisation.
        """)
    characteristic_diffusion = FiredrakeScalarExpression(None, allow_none=True, help="""
        Characteristic diffusion value used in SU/SUPG stabilisation.
        """)

    # Tracer transport model
    solve_tracer = Bool(False, help="Toggle solving the tracer transport model.").tag(config=True)
    use_limiter_for_tracers = Bool(True, help="""
        Toggle using vertex-based slope limiters for the tracer transport model.
        """).tag(config=True)
    use_tracer_conservative_form = Bool(False, help="""
        Toggle whether to solve the conservative or non-conservative form of the tracer transport
        mode.
        """).tag(config=True)
    tracer_family = Enum(
        ['dg', 'cg'],
        default_value='dg',
        help="""
        Finite element pair to use for the tracer transport model. Choose from:
          'cg': Continuous Galerkin    (Pp);
          'dg': Discontinuous Galerkin (PpDG),
        where p is the polynomial order specified by :attr:`degree_tracer`.""").tag(config=True)
    degree_tracer = NonNegativeInteger(1, help="""
        Polynomial order for tracer finite element pair :attr:`tracer_family'.
        """).tag(config=True)
    degree_increase_tracer = NonNegativeInteger(0, help="""
        When defining an enriched tracer finite element space, how much should the
        polynomial order of the finite element space by incremented? (NOTE: zero is an option)
        """).tag(config=True)
    stabilisation_tracer = Unicode(None, allow_none=True, help="""
        Stabilisation approach for tracer model, chosen from {'SU', 'SUPG', 'lax_friedrichs'}, if
        not None.
        """).tag(config=True)
    lax_friedrichs_tracer_scaling_factor = FiredrakeScalarExpression(Constant(1.0), help="""
        Scaling factor for Lax Friedrichs stabilisation term in tracer advection.
        """).tag(config=True)
    sipg_parameter_tracer = FiredrakeScalarExpression(None, allow_none=True, help="""
        Optional user-provided symemetric interior penalty parameter for the tracer model.
        Can also be set automatically using :attr:`use_automatic_sipg_parameter`.
        """).tag(config=True)
    norm_smoother = FiredrakeScalarExpression(Constant(0.0)).tag(config=True)  # TODO: help
    tracer_advective_velocity_factor = FiredrakeScalarExpression(Constant(1.0)).tag(config=True)  # TODO: help

    # Sediment model
    solve_sediment = Bool(False, help="Toggle solving the sediment model.").tag(config=True)
    sediment_family = Unicode('dg', help="""
        Finite element pair to use for the sediment transport model. Choose from:
          'cg': Continuous Galerkin    (Pp);
          'dg': Discontinuous Galerkin (PpDG),
        where p is the polynomial order specified by :attr:`degree_sediment`.""").tag(config=True)
    degree_sediment = NonNegativeInteger(1, help="""
        Polynomial order for sediment finite element pair :attr:`sediment_family'.
        """).tag(config=True)
    degree_increase_sediment = NonNegativeInteger(0, help="""
        When defining an enriched sediment finite element space, how much should the
        polynomial order of the finite element space by incremented? (NOTE: zero is an option)
        """).tag(config=True)
    stabilisation_sediment = Unicode(None, allow_none=True, help="""
        Stabilisation approach for sediment model, set to 'lax_friedrichs', if not None.
        """).tag(config=True)
    sipg_parameter_sediment = FiredrakeScalarExpression(None, allow_none=True, help="""
        Optional user-provided symemetric interior penalty parameter for the sediment model.
        Can also be set automatically using :attr:`use_automatic_sipg_parameter`.
        """).tag(config=True)
    ksp = FiredrakeScalarExpression(None, allow_none=True).tag(config=True)  # TODO: help

    # Exner transport model
    solve_exner = Bool(False, help="Toggle solving the Exner model.").tag(config=True)
    bathymetry_family = Enum(['dg', 'cg'], default_value='cg', help="""
        Finite element space to use for the Exner model. Choose from {'cg', 'dg'}.
        """).tag(config=True)
    degree_bathymetry = NonNegativeInteger(1, help="""
        Polynomial order for tracer finite element pair :attr:`tracer_family'.
        """).tag(config=True)
    morphological_acceleration_factor = FiredrakeScalarExpression(Constant(1.0)).tag(config=True)  # TODO: help
    porosity = FiredrakeScalarExpression(Constant(0.4)).tag(config=True)  # TODO: help

    # Adaptation
    adapt_field = Unicode('all_avg', help="""
        Adaptation field of interest. Commonly used values include individual scalar fields, such as
        'speed', 'elevation', 'velocity_x', 'velocity_y', 'bathymetry', as well as combined versions,
        which can be constructed from the individual fields using double underscores with either
        'avg' or 'int' in the middle. These relate to the way in which the metrics arising from the
        adaptation fields are to be combined. The former stands for metric averaging and the latter
        stands for metric intersection.

        e.g. 'elevation__int__speed'.

        The special cases of 'all_avg' and 'all_int' correspond to averaging and intersecting
        metrics arising from the three primary scalar hydrodynamics fields. That is, the former is
        equivalent to:

        'elevation__avg__velocity_x__avg__velocity_y'.

        Note that while metric averaging is commutative, metric intersection is not (although the
        differences due to ordering are negligible in practice).
        """).tag(config=True)
    region_of_interest = List(default_value=[], help="""
       A list of tuples whose first n entries determine a spatial position (in n dimensions) which
       defines the centre of the region and whose later entries determine the dimensions of the
       region. For further details, see the :attr:`ball`, :attr:`box`, etc. methods.
       """).tag(config=True)

    def __init__(self, **kwargs):
        """
        Solver
        =====
        The time-dependent shallow water system looks like

                                    ------------------------- -----   -----
              ------------- -----   |                 |     | |   |   |   |
              | A00 | A01 | | U |   |  T + C + V + D  |  G  | | U |   | 0 |
        A x = ------------- ----- = |                 |     | |   | = |   |  = b,
              | A10 | A11 | | H |   ------------------------- -----   -----
              ------------- -----   |        B        |  T  | | H |   | 0 |
                                    ------------------------- -----   -----

        where:
         * T - time derivative;
         * C - Coriolis;
         * V - viscosity;
         * D - quadratic drag;
         * G - gravity;
         * B - bathymetry.

        We apply a multiplicative fieldsplit preconditioner, i.e. block Gauss-Seidel:

            ---------------- ------------ ----------------
            | I |     0    | |   I  | 0 | | A00^{-1} | 0 |
        P = ---------------- ------------ ----------------.
            | 0 | A11^{-1} | | -A10 | 0 | |    0     | I |
            ---------------- ------------ ----------------
        """
        self.default_solver_parameters = {
            "shallow_water": {
                "ksp_type": "gmres",
                "pc_type": "fieldsplit",
                "pc_fieldsplit_type": "multiplicative",
                "fieldsplit_U_2d": {
                    "ksp_type": "preonly",
                    "ksp_max_it": 10000,
                    "ksp_rtol": 1.0e-05,
                    "pc_type": "sor",
                },
                "fieldsplit_H_2d": {
                    "ksp_type": "preonly",
                    "ksp_max_it": 10000,
                    "ksp_rtol": 1.0e-05,
                    # "pc_type": "sor",
                    "pc_type": "jacobi",
                },
            },
            "tracer": {
                "ksp_type": "gmres",
                "pc_type": "sor",
            },
            "sediment": {
                "ksp_type": "gmres",
                "pc_type": "sor",
            },
            "exner": {
                "ksp_type": "gmres",
                "pc_type": "sor",
            }
        }
        self.solver_parameters = self.default_solver_parameters
        self.adjoint_solver_parameters.update(self.solver_parameters)
        super(CoupledOptions, self).__init__(**kwargs)

        # Check setup
        if not np.any([self.solve_swe, self.solve_tracer, self.solve_sediment, self.solve_exner]):
            print_output("No equation set specified.")
            sys.exit(0)
        if self.solve_tracer and self.solve_sediment:
            raise NotImplementedError("Model does not support both tracers and sediment.")
        if self.solve_exner and not self.solve_sediment:
            raise NotImplementedError("Model does not support Exner without sediment.")

        # Metadata
        self.solve_flags = (
            self.solve_swe,
            self.solve_tracer,
            self.solve_sediment,
            self.solve_exner,
        )
        self.solve_fields = (
            "shallow_water",
            "tracer",
            "sediment",
            "bathymetry",
        )

    def set_initial_condition(self, prob):
        u, eta = prob.fwd_solutions[0].split()
        u.interpolate(as_vector(self.base_velocity))

    def set_initial_condition_tracer(self, prob):
        prob.fwd_solutions_tracer[0].assign(0.0)

    def set_terminal_condition(self, prob):
        z, zeta = prob.adj_solutions[-1].split()
        z.assign(0.0)
        zeta.assign(0.0)

    def set_terminal_condition_tracer(self, prob):
        prob.adj_solutions_tracer[0].assign(0.0)

    def set_terminal_condition_sediment(self, prob):
        prob.adj_solutions_sediment[0].assign(0.0)

    def set_terminal_condition_exner(self, prob):
        prob.adj_solutions_exner[0].assign(0.0)

    def set_bathymetry(self, fs):
        """Should be implemented in derived class."""
        return Function(fs).assign(self.base_bathymetry)

    def set_advective_velocity_factor(self, fs):
        """Should be implemented in derived class."""
        return Constant(1.0)

    def set_viscosity(self, fs):
        """Should be implemented in derived class."""
        return None if np.isclose(self.base_viscosity, 0.0) else Constant(self.base_viscosity)

    def set_diffusivity(self, fs):
        """Should be implemented in derived class."""
        return None if np.isclose(self.base_diffusivity, 0.0) else Constant(self.base_diffusivity)

    def set_inflow(self, fs):
        """Should be implemented in derived class."""
        return

    def set_coriolis(self, fs):
        """Should be implemented in derived class."""
        return

    def set_tracer_source(self, fs):
        """Should be implemented in derived class."""
        return

    def set_sediment_source(self, fs):
        """Should be implemented in derived class."""
        return

    def set_sediment_sink(self, sediment_model, fs):
        """Should be implemented in derived class."""
        return

    def set_sediment_depth_integ_source(self, fs):
        """Should be implemented in derived class."""
        return

    def set_sediment_depth_integ_sink(self, fs):
        """Should be implemented in derived class."""
        return

    def set_quadratic_drag_coefficient(self, fs):
        """Should be implemented in derived class."""
        return

    def set_manning_drag_coefficient(self, fs):
        if self.friction == 'manning':
            return Constant(self.friction_coeff or 0.02)

    def get_velocity(self, t):
        raise NotImplementedError("Should be implemented in derived class.")

    def get_eta_tilde(self, prob, i):
        u, eta = prob.fwd_solutions[i].split()
        if not self.wetting_and_drying:
            return eta
        bathymetry_displacement = prob.equations[i].shallow_water.depth.wd_bathymetry_displacement
        return eta + bathymetry_displacement(eta)

    def get_export_func(self, prob, i, **kwargs):
        if self.wetting_and_drying:
            eta_tilde = Function(prob.P1DG[i], name="Modified elevation")
            self.eta_tilde_file._topology = None

        def export_func():
            if self.wetting_and_drying:
                eta_tilde.project(self.get_eta_tilde(prob, i))
                self.eta_tilde_file.write(eta_tilde)

        return export_func

    def check_mesh_reynolds_number(self, nu, characteristic_velocity=None, mesh=None, index=None):
        """
        Compute the mesh Reynolds number using provided characteristic velocity, viscosity field
        and mesh.
        """
        if nu is None:
            Re_h = None
            self.print_debug("INIT: Cannot compute mesh Reynolds number for inviscid problems")
            return None, None, None
        u = characteristic_velocity or self.characteristic_velocity
        if u is None:
            raise ValueError("Cannot enforce mesh Reynolds number without characteristic velocity!")
        if index is None:
            self.print_debug("INIT: Computing mesh Reynolds number...")
        else:
            self.print_debug("INIT: Computing Reynolds number on mesh {:d}...".format(index))

        # Get local mesh element size
        if mesh is None:
            if isinstance(u, Function):
                mesh = u.function_space().mesh()
            elif isinstance(nu, Function):
                mesh = nu.function_space().mesh()
            else:
                raise ValueError("Cannot compute mesh Reynolds number without a mesh!")
        stats = MeshStats(self, mesh)  # TODO: Build into solver

        # Compute elementwise mesh Reynolds number
        fs = nu.function_space() if isinstance(nu, Function) else FunctionSpace(mesh, "CG", 1)
        # fs = stats._P0
        Re_h = Function(fs, name="Reynolds number")
        Re_h.project(stats.dx*sqrt(dot(u, u))/nu)
        # Re_h.interpolate(stats.dx*sqrt(dot(u, u))/nu)
        Re_h_vec = Re_h.vector().gather()
        Re_h_min = Re_h_vec.min()
        Re_h_max = Re_h_vec.max()
        Re_h_mean = np.mean(Re_h_vec)

        # Print to screen and return
        lg = lambda x: '<' if x < 1 else '>'
        msg = "INIT:   min(Re_h)  = {:11.4e} {:1s} 1"
        self.print_debug(msg.format(Re_h_min, lg(Re_h_min)))
        msg = "INIT:   max(Re_h)  = {:11.4e} {:1s} 1"
        self.print_debug(msg.format(Re_h_max, lg(Re_h_max)))
        msg = "INIT:   mean(Re_h) = {:11.4e} {:1s} 1"
        self.print_debug(msg.format(Re_h_mean, lg(Re_h_mean)))
        return Re_h, Re_h_min, Re_h_max

    def enforce_mesh_reynolds_number(self, fs, target, characteristic_velocity=None, index=None):
        """
        Enforce the mesh Reynolds number specified by :attr:`target_mesh_reynolds_number`.
        Also needs a characteristic velocity (either passed as a keyword argument, or read from
        :attr:`characteristic_velocity`) and a :class:`FunctionSpace`.
        """
        Re_h = target
        if Re_h is None:
            raise ValueError("Cannot enforce mesh Reynolds number for inviscid problems!")
        u = characteristic_velocity or self.characteristic_velocity
        if u is None:
            raise ValueError("Cannot enforce mesh Reynolds number without characteristic velocity!")
        if index is None:
            msg = "INIT: Enforcing mesh Reynolds number {:.4e}..."
            self.print_debug(msg.format(Re_h))
        else:
            msg = "INIT: Enforcing Reynolds number {:.4e} on mesh {:d}..."
            self.print_debug(msg.format(Re_h, index))

        # Get local mesh element size
        stats = MeshStats(self, fs.mesh())  # TODO: Build into solver
        # fs = stats._P0

        # Compute viscosity which yields target mesh Reynolds number
        nu = Function(fs, name="Horizontal viscosity")
        nu.project(stats.dx*sqrt(dot(u, u))/Re_h)
        # nu.interpolate(stats.dx*sqrt(dot(u, u))/Re_h)
        return nu

    def increase_degree(self, adapt_field):
        if adapt_field == 'tracer':
            assert self.degree_increase_tracer != 0
            self.degree_tracer += self.degree_increase_tracer
        elif adapt_field == 'sediment':
            assert self.degree_increase_sediment != 0
            self.degree_sediment += self.degree_increase_sediment
        elif adapt_field == 'bathymetry':
            assert self.degree_increase_bathymetry != 0
            self.degree_bathymetry += self.degree_increase_bathymetry
        else:
            assert self.degree_increase != 0
            self.degree += self.degree_increase


class ReynoldsNumberArray(object):
    """
    Custom array object to hold values of the Renolds number on a sequence of meshes. Value
    assignment to key `i` has been overriden to take a (velocity, viscosity) tuple, which will be
    used to compute the Reynolds number on the ith mesh.
    """
    def __init__(self, meshes, op):
        self._meshes = meshes
        self._data = [None for mesh in self._meshes]
        self._min = [None for mesh in self._meshes]
        self._max = [None for mesh in self._meshes]
        self._op = op

    def __getitem__(self, i):
        return self._data[i]

    def __setitem__(self, i, value):
        (u, nu) = value
        mesh = self._meshes[i]
        Re_h, Re_h_min, Re_h_max = self._op.check_mesh_reynolds_number(nu, u, mesh=mesh, index=i)
        self._data[i] = Re_h
        self._min[i] = Re_h_min
        self._max[i] = Re_h_max

    def min(self, i):
        return self._min[i]

    def max(self, i):
        return self._max[i]
