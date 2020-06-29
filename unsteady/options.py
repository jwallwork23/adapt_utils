from thetis import *
from thetis.configuration import *

from adapt_utils.options import Options


__all__ = ["CoupledOptions"]


# TODO: Improve doc
class CoupledOptions(Options):
    """Parameters for coupled shallow water - tracer transport model."""

    # Physics
    base_viscosity = NonNegativeFloat(0.0).tag(config=True)
    base_diffusivity = NonNegativeFloat(0.0).tag(config=True)
    base_velocity = [0.0, 0.0]
    g = FiredrakeScalarExpression(Constant(9.81)).tag(config=True)
    friction = Unicode(None, allow_none=True).tag(config=True)
    friction_coeff = NonNegativeFloat(None, allow_none=True).tag(config=True)

    # Common model
    implicitness_theta = NonNegativeFloat(0.5).tag(config=True)

    # Shallow water model
    solve_swe = Bool(True).tag(config=True)
    family = Enum(['dg-dg', 'rt-dg', 'dg-cg', 'cg-cg'], default_value='dg-dg').tag(config=True)
    grad_div_viscosity = Bool(False).tag(config=True)
    grad_depth_viscosity = Bool(False).tag(config=True)
    wetting_and_drying = Bool(False).tag(config=True)
    wetting_and_drying_alpha = FiredrakeScalarExpression(Constant(4.3)).tag(config=True)
    lax_friedrichs_velocity_scaling_factor = FiredrakeScalarExpression(None, allow_none=True).tag(config=True)
    sipg_parameter = FiredrakeScalarExpression(None, allow_none=True).tag(config=True)

    # Tracer transport model
    solve_tracer = Bool(False).tag(config=True)
    use_limiter_for_tracers = Bool(True).tag(config=True)
    tracer_family = Enum(['dg', 'cg'], default_value='dg').tag(config=True)
    lax_friedrichs_tracer_scaling_factor = FiredrakeScalarExpression(Constant(1.0)).tag(config=True)
    sipg_parameter_tracer = FiredrakeScalarExpression(None, allow_none=True).tag(config=True)
    norm_smoother = FiredrakeScalarExpression(Constant(1.0)).tag(config=True)
    tracer_advective_velocity_factor = FiredrakeScalarExpression(Constant(1.0)).tag(config=True)

    # Adaptation
    adapt_field = Unicode('all_avg', help="Adaptation field of interest.").tag(config=True)
    region_of_interest = List(default_value=[]).tag(config=True)

    def __init__(self, **kwargs):
        self.degree_increase = 0

        # Solver
        # =====
        # The time-dependent shallow water system looks like
        #
        #                             ------------------------- -----   -----
        #       ------------- -----   |                 |     | |   |   |   |
        #       | A00 | A01 | | U |   |  T + C + V + D  |  G  | | U |   | 0 |
        # A x = ------------- ----- = |                 |     | |   | = |   |  = b,
        #       | A10 | A11 | | H |   ------------------------- -----   -----
        #       ------------- -----   |        B        |  T  | | H |   | 0 |
        #                             ------------------------- -----   -----
        #
        # where:
        #  * T - time derivative;
        #  * C - Coriolis;
        #  * V - viscosity;
        #  * D - quadratic drag;
        #  * G - gravity;
        #  * B - bathymetry.
        #
        # We apply a multiplicative fieldsplit preconditioner, i.e. block Gauss-Seidel:
        #
        #     ---------------- ------------ ----------------
        #     | I |     0    | |   I  | 0 | | A00^{-1} | 0 |
        # P = ---------------- ------------ ----------------.
        #     | 0 | A11^{-1} | | -A10 | 0 | |    0     | I |
        #     ---------------- ------------ ----------------
        self.solver_parameters = {
            "shallow_water": {
                # "snes_converged_reason": None,
                "ksp_type": "gmres",
                # "ksp_converged_reason": None,
                "pc_type": "fieldsplit",
                "pc_fieldsplit_type": "multiplicative",
                "fieldsplit_U_2d": {
                    "ksp_type": "preonly",
                    "ksp_max_it": 10000,
                    "ksp_rtol": 1.0e-05,
                    "pc_type": "sor",
                    # "ksp_view": None,
                    # "ksp_converged_reason": None,
                },
                "fieldsplit_H_2d": {
                    "ksp_type": "preonly",
                    "ksp_max_it": 10000,
                    "ksp_rtol": 1.0e-05,
                    # "pc_type": "sor",
                    "pc_type": "jacobi",
                    # "ksp_view": None,
                    # "ksp_converged_reason": None,
                },
            },
            "tracer": {
                "ksp_type": "gmres",
                "pc_type": "sor",
                # "ksp_monitor": None,
                # "ksp_converged_reason": None,
            }
        }
        self.adjoint_solver_parameters.update(self.solver_parameters)
        super(CoupledOptions, self).__init__(**kwargs)

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

    def set_bathymetry(self, fs):
        """Should be implemented in derived class."""
        return Function(fs).assign(1.0)

    def set_viscosity(self, fs):
        """Should be implemented in derived class."""
        return None if np.allclose(self.base_viscosity, 0.0) else Constant(self.base_viscosity)

    def set_source_tracer(self, fs):
        """Should be implemented in derived class."""
        ero = None
        depo = None
        return ero, depo

    def set_diffusivity(self, fs):
        """Should be implemented in derived class."""
        return None if np.allclose(self.base_diffusivity, 0.0) else Constant(self.base_diffusivity)

    def set_inflow(self, fs):
        """Should be implemented in derived class."""
        return

    def set_coriolis(self, fs):
        """Should be implemented in derived class."""
        return

    def set_tracer_source(self, fs):
        """Should be implemented in derived class."""
        return

    def set_quadratic_drag_coefficient(self, fs):
        """Should be implemented in derived class."""
        return

    def set_manning_drag_coefficient(self, fs):
        if self.friction == 'manning':
            return Constant(self.friction_coeff or 0.02)

    def get_eta_tilde(self, prob, i):
        u, eta = prob.fwd_solutions[i].split()
        if not self.wetting_and_drying:
            return eta
        bathymetry_displacement = prob.equations[i].shallow_water.depth.wd_bathymetry_displacement
        return eta + bathymetry_displacement(eta)

    def get_export_func(self, prob, i):
        if self.wetting_and_drying:
            eta_tilde = Function(prob.P1DG[i], name="Modified elevation")
            self.eta_tilde_file._topology = None

            def export_func():
                eta_tilde.project(self.get_eta_tilde(prob, i))
                self.eta_tilde_file.write(eta_tilde)

            return export_func
