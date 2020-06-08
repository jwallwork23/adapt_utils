from thetis import *
from thetis.configuration import *

from adapt_utils.options import Options


__all__ = ["ShallowWaterOptions"]


class ShallowWaterOptions(Options):
    """
    Parameters for shallow water solver.
    """
    solve_tracer = Bool(False).tag(config=True)
    solve_swe = Bool(True).tag(config=True)

    # Physical
    base_viscosity = NonNegativeFloat(0.0).tag(config=True)
    base_diffusivity = NonNegativeFloat(0.0).tag(config=True)
    g = FiredrakeScalarExpression(Constant(9.81)).tag(config=True)

    # Model
    grad_div_viscosity = Bool(False).tag(config=True)
    grad_depth_viscosity = Bool(False).tag(config=True)
    family = Enum(['dg-dg', 'rt-dg', 'dg-cg', 'cg-cg'], default_value='dg-dg').tag(config=True)
    wetting_and_drying = Bool(False).tag(config=True)
    wetting_and_drying_alpha = FiredrakeScalarExpression(Constant(4.3)).tag(config=True)
    implicitness_theta = NonNegativeFloat(0.5).tag(config=True)

    # Adaptation
    adapt_field = Unicode('all_avg', help="Adaptation field of interest.").tag(config=True)
    region_of_interest = List(default_value=[]).tag(config=True)

    def __init__(self, **kwargs):
        self.degree_increase = 0
        super(ShallowWaterOptions, self).__init__(**kwargs)

    def set_bathymetry(self, fs):
        raise NotImplementedError("Should be implemented in derived class.")

    def set_viscosity(self, fs):
        """Should be implemented in derived class."""
        if np.allclose(self.base_viscosity, 0.0):
            viscosity = None
        else:
            viscosity = Constant(self.base_viscosity)
        return viscosity

    def set_source_tracer(self, fs):
        """Should be implemented in derived class."""
        # return self.depo, self.ero  # TODO
        return

    def set_diffusivity(self, fs):
        """Should be implemented in derived class."""
        if np.allclose(self.base_diffusivity, 0.0):
            diffusivity = None
        else:
            diffusivity = Constant(self.base_diffusivity)
        return diffusivity

    def set_inflow(self, fs):
        """Should be implemented in derived class."""
        return

    def set_coriolis(self, fs):
        """Should be implemented in derived class."""
        return

    def set_quadratic_drag_coefficient(self, fs):
        """Should be implemented in derived class."""
        return

    def set_manning_drag_coefficient(self, fs):
        """Should be implemented in derived class."""
        return

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
